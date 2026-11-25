# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Local Quarry scanner for memory operations and pre-tool flow checks."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from pathlib import Path
from time import perf_counter
from typing import Any

from quarry import DSLScanner, ModelInferenceBackend, SafetyLimits, ScanResult
from quarry.config import QuarryConfig

from ..events import MemoryMutation, MemoryRead
from .inputs import (
    MAX_DEPTH,
    MAX_TOOL_ARGUMENT_BYTES,
    MEMORY_V1_INPUTS,
    InputBuilder,
    MemoryReadEvent,
)
from .profiles import profile_rules

_SCAN_DEADLINE_SECONDS = 3.0
_SAFETY_LIMITS = SafetyLimits(
    rule_timeout=1.5,
    max_total_size=5_000_000,
    max_tool_argument_size=MAX_TOOL_ARGUMENT_BYTES,
    max_nesting_depth=MAX_DEPTH,
    max_timeline_items=128,
    max_model_calls=4,
)


def _require_declared_inputs() -> None:
    # Quarry 0.1.1 on PyPI predates declared inputs but shares its version number
    # with later builds, so check the feature rather than the version.
    if "inputs" not in {item.name for item in fields(QuarryConfig)}:
        raise RuntimeError(
            "Memory protection needs a Quarry build with declared inputs; the installed "
            "cisco-project-quarry does not support them. See docs/memory_protection.md."
        )


@dataclass(frozen=True, slots=True)
class ScanOutcome:
    """Authoritative Quarry verdict with per-rule enforcement intent."""

    matched: bool
    rules: tuple[str, ...]
    scan_ms: float
    blocking_rules: tuple[str, ...] = ()
    monitor_incomplete: bool = False


class QuarryMemoryScanner:
    """A profile selects default memory and flow rules; custom texts replace one stage.

    ``full_text_scan`` chooses the content mode: by default long values keep only their
    head and tail and are marked ``truncated``. Adapters read it so every event in a run
    uses the same mode.
    """

    def __init__(
        self,
        rule_texts: Sequence[str] | None = None,
        *,
        tool_rule_texts: Sequence[str] | None = None,
        profile: str = "balanced",
        model_config: Mapping[str, Any] | None = None,
        model_adapters: Mapping[str, Any] | None = None,
        include_semantic_judge: bool = False,
        full_text_scan: bool = False,
    ) -> None:
        _require_declared_inputs()
        if isinstance(rule_texts, str) or isinstance(tool_rule_texts, str):
            raise ValueError("Rule texts must be sequences of Quarry rule strings")
        texts, actions, flow_texts, flow_actions = profile_rules(profile)
        if rule_texts is not None:
            texts, actions = tuple(rule_texts), {}
        if tool_rule_texts is not None:
            flow_texts, flow_actions = tuple(tool_rule_texts), {}
        if include_semantic_judge and (model_config is None or "judge" not in model_config):
            raise ValueError("Semantic judge needs a configured Quarry 'judge' model")
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("At least one nonempty memory rule is required")
        if any(not isinstance(text, str) or not text.strip() for text in flow_texts):
            raise ValueError("Tool rules must be nonempty Quarry rule strings")
        if model_config is None and model_adapters is not None:
            raise ValueError("model_adapters require model_config")
        self._deadline_seconds = 15.0 if model_config is not None else _SCAN_DEADLINE_SECONDS
        limits = (
            replace(_SAFETY_LIMITS, rule_timeout=10.0)
            if model_config is not None else _SAFETY_LIMITS
        )
        config = (
            QuarryConfig.from_mapping(
                {"config_version": 2, "inputs": MEMORY_V1_INPUTS, "models": dict(model_config)},
                Path(__file__).parent,
                require_rules=False,
            )
            if model_config is not None
            else QuarryConfig(config_version=2, inputs=MEMORY_V1_INPUTS)
        )
        backend = (
            ModelInferenceBackend(config, adapters=model_adapters)
            if model_config is not None else None
        )
        self._scanner = self._build_scanner(texts, config, backend, limits)
        self._actions = {**{name: "enforce" for name in self._scanner.loaded_rules}, **actions}
        self._flow_scanner = (
            self._build_scanner(flow_texts, config, backend, limits) if flow_texts else None
        )
        self._semantic_scanner = (
            self._build_scanner(
                ((Path(__file__).with_name("rules") / "semantic_judge.rule").read_text(
                    encoding="utf-8"
                ),),
                config,
                backend,
                limits,
            )
            if include_semantic_judge else None
        )
        self._flow_actions = flow_actions or (
            {name: "enforce" for name in self._flow_scanner.loaded_rules}
            if self._flow_scanner is not None else {}
        )
        self.profile = profile
        self.full_text_scan = bool(full_text_scan)

    @staticmethod
    def _build_scanner(
        texts: Sequence[str],
        config: QuarryConfig,
        backend: ModelInferenceBackend | None,
        limits: SafetyLimits,
    ) -> DSLScanner:
        return DSLScanner(
            replace(config, quarry_rules_paths=(), quarry_rule_texts=tuple(texts)),
            limits=limits,
            model_inference_backend=backend,
        )

    @property
    def supports_tool_flow(self) -> bool:
        return self._flow_scanner is not None

    @staticmethod
    def _outcome(
        result: ScanResult, start: float, actions: Mapping[str, str], *, observational: bool
    ) -> ScanOutcome:
        # Quarry may report partial matches alongside an incomplete scan.
        result.raise_for_status()
        rules = tuple(result.rule_matches)
        blocking = () if observational else tuple(
            name for name in rules if actions[name] == "enforce"
        )
        return ScanOutcome(bool(rules), rules, (perf_counter() - start) * 1000, blocking)

    @staticmethod
    def _with_semantic(
        outcome: ScanOutcome, result: ScanResult | None, start: float
    ) -> ScanOutcome:
        if result is None:
            return replace(
                outcome, scan_ms=(perf_counter() - start) * 1000, monitor_incomplete=True
            )
        try:
            result.raise_for_status()
        except Exception:
            return replace(
                outcome, scan_ms=(perf_counter() - start) * 1000, monitor_incomplete=True
            )
        rules = tuple(dict.fromkeys((*outcome.rules, *result.rule_matches)))
        return replace(
            outcome, matched=bool(rules), rules=rules, scan_ms=(perf_counter() - start) * 1000
        )

    @staticmethod
    def _scan_sync(
        scanner: DSLScanner, data: dict[str, Any], deadline_seconds: float
    ) -> ScanResult:
        """Quarry's sync API uses asyncio.run; bridge only an already running loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return scanner.scan_data(data, deadline_seconds=deadline_seconds)
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(
                scanner.scan_data, data, deadline_seconds=deadline_seconds
            ).result()

    @staticmethod
    def _memory_data(event: MemoryMutation | MemoryRead) -> dict[str, Any]:
        if not isinstance(event, (MemoryMutation, MemoryRead)):
            raise TypeError("Memory scans take a v1 MemoryMutation or MemoryRead event")
        return InputBuilder.memory(event)

    def scan(self, event: MemoryMutation | MemoryRead) -> ScanOutcome:
        """Scan one v1 event. Reads are observational: they never block."""
        start = perf_counter()
        data = self._memory_data(event)
        result = self._scan_sync(self._scanner, data, self._deadline_seconds)
        outcome = self._outcome(
            result, start, self._actions, observational=isinstance(event, MemoryRead)
        )
        if self._semantic_scanner is None:
            return outcome
        try:
            semantic = self._scan_sync(self._semantic_scanner, data, self._deadline_seconds)
        except Exception:
            semantic = None
        return self._with_semantic(outcome, semantic, start)

    async def ascan(self, event: MemoryMutation | MemoryRead) -> ScanOutcome:
        start = perf_counter()
        data = self._memory_data(event)
        result = await self._scanner.scan_data_async(
            data, deadline_seconds=self._deadline_seconds
        )
        outcome = self._outcome(
            result, start, self._actions, observational=isinstance(event, MemoryRead)
        )
        if self._semantic_scanner is None:
            return outcome
        try:
            semantic = await self._semantic_scanner.scan_data_async(
                data, deadline_seconds=self._deadline_seconds
            )
        except Exception:
            semantic = None
        return self._with_semantic(outcome, semantic, start)

    def _flow_data(
        self,
        reads: Sequence[MemoryReadEvent],
        tool_name: str,
        tool_call_id: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        return InputBuilder.tool_flow(
            reads, tool_name, tool_call_id, arguments, full_text=self.full_text_scan
        )

    def scan_tool_flow(
        self,
        reads: Sequence[MemoryReadEvent],
        tool_name: str,
        tool_call_id: str,
        arguments: Mapping[str, Any],
    ) -> ScanOutcome:
        start = perf_counter()
        if self._flow_scanner is None:
            return ScanOutcome(False, (), (perf_counter() - start) * 1000)
        data = self._flow_data(reads, tool_name, tool_call_id, arguments)
        result = self._scan_sync(self._flow_scanner, data, self._deadline_seconds)
        return self._outcome(result, start, self._flow_actions, observational=False)

    async def ascan_tool_flow(
        self,
        reads: Sequence[MemoryReadEvent],
        tool_name: str,
        tool_call_id: str,
        arguments: Mapping[str, Any],
    ) -> ScanOutcome:
        start = perf_counter()
        if self._flow_scanner is None:
            return ScanOutcome(False, (), (perf_counter() - start) * 1000)
        data = self._flow_data(reads, tool_name, tool_call_id, arguments)
        result = await self._flow_scanner.scan_data_async(
            data, deadline_seconds=self._deadline_seconds
        )
        return self._outcome(result, start, self._flow_actions, observational=False)
