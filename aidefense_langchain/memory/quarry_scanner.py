"""Local Quarry scanner for memory operations and pre-tool flow checks."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

from quarry import DSLScanner, ModelInferenceBackend, SafetyLimits, ScanResult
from quarry.config import QuarryConfig

from .quarry_inputs import INPUTS, MAX_DEPTH, READ_OPERATIONS, InputBuilder, MemoryReadEvent
from .quarry_profiles import profile_rules

_SCAN_DEADLINE_SECONDS = 3.0
_SAFETY_LIMITS = SafetyLimits(
    rule_timeout=1.5,
    max_total_size=5_000_000,
    max_tool_argument_size=64_000,
    max_nesting_depth=MAX_DEPTH,
    max_timeline_items=128,
    max_model_calls=4,
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
    """A profile selects default memory and flow rules; custom texts replace one stage."""

    def __init__(
        self,
        rule_texts: Sequence[str] | None = None,
        *,
        tool_rule_texts: Sequence[str] | None = None,
        profile: str = "balanced",
        model_config: Mapping[str, Any] | None = None,
        model_adapters: Mapping[str, Any] | None = None,
        include_semantic_judge: bool = False,
    ) -> None:
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
                {"config_version": 2, "inputs": INPUTS, "models": dict(model_config)},
                Path(__file__).parent,
                require_rules=False,
            )
            if model_config is not None
            else QuarryConfig(config_version=2, inputs=INPUTS)
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

    def scan(
        self,
        value: Any,
        *,
        operation: str,
        namespace: tuple[str, ...],
        key: str,
        writer_tool: str,
        reader_tool: str = "unknown",
        origin_kind: str = "unknown",
        trust_level: str = "unknown",
        target_class: str = "unknown",
        provenance_status: str = "unknown",
    ) -> ScanOutcome:
        start = perf_counter()
        data = InputBuilder.memory(
            value, operation=operation, namespace=namespace, key=key,
            writer_tool=writer_tool, reader_tool=reader_tool, origin_kind=origin_kind,
            trust_level=trust_level, target_class=target_class,
            provenance_status=provenance_status,
        )
        result = self._scan_sync(self._scanner, data, self._deadline_seconds)
        outcome = self._outcome(
            result, start, self._actions, observational=operation in READ_OPERATIONS
        )
        if self._semantic_scanner is None:
            return outcome
        try:
            semantic = self._scan_sync(self._semantic_scanner, data, self._deadline_seconds)
        except Exception:
            semantic = None
        return self._with_semantic(outcome, semantic, start)

    async def ascan(
        self,
        value: Any,
        *,
        operation: str,
        namespace: tuple[str, ...],
        key: str,
        writer_tool: str,
        reader_tool: str = "unknown",
        origin_kind: str = "unknown",
        trust_level: str = "unknown",
        target_class: str = "unknown",
        provenance_status: str = "unknown",
    ) -> ScanOutcome:
        start = perf_counter()
        data = InputBuilder.memory(
            value, operation=operation, namespace=namespace, key=key,
            writer_tool=writer_tool, reader_tool=reader_tool, origin_kind=origin_kind,
            trust_level=trust_level, target_class=target_class,
            provenance_status=provenance_status,
        )
        result = await self._scanner.scan_data_async(
            data, deadline_seconds=self._deadline_seconds
        )
        outcome = self._outcome(
            result, start, self._actions, observational=operation in READ_OPERATIONS
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
        data = InputBuilder.tool_flow(reads, tool_name, tool_call_id, arguments)
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
        data = InputBuilder.tool_flow(reads, tool_name, tool_call_id, arguments)
        result = await self._flow_scanner.scan_data_async(
            data, deadline_seconds=self._deadline_seconds
        )
        return self._outcome(result, start, self._flow_actions, observational=False)
