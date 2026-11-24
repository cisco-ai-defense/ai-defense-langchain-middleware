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

"""LangGraph Store write boundary and sidecar provenance."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Callable, Iterable

from langgraph.store.base import BaseStore, Op, PutOp, Result

from ..events import MemoryInspectionError, MemoryMutation, Provenance, no_content
from ..quarry.scanner import QuarryMemoryScanner, ScanOutcome
from .provenance import current_memory_origin, sidecar_key, sidecar_record, snapshot_json
from .store_reads import SIDECAR_NAMESPACE, StoreReadMixin, reserved
from .store_types import (
    MemoryDecision,
    MemoryInspectionFailed,
    MemoryMode,
    MemoryWriteBlocked,
    actor_label,
    current_actor,
    decision_for,
)

logger = logging.getLogger("aidefense.langchain.memory")


class GuardedMemoryStore(StoreReadMixin, BaseStore):
    """Scan Store writes before commit and observe reads before release.

    Pass this wrapper to the agent and its tools. A reserved Store namespace
    retains value-bound origin metadata; original Store values stay unchanged.
    """

    def __init__(
        self,
        backing: BaseStore,
        scanner: QuarryMemoryScanner | None,
        *,
        mode: MemoryMode = "enforce",
        protected_keys: Iterable[tuple[tuple[str, ...], str]] = (),
        on_decision: Callable[[MemoryDecision], None] | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("enforce", "monitor", "off"):
            raise ValueError("mode must be 'enforce', 'monitor', or 'off'")
        if mode != "off" and scanner is None:
            raise ValueError("scanner is required unless mode is 'off'")
        keys = frozenset(protected_keys)
        if any(
            not isinstance(namespace, tuple) or not namespace
            or any(not isinstance(part, str) or not part for part in namespace)
            or not isinstance(key, str) or not key
            for namespace, key in keys
        ):
            raise ValueError("protected_keys must contain namespace/key pairs")
        self._backing = backing
        self._scanner = scanner
        self._mode = mode
        self._protected_keys = keys
        self._on_decision = on_decision
        self.supports_ttl = backing.supports_ttl
        self.ttl_config = backing.ttl_config

    def _target_class(self, namespace: tuple[str, ...], key: str) -> str:
        return "protected" if (namespace, key) in self._protected_keys else "ordinary"

    def _emit(self, decision: MemoryDecision) -> None:
        if self._on_decision is None:
            return
        try:
            self._on_decision(decision)
        except Exception:
            logger.warning("Memory decision callback failed")

    def _error(self, operation: str, namespace: tuple[str, ...], key: str) -> MemoryInspectionFailed:
        return MemoryInspectionFailed(MemoryDecision(
            operation, namespace, key, actor_label(current_actor()), "error", (), 0.0,
            target_class=self._target_class(namespace, key),
        ))

    def _write_event(self, op: PutOp) -> MemoryMutation:
        assert self._scanner is not None
        origin = current_memory_origin()
        return MemoryMutation(
            schema_version=1,
            operation="delete" if op.value is None else "write",
            target=self._store_target(op.namespace, op.key),
            source=current_actor(),
            content=no_content() if op.value is None else self._content(op.value),
            provenance=Provenance(origin=origin.kind, trust=origin.trust),
            target_class=self._target_class(op.namespace, op.key),
        )

    def _handle_write_outcome(self, event: MemoryMutation, outcome: ScanOutcome) -> MemoryDecision:
        decision = decision_for(
            event,
            "block" if outcome.blocking_rules and self._mode == "enforce"
            else "monitor" if outcome.matched else "allow",
            outcome.rules,
            outcome.scan_ms,
            monitor_incomplete=outcome.monitor_incomplete,
        )
        if decision.status == "block":
            self._emit(decision)
            raise MemoryWriteBlocked(decision)
        return decision

    def _write_failed(self, op: PutOp, exc: Exception) -> MemoryDecision:
        error = self._error("delete" if op.value is None else "write", op.namespace, op.key)
        if self._mode == "monitor":
            return error.decision
        self._emit(error.decision)
        raise error from exc

    def _inspect_write(self, op: PutOp) -> MemoryDecision:
        assert self._scanner is not None
        try:
            event = self._write_event(op)
            outcome = self._scanner.scan(event)
        except Exception as exc:
            return self._write_failed(op, exc)
        return self._handle_write_outcome(event, outcome)

    async def _ainspect_write(self, op: PutOp) -> MemoryDecision:
        assert self._scanner is not None
        try:
            event = self._write_event(op)
            outcome = await self._scanner.ascan(event)
        except Exception as exc:
            return self._write_failed(op, exc)
        return self._handle_write_outcome(event, outcome)

    @staticmethod
    def _sidecar_op(op: PutOp) -> PutOp:
        metadata = None
        if op.value is not None:
            try:
                metadata = sidecar_record(op.value, current_memory_origin(), current_actor())
            except MemoryInspectionError:
                # Too large or unsupported to fingerprint: later reads report "missing".
                metadata = None
        return PutOp(
            SIDECAR_NAMESPACE, sidecar_key(op.namespace, op.key), metadata,
            index=False, ttl=op.ttl,
        )

    def _prepare(self, ops: list[Op]) -> tuple[list[Op], list[int], list[MemoryDecision]]:
        expanded: list[Op] = []
        positions: list[int] = []
        decisions: list[MemoryDecision] = []
        for op in ops:
            positions.append(len(expanded))
            if isinstance(op, PutOp):
                try:
                    stable = op._replace(value=None if op.value is None else snapshot_json(op.value))
                except Exception as exc:
                    error = self._error("write", op.namespace, op.key)
                    self._emit(error.decision)
                    raise error from exc
                decision = self._inspect_write(stable)
                decisions.append(decision)
                expanded.extend((stable, self._sidecar_op(stable)))
            else:
                expanded.append(op)
        return expanded, positions, decisions

    async def _aprepare(self, ops: list[Op]) -> tuple[list[Op], list[int], list[MemoryDecision]]:
        expanded: list[Op] = []
        positions: list[int] = []
        decisions: list[MemoryDecision] = []
        for op in ops:
            positions.append(len(expanded))
            if isinstance(op, PutOp):
                try:
                    stable = op._replace(value=None if op.value is None else snapshot_json(op.value))
                except Exception as exc:
                    error = self._error("write", op.namespace, op.key)
                    self._emit(error.decision)
                    raise error from exc
                decision = await self._ainspect_write(stable)
                decisions.append(decision)
                expanded.extend((stable, self._sidecar_op(stable)))
            else:
                expanded.append(op)
        return expanded, positions, decisions

    @staticmethod
    def _check_reserved(ops: list[Op]) -> None:
        for op in ops:
            namespace = getattr(op, "namespace", getattr(op, "namespace_prefix", None))
            if isinstance(namespace, tuple) and reserved(namespace):
                raise ValueError("Reserved memory provenance namespace")

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        operations = list(ops)
        if self._mode == "off":
            return self._backing.batch(operations)
        self._check_reserved(operations)
        expanded, _, decisions = self._prepare(operations)
        # InMemoryStore resolves batch reads against the pre-write state. Read
        # and scan that snapshot before sidecar writes change root pagination
        # or the provenance of an overwritten item.
        read_ops = [op for op in operations if not isinstance(op, PutOp)]
        read_raw = self._backing.batch(read_ops)
        if len(read_raw) != len(read_ops):
            raise ValueError("Store returned an invalid read result count")
        read_results = iter(self._finish_reads(read_ops, read_raw))
        write_ops = [op for op in expanded if isinstance(op, PutOp)]
        write_raw = self._backing.batch(write_ops)
        if len(write_raw) != len(write_ops):
            raise ValueError("Store returned an invalid write result count")
        for decision in decisions:
            self._emit(replace(decision, committed=True))
        writes = iter(write_raw)
        output = []
        for op in operations:
            if isinstance(op, PutOp):
                output.append(next(writes))
                next(writes)  # provenance sidecar result
            else:
                output.append(next(read_results))
        return output

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        operations = list(ops)
        if self._mode == "off":
            return await self._backing.abatch(operations)
        self._check_reserved(operations)
        expanded, _, decisions = await self._aprepare(operations)
        read_ops = [op for op in operations if not isinstance(op, PutOp)]
        read_raw = await self._backing.abatch(read_ops)
        if len(read_raw) != len(read_ops):
            raise ValueError("Store returned an invalid read result count")
        read_results = iter(await self._afinish_reads(read_ops, read_raw))
        write_ops = [op for op in expanded if isinstance(op, PutOp)]
        write_raw = await self._backing.abatch(write_ops)
        if len(write_raw) != len(write_ops):
            raise ValueError("Store returned an invalid write result count")
        for decision in decisions:
            self._emit(replace(decision, committed=True))
        writes = iter(write_raw)
        output = []
        for op in operations:
            if isinstance(op, PutOp):
                output.append(next(writes))
                next(writes)
            else:
                output.append(next(read_results))
        return output
