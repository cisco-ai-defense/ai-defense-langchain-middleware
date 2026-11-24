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

"""LangChain hooks for memory origin, read flow, and recoverable tool blocks."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping, NotRequired, Sequence

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from .backends.checkpoint import GuardedCheckpointSaver
from .backends.history import GuardedChatMessageHistory
from .backends.store import GuardedMemoryStore
from .backends.store_types import (
    MemoryDecision, MemoryInspectionFailed, MemoryMode, MemoryWriteBlocked,
    _reset_actor, _reset_read_sink, _set_actor, _set_read_sink,
)
from .events import Actor
from .flow import MemoryTraceError, RunRegistry
from .quarry.scanner import QuarryMemoryScanner

logger = logging.getLogger("aidefense.langchain.memory")
_RUN_KEY = "aidefense_memory_run_id"


class _MemoryState(AgentState):
    aidefense_memory_run_id: NotRequired[str]


class AIDefenseMemoryMiddleware(AgentMiddleware):
    """Guard LangGraph Store and checkpoint memory with local Quarry rules.

    Pass ``protect_store(store)`` and optionally ``protect_checkpointer(saver)``
    to ``create_agent`` along with this middleware. Read matches are observed;
    writes and selected pre-tool flow rules can block. No AI Defense API key or
    global patching is used.
    """

    state_schema = _MemoryState

    def __init__(
        self,
        *,
        profile: str = "balanced",
        rule_texts: Sequence[str] | None = None,
        tool_rule_texts: Sequence[str] | None = None,
        model_config: Mapping[str, Any] | None = None,
        model_adapters: Mapping[str, Any] | None = None,
        include_semantic_judge: bool = False,
        full_text_scan: bool = False,
        mode: MemoryMode = "enforce",
        protected_keys: Iterable[tuple[tuple[str, ...], str]] = (),
        gated_tools: Iterable[str] = (),
        on_decision: Callable[[MemoryDecision], None] | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("enforce", "monitor", "off"):
            raise ValueError("mode must be 'enforce', 'monitor', or 'off'")
        self.mode = mode
        self.on_decision = on_decision
        self.protected_keys = frozenset(protected_keys)
        self.gated_tools = frozenset(gated_tools)
        self.scanner = (
            QuarryMemoryScanner(
                rule_texts=rule_texts, tool_rule_texts=tool_rule_texts, profile=profile,
                model_config=model_config, model_adapters=model_adapters,
                include_semantic_judge=include_semantic_judge,
                full_text_scan=full_text_scan,
            )
            if mode != "off" else None
        )
        if self.gated_tools and mode != "off" and not self.scanner.supports_tool_flow:
            raise ValueError("gated_tools requires a profile or custom tool flow rules")
        self._runs = RunRegistry()

    def protect_store(self, store: BaseStore) -> GuardedMemoryStore:
        """Return the Store that must be passed to the agent and its tools."""
        if isinstance(store, GuardedMemoryStore):
            raise TypeError("Store is already guarded")
        return GuardedMemoryStore(
            store, self.scanner, mode=self.mode,
            protected_keys=self.protected_keys, on_decision=self.on_decision,
        )

    def protect_checkpointer(self, saver: BaseCheckpointSaver) -> GuardedCheckpointSaver:
        """Guard checkpoint message writes and observe restored history."""
        if isinstance(saver, GuardedCheckpointSaver):
            raise TypeError("Checkpointer is already guarded")
        return GuardedCheckpointSaver(
            saver, self.scanner, mode=self.mode, on_decision=self.on_decision,
        )

    def protect_history(self, history: Any, session_id: str) -> GuardedChatMessageHistory:
        """Guard a legacy RunnableWithMessageHistory session."""
        if isinstance(history, GuardedChatMessageHistory):
            raise TypeError("History is already guarded")

        def report(decision: Any) -> None:
            self._emit(MemoryDecision(
                decision.operation, (), None, "langchain.chat_history", decision.status,
                decision.rules, decision.scan_ms,
                committed=decision.committed, released=decision.released,
                kind="session", session_id=session_id,
                item_index=None if decision.operation == "clear" else decision.item_index,
            ))

        return GuardedChatMessageHistory(
            history, session_id, self.scanner, mode=self.mode,
            on_decision=report if self.on_decision is not None else None,
        )

    def before_agent(self, state: Any, runtime: Any) -> dict[str, str] | None:
        if self.scanner is None or not self.gated_tools:
            return None
        return {_RUN_KEY: self._runs.start()}

    def after_agent(self, state: Any, runtime: Any) -> None:
        if isinstance(state, dict):
            self._runs.finish(state.get(_RUN_KEY))
        return None

    @staticmethod
    def _tool_name(request: ToolCallRequest) -> str:
        return str(request.tool_call.get("name") or "unknown")[:256]

    @staticmethod
    def _tool_call_id(request: ToolCallRequest) -> str:
        return str(request.tool_call.get("id") or "unknown")[:256]

    @classmethod
    def _tool_actor(cls, request: ToolCallRequest) -> Actor:
        return Actor(type="tool", name=cls._tool_name(request), call_id=cls._tool_call_id(request))

    def _emit(self, decision: MemoryDecision) -> None:
        if self.on_decision is None:
            return
        try:
            self.on_decision(decision)
        except Exception:
            logger.warning("Memory decision callback failed")

    @staticmethod
    def _blocked_result(request: ToolCallRequest, reason: str) -> ToolMessage:
        return ToolMessage(
            content=reason,
            tool_call_id=AIDefenseMemoryMiddleware._tool_call_id(request),
            status="error",
        )

    def _run_for(self, request: ToolCallRequest):
        state = request.state
        token = state.get(_RUN_KEY) if isinstance(state, dict) else None
        return self._runs.get(token)

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        state = request.state
        run = self._runs.get(state.get(_RUN_KEY) if isinstance(state, dict) else None)
        if run is not None:
            run.verify_model_input(request.messages)
        result = handler(request)
        if run is not None:
            run.record_model_output(result)
        return result

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        state = request.state
        run = self._runs.get(state.get(_RUN_KEY) if isinstance(state, dict) else None)
        if run is not None:
            run.verify_model_input(request.messages)
        result = await handler(request)
        if run is not None:
            run.record_model_output(result)
        return result

    def _tool_flow(self, request: ToolCallRequest, run: Any) -> ToolMessage | None:
        name = self._tool_name(request)
        if name not in self.gated_tools or self.mode == "off":
            return None
        assert self.scanner is not None
        try:
            if run is None:
                raise MemoryTraceError("Agent invocation has no memory trace")
            outcome = self.scanner.scan_tool_flow(
                run.snapshot_before(
                    self._tool_call_id(request), name, request.tool_call.get("args", {})
                ), tool_name=name, tool_call_id=self._tool_call_id(request),
                arguments=request.tool_call.get("args", {}),
            )
            status = "block" if outcome.blocking_rules and self.mode == "enforce" else (
                "monitor" if outcome.matched else "allow"
            )
            decision = MemoryDecision(
                "tool_call", (), name, name, status, outcome.rules, outcome.scan_ms,
                monitor_incomplete=outcome.monitor_incomplete, kind="tool",
            )
        except Exception:
            decision = MemoryDecision("tool_call", (), name, name, "error", (), 0.0, kind="tool")
        self._emit(decision)
        if decision.status == "block":
            return self._blocked_result(request, "Tool action blocked by memory policy; it was not run.")
        if decision.status == "error" and self.mode == "enforce":
            return self._blocked_result(request, "Memory trace inspection failed; tool was not run.")
        return None

    async def _atool_flow(self, request: ToolCallRequest, run: Any) -> ToolMessage | None:
        name = self._tool_name(request)
        if name not in self.gated_tools or self.mode == "off":
            return None
        assert self.scanner is not None
        try:
            if run is None:
                raise MemoryTraceError("Agent invocation has no memory trace")
            outcome = await self.scanner.ascan_tool_flow(
                run.snapshot_before(
                    self._tool_call_id(request), name, request.tool_call.get("args", {})
                ), tool_name=name, tool_call_id=self._tool_call_id(request),
                arguments=request.tool_call.get("args", {}),
            )
            status = "block" if outcome.blocking_rules and self.mode == "enforce" else (
                "monitor" if outcome.matched else "allow"
            )
            decision = MemoryDecision(
                "tool_call", (), name, name, status, outcome.rules, outcome.scan_ms,
                monitor_incomplete=outcome.monitor_incomplete, kind="tool",
            )
        except Exception:
            decision = MemoryDecision("tool_call", (), name, name, "error", (), 0.0, kind="tool")
        self._emit(decision)
        if decision.status == "block":
            return self._blocked_result(request, "Tool action blocked by memory policy; it was not run.")
        if decision.status == "error" and self.mode == "enforce":
            return self._blocked_result(request, "Memory trace inspection failed; tool was not run.")
        return None

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]) -> Any:
        run = self._run_for(request)
        pending_reads = []
        actor_token = _set_actor(self._tool_actor(request))
        read_token = _set_read_sink(pending_reads.append if run is not None else None)
        try:
            blocked = self._tool_flow(request, run)
            if blocked is not None:
                return blocked
            try:
                result = handler(request)
                if run is not None:
                    run.release_reads(self._tool_call_id(request), pending_reads, result)
                return result
            except (MemoryWriteBlocked, MemoryInspectionFailed) as error:
                reason = (
                    "The attempted memory write was blocked by policy."
                    if isinstance(error, MemoryWriteBlocked)
                    else "Memory inspection failed; memory action did not complete."
                )
                return self._blocked_result(request, reason)
        finally:
            _reset_read_sink(read_token)
            _reset_actor(actor_token)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]
    ) -> Any:
        run = self._run_for(request)
        pending_reads = []
        actor_token = _set_actor(self._tool_actor(request))
        read_token = _set_read_sink(pending_reads.append if run is not None else None)
        try:
            blocked = await self._atool_flow(request, run)
            if blocked is not None:
                return blocked
            try:
                result = await handler(request)
                if run is not None:
                    run.release_reads(self._tool_call_id(request), pending_reads, result)
                return result
            except (MemoryWriteBlocked, MemoryInspectionFailed) as error:
                reason = (
                    "The attempted memory write was blocked by policy."
                    if isinstance(error, MemoryWriteBlocked)
                    else "Memory inspection failed; memory action did not complete."
                )
                return self._blocked_result(request, reason)
        finally:
            _reset_read_sink(read_token)
            _reset_actor(actor_token)
