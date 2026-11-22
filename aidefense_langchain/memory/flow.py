# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Per-run evidence for memory-to-tool rules; no process-global transcript."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import replace
from threading import Lock
from typing import Any, Sequence
from uuid import uuid4

from langchain.agents.middleware import types as middleware_types
from langchain.messages import AIMessage, ToolMessage
from langgraph.types import Command

from .backends.provenance import snapshot_json
from .quarry.inputs import EMAIL_PATTERN, MemoryReadEvent

ModelResponse = middleware_types.ModelResponse
ExtendedModelResponse = getattr(middleware_types, "ExtendedModelResponse", None)


class MemoryTraceError(Exception):
    """A proposed tool call lacks complete, ordered evidence."""


def _model_calls(result: Any) -> list[dict[str, Any]]:
    if ExtendedModelResponse is not None and isinstance(result, ExtendedModelResponse):
        result = result.model_response
    if isinstance(result, AIMessage):
        messages = [result]
    elif isinstance(result, ModelResponse):
        messages = result.result
    else:
        raise MemoryTraceError("Unsupported model response")
    if not all(isinstance(message, AIMessage) for message in messages):
        raise MemoryTraceError("Unsupported model output messages")
    return [call for message in messages for call in message.tool_calls]


def _tool_text(result: Any, call_id: str) -> str:
    if isinstance(result, Command):
        update = result.update
        messages = update.get("messages") if isinstance(update, dict) else None
        if not isinstance(messages, (list, tuple)) or len(messages) != 1:
            raise MemoryTraceError("Unsupported memory-reading tool result")
        result = messages[0]
    if not isinstance(result, ToolMessage) or result.tool_call_id != call_id:
        raise MemoryTraceError("Unsupported memory-reading tool result")
    if isinstance(result.content, str):
        return result.content
    if isinstance(result.content, list) and all(
        isinstance(part, dict) and isinstance(part.get("text"), str)
        for part in result.content
    ):
        return "\n".join(part["text"] for part in result.content)
    raise MemoryTraceError("Unsupported memory-reading tool content")


def _string_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        return [leaf for part in value.values() for leaf in _string_leaves(part)]
    if isinstance(value, list):
        return [leaf for part in value for leaf in _string_leaves(part)]
    return []


def _released_spans(leaves: Sequence[str], output: str) -> list[str]:
    """Keep full copied leaves and exact recipient spans visible in tool text."""

    spans: list[str] = []
    for leaf in leaves:
        if leaf in output:
            spans.append(leaf)
        elif output and output in leaf:
            spans.append(output)
    # A read tool commonly extracts an address from a longer memory field, or
    # returns it in a sentence. Include only the address actually visible in
    # its result, never the surrounding unreleased memory text.
    for match in EMAIL_PATTERN.finditer(output):
        address = match.group(0)
        if any(address in leaf for leaf in leaves):
            spans.append(address)
    return list(dict.fromkeys(spans))


class MemoryRun:
    """Record only memory text actually released by an earlier tool call."""

    MAX_READS = 64
    MAX_BYTES = 512_000

    def __init__(self) -> None:
        self._reads: list[tuple[int, str, MemoryReadEvent]] = []
        self._proposals: dict[str, tuple[int, str, Any]] = {}
        self._released_outputs: dict[str, str] = {}
        self._verified_outputs: set[str] = set()
        self._generation = 0
        self._bytes = 0
        self._invalid = False
        self._lock = Lock()

    def record_model_output(self, result: Any) -> None:
        try:
            calls = _model_calls(result)
            checked = [
                (str(call["id"]), str(call["name"]), snapshot_json(call["args"]))
                for call in calls
            ]
            if any(not call_id or not name for call_id, name, _ in checked):
                raise MemoryTraceError("Model tool call has no identity")
        except (KeyError, TypeError, ValueError, MemoryTraceError):
            with self._lock:
                self._invalid = True
            return
        with self._lock:
            self._generation += 1
            for call_id, name, args in checked:
                if call_id in self._proposals:
                    self._invalid = True
                self._proposals[call_id] = (self._generation, name, args)

    def release_reads(
        self, call_id: str, pending: Sequence[MemoryReadEvent | None], result: Any
    ) -> None:
        if not pending:
            return
        try:
            output = _tool_text(result, call_id)
            if len(output.encode("utf-8")) > self.MAX_BYTES:
                raise MemoryTraceError("Memory-reading tool result is too large")
        except MemoryTraceError:
            with self._lock:
                self._invalid = True
            return
        with self._lock:
            proposal = self._proposals.get(call_id)
            if proposal is None or any(event is None for event in pending):
                self._invalid = True
                return
            self._released_outputs[call_id] = output
            generation = proposal[0]
            for event in pending:
                assert event is not None
                # A tool may summarize memory. Only exact text copied into its
                # visible result becomes flow evidence; no causation is assumed.
                leaves = _string_leaves(event.value)
                if len(leaves) > 256:
                    self._invalid = True
                    return
                copied = _released_spans(leaves, output)
                if not copied:
                    continue
                value = {"released_text": "\n".join(copied)}
                size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
                if len(self._reads) >= self.MAX_READS or self._bytes + size > self.MAX_BYTES:
                    self._invalid = True
                    return
                self._reads.append((generation, call_id, replace(event, value=value)))
                self._bytes += size

    def verify_model_input(self, messages: Sequence[Any]) -> None:
        """Reject a read-tool result rewritten before the model sees it."""

        visible = {
            message.tool_call_id: message
            for message in messages
            if isinstance(message, ToolMessage)
        }
        with self._lock:
            for call_id, original in self._released_outputs.items():
                if call_id not in visible:
                    # A summary or later middleware may have replaced the
                    # inspected result. A later gated tool must fail closed.
                    if call_id not in self._verified_outputs:
                        self._invalid = True
                    continue
                try:
                    actual = _tool_text(visible[call_id], call_id)
                except MemoryTraceError:
                    self._invalid = True
                    raise
                if actual != original:
                    self._invalid = True
                    raise MemoryTraceError("Memory-reading tool result changed before model input")
                self._verified_outputs.add(call_id)

    def snapshot_before(self, call_id: str, name: str, arguments: Any) -> tuple[MemoryReadEvent, ...]:
        with self._lock:
            proposal = self._proposals.get(call_id)
            if self._invalid or proposal is None:
                raise MemoryTraceError("Tool proposal or memory trace is incomplete")
            try:
                actual = snapshot_json(arguments)
            except ValueError as exc:
                raise MemoryTraceError("Tool arguments are not inspectable") from exc
            generation, proposed_name, proposed_args = proposal
            if proposed_name != name or proposed_args != actual:
                raise MemoryTraceError("Tool call differs from model proposal")
            return tuple(
                event for sequence, source_call_id, event in self._reads
                if sequence < generation and source_call_id in self._verified_outputs
            )


class RunRegistry:
    """Bounded records keyed by an invocation ID in LangChain agent state."""

    MAX_RUNS = 256

    def __init__(self) -> None:
        self._runs: OrderedDict[str, MemoryRun] = OrderedDict()
        self._lock = Lock()

    def start(self) -> str:
        token = uuid4().hex
        with self._lock:
            self._runs[token] = MemoryRun()
            while len(self._runs) > self.MAX_RUNS:
                self._runs.popitem(last=False)
        return token

    def get(self, token: str | None) -> MemoryRun | None:
        if not token:
            return None
        with self._lock:
            return self._runs.get(token)

    def finish(self, token: str | None) -> None:
        if token:
            with self._lock:
                self._runs.pop(token, None)
