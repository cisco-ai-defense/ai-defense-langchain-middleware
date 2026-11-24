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

"""Normalize and inspect LangGraph checkpoint message values."""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, NamedTuple

from langchain_core.messages import RemoveMessage, convert_to_messages, convert_to_openai_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.types import Command

from ..events import Actor, MemoryMutation, MemoryRead, Provenance, no_content, normalize_content
from ..quarry.scanner import ScanOutcome
from .provenance import current_memory_origin
from .store_types import MemoryDecision, MemoryInspectionFailed, MemoryWriteBlocked, decision_for

_CHECKPOINTER = Actor(type="framework", name="langgraph.checkpointer")


try:
    from langgraph.types import Overwrite
except ImportError:  # langgraph releases before Overwrite
    Overwrite = None

_OVERWRITE = "__overwrite__"


class _Message(NamedTuple):
    channel: str
    message_id: str | None
    index: int
    value: dict[str, Any] | None  # None for a RemoveMessage
    problem: Exception | None = None  # set when the message could not be normalized

    @property
    def removal(self) -> bool:
        return self.value is None and self.problem is None


def _thread(config: RunnableConfig) -> tuple[str, str]:
    options = config.get("configurable") if isinstance(config, dict) else None
    if not isinstance(options, dict) or not isinstance(options.get("thread_id"), str):
        raise ValueError("Checkpoint config needs a thread_id")
    thread_id = options["thread_id"]
    if not thread_id or len(thread_id) > 256:
        raise ValueError("Checkpoint thread_id is invalid")
    checkpoint_ns = options.get("checkpoint_ns", "")
    if not isinstance(checkpoint_ns, str) or len(checkpoint_ns) > 256:
        raise ValueError("Checkpoint namespace is invalid")
    return thread_id, checkpoint_ns


def _unwrap_overwrite(value: Any) -> Any:
    if Overwrite is not None and isinstance(value, Overwrite):
        return value.value
    if isinstance(value, dict):
        if len(value) == 1 and _OVERWRITE in value:
            return value[_OVERWRITE]
        if value.get("type") == _OVERWRITE and "value" in value:
            return value["value"]
    return value


def _start_messages(value: Any) -> Any:
    """The ``messages`` part of a graph input, whatever shape the state takes."""

    if isinstance(value, Command):
        if isinstance(value.update, (list, tuple)):
            return [message for key, update in value.update if key == "messages"
                    for message in _messages("messages", update)]
        value = value.update
    if value is None:
        return []
    if isinstance(value, dict):
        return value.get("messages", [])
    if isinstance(value, (list, tuple)):
        return value
    # Pydantic models, dataclasses, and other attribute-style state.
    return getattr(value, "messages", [])


def _messages(channel: str, value: Any) -> list[Any]:
    if channel == "__start__":
        value = _start_messages(value)
    value = _unwrap_overwrite(value)
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
        return [value]
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _normalized(message: Any) -> tuple[str | None, dict[str, Any] | None]:
    converted = convert_to_messages([message])
    if len(converted) != 1:
        raise ValueError("Checkpoint message cannot be normalized")
    message = converted[0]
    message_id = message.id if isinstance(message.id, str) and message.id else None
    if isinstance(message, RemoveMessage):
        if message_id is None:
            raise ValueError("Checkpoint removal has no stable message ID")
        return message_id, None
    openai_messages = convert_to_openai_messages([message])
    if not isinstance(openai_messages, list) or len(openai_messages) != 1:
        raise ValueError("Checkpoint message cannot be normalized")
    return message_id, openai_messages[0]


def _fingerprint(item: _Message) -> str | None:
    if item.problem is not None:
        # No fingerprint: never treated as unchanged, so it is reported on every checkpoint.
        return None
    value = {"removed": item.message_id} if item.value is None else item.value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class _CheckpointInspection:
    """Shared sync/async scanner calls for one checkpoint saver."""

    @staticmethod
    def _items(channels: Sequence[tuple[str, Any]]) -> list[_Message]:
        items = []
        for channel, value in channels:
            if channel not in ("messages", "__start__"):
                continue
            # Shape errors become items so they reach the decision path; monitor
            # mode then continues instead of raising from the saver.
            try:
                messages = _messages(channel, value)
            except Exception as exc:
                items.append(_Message(channel, None, 0, None, exc))
                continue
            for index, message in enumerate(messages):
                try:
                    message_id, normalized = _normalized(message)
                except Exception as exc:
                    items.append(_Message(channel, None, index, None, exc))
                    continue
                items.append(_Message(channel, message_id, index, normalized))
        return items

    def _new_items_from_previous(
        self, previous: CheckpointTuple | None, channels: Sequence[tuple[str, Any]]
    ) -> list[_Message]:
        old = (
            previous.checkpoint.get("channel_values", {}).get("messages", [])
            if previous is not None
            else []
        )
        unchanged = Counter(
            digest for item in self._items([("messages", old)])
            if (digest := _fingerprint(item)) is not None
        )
        result = []
        for item in self._items(channels):
            digest = _fingerprint(item)
            if item.channel == "messages" and digest is not None and unchanged[digest] > 0:
                unchanged[digest] -= 1
            else:
                result.append(item)
        return result

    def _new_items(
        self, config: RunnableConfig, channels: Sequence[tuple[str, Any]]
    ) -> list[_Message]:
        return self._new_items_from_previous(self._backing.get_tuple(config), channels)

    async def _anew_items(
        self, config: RunnableConfig, channels: Sequence[tuple[str, Any]]
    ) -> list[_Message]:
        return self._new_items_from_previous(await self._backing.aget_tuple(config), channels)

    @staticmethod
    def _target(config: RunnableConfig, item: _Message) -> dict[str, Any]:
        thread_id, checkpoint_ns = _thread(config)
        return {
            "kind": "checkpoint",
            "framework": "langgraph",
            "session_id": thread_id,
            "namespace": [checkpoint_ns or "root", item.channel],
            "key": item.message_id,
            "item_index": item.index,
        }

    def _event(
        self, config: RunnableConfig, item: _Message, *, read: bool
    ) -> MemoryMutation | MemoryRead:
        assert self._scanner is not None
        if item.problem is not None:
            raise item.problem
        target = self._target(config, item)
        if read:
            return MemoryRead(
                schema_version=1,
                operation="list",
                target=target,
                source=_CHECKPOINTER,
                content=normalize_content(
                    item.value, message=True, full_text=self._scanner.full_text_scan
                ),
            )
        origin = current_memory_origin()
        return MemoryMutation(
            schema_version=1,
            operation="delete" if item.removal else "write",
            target=target,
            source=_CHECKPOINTER,
            content=no_content() if item.removal else normalize_content(
                item.value, message=True, full_text=self._scanner.full_text_scan
            ),
            provenance=Provenance(origin=origin.kind, trust=origin.trust),
        )

    @staticmethod
    def _error_decision(config: RunnableConfig, item: _Message, *, read: bool) -> MemoryDecision:
        try:
            thread_id, checkpoint_ns = _thread(config)
        except ValueError:
            thread_id, checkpoint_ns = None, ""
        return MemoryDecision(
            operation="list" if read else "delete" if item.removal else "write",
            namespace=(checkpoint_ns or "root", item.channel),
            key=item.message_id,
            writer_tool=_CHECKPOINTER.name,
            status="error",
            rules=(),
            scan_ms=0.0,
            kind="checkpoint",
            session_id=thread_id,
            item_index=item.index,
        )

    def _decide(
        self, event: MemoryMutation | MemoryRead, outcome: ScanOutcome, *, read: bool
    ) -> MemoryDecision:
        status = (
            "block" if not read and outcome.blocking_rules and self._mode == "enforce"
            else "monitor" if outcome.matched
            else "allow"
        )
        decision = decision_for(
            event, status, outcome.rules, outcome.scan_ms,
            monitor_incomplete=outcome.monitor_incomplete,
        )
        if decision.status == "block":
            self._emit(decision)
            raise MemoryWriteBlocked(decision)
        if read:
            decision = replace(decision, released=True)
            self._emit(decision)
        return decision

    def _failed(
        self, config: RunnableConfig, item: _Message, exc: Exception, *, read: bool
    ) -> MemoryDecision:
        decision = self._error_decision(config, item, read=read)
        # A monitored write is emitted by the caller once it commits.
        if read or self._mode == "enforce":
            self._emit(decision)
        if self._mode == "enforce":
            raise MemoryInspectionFailed(decision) from exc
        return decision

    def _scan_one(self, config: RunnableConfig, item: _Message, *, read: bool) -> MemoryDecision:
        try:
            event = self._event(config, item, read=read)
            outcome = self._scanner.scan(event)
        except Exception as exc:
            return self._failed(config, item, exc, read=read)
        return self._decide(event, outcome, read=read)

    async def _ascan_one(
        self, config: RunnableConfig, item: _Message, *, read: bool
    ) -> MemoryDecision:
        try:
            event = self._event(config, item, read=read)
            outcome = await self._scanner.ascan(event)
        except Exception as exc:
            return self._failed(config, item, exc, read=read)
        return self._decide(event, outcome, read=read)

    def _replayed(self, item: CheckpointTuple) -> list[_Message]:
        channels = item.checkpoint.get("channel_values", {})
        parts = self._items(list(channels.items())) if isinstance(channels, dict) else []
        for _, channel, value in item.pending_writes or []:
            parts.extend(self._items([(channel, value)]))
        # A pending removal releases no text; it was checked when it was written.
        return [part for part in parts if not part.removal]

    def _observe(self, item: CheckpointTuple | None) -> CheckpointTuple | None:
        if item is None or self._mode == "off":
            return item
        safe = copy.deepcopy(item)
        for part in self._replayed(safe):
            self._scan_one(safe.config, part, read=True)
        return safe

    async def _aobserve(self, item: CheckpointTuple | None) -> CheckpointTuple | None:
        if item is None or self._mode == "off":
            return item
        safe = copy.deepcopy(item)
        for part in self._replayed(safe):
            await self._ascan_one(safe.config, part, read=True)
        return safe
