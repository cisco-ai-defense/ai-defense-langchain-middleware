# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Normalize and inspect LangGraph checkpoint message values."""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import replace
from typing import Any
from collections.abc import Sequence

from langchain_core.messages import RemoveMessage, convert_to_messages, convert_to_openai_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.types import Command

from ..quarry.scanner import ScanOutcome
from .provenance import MemoryOrigin, current_memory_origin
from .store_types import MemoryDecision, MemoryInspectionFailed, MemoryWriteBlocked

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


def _namespace(config: RunnableConfig, channel: str) -> tuple[str, ...]:
    thread_id, checkpoint_ns = _thread(config)
    return ("checkpoint", thread_id, checkpoint_ns or "root", channel)


def _messages(channel: str, value: Any) -> list[Any]:
    if channel == "__start__":
        if isinstance(value, Command):
            value = value.update or {}
        if not isinstance(value, dict):
            raise ValueError("Checkpoint start input cannot be inspected")
        value = value.get("messages", [])
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
        return [value]
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _normalized(message: Any) -> dict[str, Any]:
    converted = convert_to_messages([message])
    if len(converted) != 1:
        raise ValueError("Checkpoint message cannot be normalized")
    message = converted[0]
    if isinstance(message, RemoveMessage):
        if not isinstance(message.id, str) or not message.id:
            raise ValueError("Checkpoint removal has no stable message ID")
        return {"removed_message_id": message.id}
    openai_messages = convert_to_openai_messages([message])
    if not isinstance(openai_messages, list) or len(openai_messages) != 1:
        raise ValueError("Checkpoint message cannot be normalized")
    return openai_messages[0]


def _fingerprint(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)



class _CheckpointInspection:
    """Shared sync/async scanner calls for one checkpoint saver."""

    @staticmethod
    def _items(channels: Sequence[tuple[str, Any]]) -> list[tuple[str, str, dict[str, Any]]]:
        items = []
        for channel, value in channels:
            if channel not in ("messages", "__start__"):
                continue
            for index, message in enumerate(_messages(channel, value)):
                content = _normalized(message)
                key = content.get("removed_message_id", str(index))
                items.append((channel, key, content))
        return items

    def _new_items_from_previous(
        self, previous: CheckpointTuple | None, channels: Sequence[tuple[str, Any]]
    ) -> list[tuple[str, str, dict[str, Any]]]:
        old = (
            previous.checkpoint.get("channel_values", {}).get("messages", [])
            if previous is not None
            else []
        )
        unchanged = Counter(
            _fingerprint(value)
            for _, _, value in self._items([("messages", old)])
        )
        result = []
        for channel, key, value in self._items(channels):
            digest = _fingerprint(value)
            if channel == "messages" and unchanged[digest] > 0:
                unchanged[digest] -= 1
            else:
                result.append((channel, key, value))
        return result

    def _new_items(
        self, config: RunnableConfig, channels: Sequence[tuple[str, Any]]
    ) -> list[tuple[str, str, dict[str, Any]]]:
        return self._new_items_from_previous(self._backing.get_tuple(config), channels)

    async def _anew_items(
        self, config: RunnableConfig, channels: Sequence[tuple[str, Any]]
    ) -> list[tuple[str, str, dict[str, Any]]]:
        return self._new_items_from_previous(await self._backing.aget_tuple(config), channels)

    def _decision(
        self,
        *,
        config: RunnableConfig,
        channel: str,
        key: str,
        operation: str,
        outcome: ScanOutcome | None,
    ) -> MemoryDecision:
        matched = bool(outcome and outcome.rules)
        blocking = bool(outcome and getattr(outcome, "blocking_rules", outcome.rules))
        status = (
            "error" if outcome is None
            else "block" if operation in ("checkpoint_write", "checkpoint_delete") and blocking and self._mode == "enforce"
            else "monitor" if matched
            else "allow"
        )
        origin = MemoryOrigin() if operation == "checkpoint_read" else current_memory_origin()
        return MemoryDecision(
            operation=operation,
            namespace=_namespace(config, channel),
            key=key,
            writer_tool="langgraph.checkpointer",
            status=status,
            rules=outcome.rules if outcome else (),
            scan_ms=outcome.scan_ms if outcome else 0.0,
            origin_kind=origin.kind,
            trust_level=origin.trust,
            monitor_incomplete=outcome.monitor_incomplete if outcome else False,
        )

    def _scan_one(
        self, config: RunnableConfig, item: tuple[str, str, dict[str, Any]], *, read: bool
    ) -> MemoryDecision:
        channel, key, content = item
        operation = (
            "checkpoint_read" if read
            else "checkpoint_delete" if "removed_message_id" in content
            else "checkpoint_write"
        )
        try:
            assert self._scanner is not None
            origin = current_memory_origin() if not read else None
            outcome = self._scanner.scan(
                content,
                operation=operation,
                namespace=_namespace(config, channel),
                key=key,
                writer_tool="langgraph.checkpointer",
                origin_kind=origin.kind if origin else "unknown",
                trust_level=origin.trust if origin else "unknown",
            )
        except Exception as exc:
            decision = self._decision(
                config=config, channel=channel, key=key, operation=operation, outcome=None
            )
            self._emit(decision)
            if self._mode == "enforce":
                raise MemoryInspectionFailed(decision) from exc
            return decision
        decision = self._decision(
            config=config, channel=channel, key=key, operation=operation, outcome=outcome
        )
        if decision.status == "block":
            self._emit(decision)
            raise MemoryWriteBlocked(decision)
        if read:
            self._emit(replace(decision, released=True))
        return decision

    async def _ascan_one(
        self, config: RunnableConfig, item: tuple[str, str, dict[str, Any]], *, read: bool
    ) -> MemoryDecision:
        channel, key, content = item
        operation = (
            "checkpoint_read" if read
            else "checkpoint_delete" if "removed_message_id" in content
            else "checkpoint_write"
        )
        try:
            assert self._scanner is not None
            origin = current_memory_origin() if not read else None
            outcome = await self._scanner.ascan(
                content,
                operation=operation,
                namespace=_namespace(config, channel),
                key=key,
                writer_tool="langgraph.checkpointer",
                origin_kind=origin.kind if origin else "unknown",
                trust_level=origin.trust if origin else "unknown",
            )
        except Exception as exc:
            decision = self._decision(
                config=config, channel=channel, key=key, operation=operation, outcome=None
            )
            self._emit(decision)
            if self._mode == "enforce":
                raise MemoryInspectionFailed(decision) from exc
            return decision
        decision = self._decision(
            config=config, channel=channel, key=key, operation=operation, outcome=outcome
        )
        if decision.status == "block":
            self._emit(decision)
            raise MemoryWriteBlocked(decision)
        if read:
            self._emit(replace(decision, released=True))
        return decision

    def _observe(self, item: CheckpointTuple | None) -> CheckpointTuple | None:
        if item is None or self._mode == "off":
            return item
        safe = copy.deepcopy(item)
        channels = safe.checkpoint.get("channel_values", {})
        if isinstance(channels, dict):
            for part in self._items(list(channels.items())):
                self._scan_one(safe.config, part, read=True)
        for _, channel, value in safe.pending_writes or []:
            for part in self._items([(channel, value)]):
                self._scan_one(safe.config, part, read=True)
        return safe

    async def _aobserve(self, item: CheckpointTuple | None) -> CheckpointTuple | None:
        if item is None or self._mode == "off":
            return item
        safe = copy.deepcopy(item)
        channels = safe.checkpoint.get("channel_values", {})
        if isinstance(channels, dict):
            for part in self._items(list(channels.items())):
                await self._ascan_one(safe.config, part, read=True)
        for _, channel, value in safe.pending_writes or []:
            for part in self._items([(channel, value)]):
                await self._ascan_one(safe.config, part, read=True)
        return safe
