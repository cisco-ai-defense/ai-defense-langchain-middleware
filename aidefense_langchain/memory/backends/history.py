# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Quarry inspection at LangChain's standalone chat-history boundary."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, replace
from typing import Callable, Literal, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, convert_to_openai_messages

from .provenance import current_memory_origin
from ..quarry.scanner import QuarryMemoryScanner, ScanOutcome

HistoryMode = Literal["enforce", "monitor", "off"]
HistoryStatus = Literal["allow", "monitor", "block", "error"]
_MAX_MESSAGES = 4096
logger = logging.getLogger("aidefense.langchain.memory.backends.history")


@dataclass(frozen=True, slots=True)
class HistoryDecision:
    session_id: str
    operation: str
    item_index: int
    status: HistoryStatus
    rules: tuple[str, ...]
    scan_ms: float
    committed: bool = False
    released: bool = False


class HistoryWriteBlocked(Exception):
    def __init__(self, decision: HistoryDecision) -> None:
        super().__init__("Chat-history write blocked by policy")
        self.decision = decision


class HistoryInspectionFailed(Exception):
    def __init__(self, decision: HistoryDecision | None = None) -> None:
        super().__init__("Chat history could not be inspected")
        self.decision = decision


def _detached_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    if len(messages) > _MAX_MESSAGES:
        raise HistoryInspectionFailed()
    try:
        copied = copy.deepcopy(list(messages))
    except Exception as exc:
        raise HistoryInspectionFailed() from exc
    if any(not isinstance(message, BaseMessage) for message in copied):
        raise HistoryInspectionFailed()
    return copied


def _visible_message(message: BaseMessage) -> dict:
    """Reject content the scanner cannot represent as the model will see it."""

    content = message.content
    if message.additional_kwargs or not (
        isinstance(content, str)
        or (
            isinstance(content, list)
            and all(
                isinstance(block, dict)
                and isinstance(block.get("text"), str)
                and set(block) in ({"text"}, {"type", "text"})
                and block.get("type", "text") == "text"
                for block in content
            )
        )
    ):
        raise HistoryInspectionFailed()
    try:
        normalized = convert_to_openai_messages([message], pass_through_unknown_blocks=False)
        if not isinstance(normalized, list) or len(normalized) != 1:
            raise ValueError("Unexpected normalized history")
        return normalized[0]
    except Exception as exc:
        raise HistoryInspectionFailed() from exc


class GuardedChatMessageHistory(BaseChatMessageHistory):
    """Wrap one session for use with ``RunnableWithMessageHistory``.

    Writes are scanned before the backing history sees them. Saved messages
    are scanned again on reads, but content matches on reads are observational.
    This wrapper does not change message order or silently remove messages.
    """

    def __init__(
        self,
        backing: BaseChatMessageHistory,
        session_id: str,
        scanner: QuarryMemoryScanner | None,
        *,
        mode: HistoryMode = "enforce",
        on_decision: Callable[[HistoryDecision], None] | None = None,
    ) -> None:
        if not isinstance(backing, BaseChatMessageHistory):
            raise TypeError("backing must be a BaseChatMessageHistory")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        if mode not in ("enforce", "monitor", "off"):
            raise ValueError("Unsupported history mode")
        if mode != "off" and scanner is None:
            raise ValueError("scanner is required unless mode is off")
        self.backing = backing
        self.session_id = session_id
        self.scanner = scanner
        self.mode = mode
        self.on_decision = on_decision

    def _emit(self, decision: HistoryDecision) -> None:
        if self.on_decision is not None:
            try:
                self.on_decision(decision)
            except Exception:
                # Audit callback failure must not change a completed decision.
                logger.warning("History decision callback failed")

    def _decide(
        self, outcome: ScanOutcome, *, operation: str, index: int, read: bool
    ) -> HistoryDecision:
        blocking = bool(outcome.blocking_rules) and self.mode == "enforce" and not read
        return HistoryDecision(
            self.session_id,
            operation,
            index,
            "block" if blocking else "monitor" if outcome.matched else "allow",
            outcome.rules,
            outcome.scan_ms,
        )

    def _scan(self, value: object, operation: str, index: int, read: bool) -> HistoryDecision:
        assert self.scanner is not None
        origin = current_memory_origin() if not read else None
        try:
            outcome = self.scanner.scan(
                value,
                operation=operation,
                namespace=("chat_history", self.session_id),
                key=str(index),
                writer_tool="langchain.chat_history",
                origin_kind=origin.kind if origin else "unknown",
                trust_level=origin.trust if origin else "unknown",
            )
        except Exception as exc:
            decision = HistoryDecision(self.session_id, operation, index, "error", (), 0.0)
            if self.mode != "monitor":
                self._emit(decision)
                raise HistoryInspectionFailed(decision) from exc
            return decision
        decision = self._decide(outcome, operation=operation, index=index, read=read)
        if decision.status == "block":
            self._emit(decision)
            raise HistoryWriteBlocked(decision)
        return decision

    async def _ascan(self, value: object, operation: str, index: int, read: bool) -> HistoryDecision:
        assert self.scanner is not None
        origin = current_memory_origin() if not read else None
        try:
            outcome = await self.scanner.ascan(
                value,
                operation=operation,
                namespace=("chat_history", self.session_id),
                key=str(index),
                writer_tool="langchain.chat_history",
                origin_kind=origin.kind if origin else "unknown",
                trust_level=origin.trust if origin else "unknown",
            )
        except Exception as exc:
            decision = HistoryDecision(self.session_id, operation, index, "error", (), 0.0)
            if self.mode != "monitor":
                self._emit(decision)
                raise HistoryInspectionFailed(decision) from exc
            return decision
        decision = self._decide(outcome, operation=operation, index=index, read=read)
        if decision.status == "block":
            self._emit(decision)
            raise HistoryWriteBlocked(decision)
        return decision

    @property
    def messages(self) -> list[BaseMessage]:
        copied = _detached_messages(self.backing.messages)
        if self.mode != "off":
            decisions = [
                self._scan(_visible_message(message), "history_read", index, True)
                for index, message in enumerate(copied)
            ]
            for decision in decisions:
                self._emit(replace(decision, released=True))
        return copied

    async def aget_messages(self) -> list[BaseMessage]:
        copied = _detached_messages(await self.backing.aget_messages())
        if self.mode != "off":
            decisions = [
                await self._ascan(_visible_message(message), "history_read", index, True)
                for index, message in enumerate(copied)
            ]
            for decision in decisions:
                self._emit(replace(decision, released=True))
        return copied

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        copied = _detached_messages(messages)
        decisions = [] if self.mode == "off" else [
            self._scan(_visible_message(message), "history_write", index, False)
            for index, message in enumerate(copied)
        ]
        self.backing.add_messages(copied)
        for decision in decisions:
            self._emit(replace(decision, committed=True))

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        copied = _detached_messages(messages)
        decisions = [] if self.mode == "off" else [
            await self._ascan(_visible_message(message), "history_write", index, False)
            for index, message in enumerate(copied)
        ]
        await self.backing.aadd_messages(copied)
        for decision in decisions:
            self._emit(replace(decision, committed=True))

    def clear(self) -> None:
        decision = None if self.mode == "off" else self._scan(None, "history_delete", 0, False)
        self.backing.clear()
        if decision is not None:
            self._emit(replace(decision, committed=True))

    async def aclear(self) -> None:
        decision = None if self.mode == "off" else await self._ascan(None, "history_delete", 0, False)
        await self.backing.aclear()
        if decision is not None:
            self._emit(replace(decision, committed=True))
