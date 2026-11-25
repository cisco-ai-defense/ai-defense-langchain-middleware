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

"""Quarry inspection at LangChain's standalone chat-history boundary."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, replace
from typing import Callable, Literal, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from ..events import Actor, MemoryMutation, MemoryRead, Provenance, no_content, normalize_content
from ..quarry.scanner import QuarryMemoryScanner, ScanOutcome
from .provenance import current_memory_origin

HistoryMode = Literal["enforce", "monitor", "off"]
HistoryStatus = Literal["allow", "monitor", "block", "error"]
_MAX_MESSAGES = 4096
logger = logging.getLogger("aidefense.langchain.memory.backends.history")
_CHAT_HISTORY = Actor(type="framework", name="langchain.chat_history")


@dataclass(frozen=True, slots=True)
class HistoryDecision:
    """One history decision; ``operation`` is the v1 operation (append, list, clear)."""

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


_ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
_MESSAGE_FIELDS = ("name", "tool_call_id", "tool_calls", "invalid_tool_calls")


def _detached_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    copied = copy.deepcopy(list(messages))
    if any(not isinstance(message, BaseMessage) for message in copied):
        raise ValueError("Chat history holds a value that is not a message")
    return copied


def _present(value: object) -> bool:
    return value is not None and not (isinstance(value, (str, list, dict)) and not value)


def _visible_message(message: BaseMessage) -> dict:
    """Everything a later reader of the message can see, so nothing hides from rules.

    Empty provider fields, such as OpenAI's ``refusal: None``, carry no content
    and are dropped. Other extra fields and non-text blocks are kept as data.
    """

    # Streamed chunks have types such as "AIMessageChunk".
    kind = message.type.removesuffix("MessageChunk").lower()
    role = getattr(message, "role", None) if kind == "chat" else _ROLES.get(kind, kind)
    visible: dict = {"role": role or kind, "content": message.content}
    for field in _MESSAGE_FIELDS:
        value = getattr(message, field, None)
        if _present(value):
            visible[field] = value
    extra = {key: value for key, value in message.additional_kwargs.items() if _present(value)}
    if extra:
        visible["additional_kwargs"] = extra
    return visible


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

    def _event(
        self, message: BaseMessage | None, operation: str, index: int
    ) -> MemoryMutation | MemoryRead:
        assert self.scanner is not None
        if operation == "clear":
            target = {
                "kind": "session", "framework": "langchain", "scope": "collection",
                "session_id": self.session_id,
            }
        else:
            target = {
                "kind": "session", "framework": "langchain", "session_id": self.session_id,
                "key": message.id if isinstance(message.id, str) and message.id else None,
                "item_index": index,
            }
        content = no_content() if message is None else normalize_content(
            _visible_message(message), message=True, full_text=self.scanner.full_text_scan
        )
        if operation == "list":
            return MemoryRead(
                schema_version=1, operation="list", target=target, source=_CHAT_HISTORY,
                content=content,
            )
        origin = current_memory_origin()
        return MemoryMutation(
            schema_version=1, operation=operation, target=target, source=_CHAT_HISTORY,
            content=content, provenance=Provenance(origin=origin.kind, trust=origin.trust),
        )

    def _failed(self, operation: str, index: int, exc: Exception) -> HistoryDecision:
        decision = HistoryDecision(self.session_id, operation, index, "error", (), 0.0)
        if self.mode != "monitor":
            self._emit(decision)
            raise HistoryInspectionFailed(decision) from exc
        return decision

    def _checked(self, outcome: ScanOutcome, operation: str, index: int) -> HistoryDecision:
        decision = self._decide(outcome, operation=operation, index=index, read=operation == "list")
        if decision.status == "block":
            self._emit(decision)
            raise HistoryWriteBlocked(decision)
        return decision

    def _scan(self, message: BaseMessage | None, operation: str, index: int) -> HistoryDecision:
        assert self.scanner is not None
        try:
            outcome = self.scanner.scan(self._event(message, operation, index))
        except Exception as exc:
            return self._failed(operation, index, exc)
        return self._checked(outcome, operation, index)

    async def _ascan(
        self, message: BaseMessage | None, operation: str, index: int
    ) -> HistoryDecision:
        assert self.scanner is not None
        try:
            outcome = await self.scanner.ascan(self._event(message, operation, index))
        except Exception as exc:
            return self._failed(operation, index, exc)
        return self._checked(outcome, operation, index)

    def _detach(
        self, messages: Sequence[BaseMessage], operation: str
    ) -> tuple[list[BaseMessage], list[HistoryDecision]]:
        """Copy messages, or report why not; monitor mode then passes them through."""

        try:
            return _detached_messages(messages), []
        except Exception as exc:
            return list(messages), [self._failed(operation, 0, exc)]

    def _read_window(self, copied: list[BaseMessage]) -> tuple[int, list[HistoryDecision]]:
        """Scan only the newest messages of a long history.

        Reads never block, so older messages are still released, with one error
        decision saying they were not inspected.
        """

        start = max(0, len(copied) - _MAX_MESSAGES)
        if not start:
            return 0, []
        logger.warning("Chat history has %d messages; the oldest %d were not inspected",
                       len(copied), start)
        return start, [HistoryDecision(self.session_id, "list", 0, "error", (), 0.0)]

    def _write_batch(self, copied: list[BaseMessage]) -> list[HistoryDecision]:
        if len(copied) <= _MAX_MESSAGES:
            return []
        return [self._failed("append", 0, ValueError("Too many messages in one write"))]

    @property
    def messages(self) -> list[BaseMessage]:
        if self.mode == "off":
            return self.backing.messages
        copied, decisions = self._detach(self.backing.messages, "list")
        if not decisions:
            start, decisions = self._read_window(copied)
            decisions += [
                self._scan(message, "list", index)
                for index, message in enumerate(copied[start:], start)
            ]
        for decision in decisions:
            self._emit(replace(decision, released=True))
        return copied

    async def aget_messages(self) -> list[BaseMessage]:
        if self.mode == "off":
            return await self.backing.aget_messages()
        copied, decisions = self._detach(await self.backing.aget_messages(), "list")
        if not decisions:
            start, decisions = self._read_window(copied)
            decisions += [
                await self._ascan(message, "list", index)
                for index, message in enumerate(copied[start:], start)
            ]
        for decision in decisions:
            self._emit(replace(decision, released=True))
        return copied

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if self.mode == "off":
            self.backing.add_messages(messages)
            return
        copied, decisions = self._detach(messages, "append")
        decisions = decisions or self._write_batch(copied) or [
            self._scan(message, "append", index)
            for index, message in enumerate(copied)
        ]
        self.backing.add_messages(copied)
        for decision in decisions:
            self._emit(replace(decision, committed=True))

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        if self.mode == "off":
            await self.backing.aadd_messages(messages)
            return
        copied, decisions = self._detach(messages, "append")
        decisions = decisions or self._write_batch(copied) or [
            await self._ascan(message, "append", index)
            for index, message in enumerate(copied)
        ]
        await self.backing.aadd_messages(copied)
        for decision in decisions:
            self._emit(replace(decision, committed=True))

    def clear(self) -> None:
        decision = None if self.mode == "off" else self._scan(None, "clear", 0)
        self.backing.clear()
        if decision is not None:
            self._emit(replace(decision, committed=True))

    async def aclear(self) -> None:
        decision = None if self.mode == "off" else await self._ascan(None, "clear", 0)
        await self.backing.aclear()
        if decision is not None:
            self._emit(replace(decision, committed=True))
