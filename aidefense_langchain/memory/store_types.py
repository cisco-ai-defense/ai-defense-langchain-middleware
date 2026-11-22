# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Small types and per-tool context shared by Store and middleware."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Callable, Literal

MemoryMode = Literal["enforce", "monitor", "off"]
DecisionStatus = Literal["allow", "monitor", "block", "error"]
_writer_tool: ContextVar[str] = ContextVar("aidefense_memory_writer_tool", default="application")
_read_sink: ContextVar[Callable[[Any | None], None] | None] = ContextVar(
    "aidefense_memory_read_sink", default=None
)


def current_tool() -> str:
    return _writer_tool.get()


def current_read_sink() -> Callable[[Any | None], None] | None:
    return _read_sink.get()


def _set_writer_tool(name: str) -> Token:
    return _writer_tool.set(name)


def _reset_writer_tool(token: Token) -> None:
    _writer_tool.reset(token)


def _set_read_sink(sink: Callable[[Any | None], None] | None) -> Token:
    return _read_sink.set(sink)


def _reset_read_sink(token: Token) -> None:
    _read_sink.reset(token)


@dataclass(frozen=True, slots=True)
class MemoryDecision:
    """One boundary decision; memory content is intentionally omitted."""

    operation: str
    namespace: tuple[str, ...]
    key: str
    writer_tool: str
    status: DecisionStatus
    rules: tuple[str, ...]
    scan_ms: float
    committed: bool = False
    released: bool | None = None
    reader_tool: str = ""
    origin_kind: str = "unknown"
    trust_level: str = "unknown"
    target_class: str = "unknown"
    provenance_status: str = "missing"
    monitor_incomplete: bool = False


class MemoryWriteBlocked(Exception):
    """A mutation was rejected before reaching the backing Store."""

    def __init__(self, decision: MemoryDecision) -> None:
        super().__init__("Memory write blocked by policy")
        self.decision = decision


class MemoryInspectionFailed(Exception):
    """A memory boundary could not be inspected completely."""

    def __init__(self, decision: MemoryDecision) -> None:
        super().__init__("Memory inspection failed")
        self.decision = decision
