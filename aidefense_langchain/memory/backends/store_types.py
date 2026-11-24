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

"""Small types and per-tool context shared by Store and middleware."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Callable, Literal

from ..events import Actor, MemoryMutation, MemoryRead

MemoryMode = Literal["enforce", "monitor", "off"]
DecisionStatus = Literal["allow", "monitor", "block", "error"]
_APPLICATION = Actor(type="application")
_actor: ContextVar[Actor] = ContextVar("aidefense_memory_actor", default=_APPLICATION)
_read_sink: ContextVar[Callable[[Any | None], None] | None] = ContextVar(
    "aidefense_memory_read_sink", default=None
)


def current_actor() -> Actor:
    """Who is touching memory now: the running tool, or the application."""

    return _actor.get()


def actor_label(actor: Actor | None) -> str:
    if actor is None:
        return "unknown"
    return actor.name or actor.type


def current_read_sink() -> Callable[[Any | None], None] | None:
    return _read_sink.get()


def _set_actor(actor: Actor) -> Token:
    return _actor.set(actor)


def _reset_actor(token: Token) -> None:
    _actor.reset(token)


def _set_read_sink(sink: Callable[[Any | None], None] | None) -> Token:
    return _read_sink.set(sink)


def _reset_read_sink(token: Token) -> None:
    _read_sink.reset(token)


@dataclass(frozen=True, slots=True)
class MemoryDecision:
    """One boundary decision; memory content is intentionally omitted.

    ``operation`` is the v1 operation and ``kind`` the v1 target kind.
    """

    operation: str
    namespace: tuple[str, ...]
    key: str | None
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
    kind: str = "store"
    session_id: str | None = None
    item_index: int | None = None


def decision_for(
    event: MemoryMutation | MemoryRead,
    status: DecisionStatus,
    rules: tuple[str, ...] = (),
    scan_ms: float = 0.0,
    *,
    monitor_incomplete: bool = False,
) -> MemoryDecision:
    read = isinstance(event, MemoryRead)
    target, provenance = event.target, event.provenance
    return MemoryDecision(
        operation=event.operation,
        namespace=tuple(target.namespace),
        key=target.key,
        writer_tool=actor_label(provenance.writer),
        status=status,
        rules=rules,
        scan_ms=scan_ms,
        reader_tool=actor_label(event.source) if read else "",
        origin_kind=provenance.origin,
        trust_level=provenance.trust,
        target_class=event.target_class,
        provenance_status=provenance.status or "missing",
        monitor_incomplete=monitor_incomplete,
        kind=target.kind,
        session_id=target.session_id,
        item_index=target.item_index,
    )


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
