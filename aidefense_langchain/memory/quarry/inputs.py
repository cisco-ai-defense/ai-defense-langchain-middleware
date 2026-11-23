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

"""Memory event v1 declared to Quarry, and the scan data built from v1 events."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, get_args

from ..events import (
    SCHEMA_VERSION,
    ActorType,
    Content,
    MemoryMutation,
    MemoryRead,
    Origin,
    ProvenanceStatus,
    Target,
    TargetClass,
    Trust,
    normalize_content,
)

MAX_METADATA_BYTES = 256
MAX_NAMESPACE_PARTS = 16
MAX_DEPTH = 64
MAX_FLOW_READS = 64
MAX_TOOL_ARGUMENT_BYTES = 64_000
MUTATIONS = "memory.mutation_requested"
READS = "memory.read_returned"


def _one_of(values: Sequence[str]) -> dict[str, Any]:
    return {"type": "string", "values": list(values)}


def _field_values(model: type, name: str) -> tuple[str, ...]:
    return get_args(model.model_fields[name].annotation)


_TEXT = {"type": "string"}
_OPTIONAL_TEXT = {"type": "string", "optional": True}
_ACTOR = {
    "type": "object",
    "fields": {
        "type": _one_of(get_args(ActorType)),
        "name": _OPTIONAL_TEXT,
        "call_id": _OPTIONAL_TEXT,
    },
}
_COMMON = {
    "schema_version": {"type": "integer", "values": [SCHEMA_VERSION]},
    "target": {
        "type": "object",
        "fields": {
            "kind": _one_of(_field_values(Target, "kind")),
            "framework": _TEXT,
            "scope": _one_of(_field_values(Target, "scope")),
            "namespace": {"type": "list", "items": "string"},
            "namespace_path": _TEXT,
            "key": _OPTIONAL_TEXT,
            "session_id": _OPTIONAL_TEXT,
            "item_index": {"type": "integer", "optional": True},
        },
    },
    "source": _ACTOR,
    "content": {
        "type": "object",
        "fields": {
            "text": _TEXT,
            "format": _one_of(_field_values(Content, "format")),
            "truncated": {"type": "boolean"},
        },
    },
    "provenance": {
        "type": "object",
        "fields": {
            "origin": _one_of(get_args(Origin)),
            "trust": _one_of(get_args(Trust)),
            "status": {**_one_of(get_args(ProvenanceStatus)), "optional": True},
            "writer": {**_ACTOR, "optional": True},
            "writer_run_id": _OPTIONAL_TEXT,
        },
    },
    "target_class": _one_of(get_args(TargetClass)),
}

MEMORY_V1_INPUTS = {
    MUTATIONS: {
        "facts": {
            **_COMMON,
            "operation": _one_of(_field_values(MemoryMutation, "operation")),
        },
    },
    READS: {
        "facts": {
            **_COMMON,
            "operation": _one_of(_field_values(MemoryRead, "operation")),
            "query": _OPTIONAL_TEXT,
        },
    },
}


@dataclass(frozen=True, slots=True)
class MemoryReadEvent:
    """A released v1 read, with the stored value flow checks compare to tool output."""

    value: Any
    event: MemoryRead


def _bounded(value: str | None, field: str) -> None:
    if value is not None and len(value.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError(f"{field} must be at most {MAX_METADATA_BYTES} bytes")


def _check_bounds(event: MemoryMutation | MemoryRead) -> None:
    target = event.target
    if len(target.namespace) > MAX_NAMESPACE_PARTS:
        raise ValueError(f"namespace must have at most {MAX_NAMESPACE_PARTS} parts")
    for part in target.namespace:
        _bounded(part, "namespace part")
    for field in ("framework", "key", "session_id"):
        _bounded(getattr(target, field), field)
    for label, actor in (("source", event.source), ("writer", event.provenance.writer)):
        if actor is not None:
            _bounded(actor.name, f"{label} name")
            _bounded(actor.call_id, f"{label} call_id")
    _bounded(event.provenance.writer_run_id, "writer_run_id")


class InputBuilder:
    """Convert validated v1 events to the declared sections Quarry scans."""

    @staticmethod
    def item(event: MemoryMutation | MemoryRead, event_id: str) -> dict[str, Any]:
        _bounded(event_id, "event_id")
        _check_bounds(event)
        return {"id": event_id, **event.model_dump(mode="json", exclude_none=True)}

    @classmethod
    def memory(cls, event: MemoryMutation | MemoryRead) -> dict[str, Any]:
        # Both sections are always present; a missing one leaves its rules incomplete.
        item = cls.item(event, "memory-event")
        read = isinstance(event, MemoryRead)
        return {
            "memory": {
                "mutation_requested": [] if read else [item],
                "read_returned": [item] if read else [],
            }
        }

    @classmethod
    def tool_flow(
        cls,
        reads: Sequence[MemoryReadEvent],
        tool_name: str,
        tool_call_id: str,
        arguments: Mapping[str, Any],
        *,
        full_text: bool = False,
    ) -> dict[str, Any]:
        for value, field in ((tool_name, "tool_name"), (tool_call_id, "tool_call_id")):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a nonempty string")
            _bounded(value, field)
        if len(reads) > MAX_FLOW_READS:
            raise ValueError("Too many memory reads in the tool flow")
        if not isinstance(arguments, Mapping):
            raise ValueError("arguments must be a JSON object")
        try:
            encoded_args = json.dumps(
                dict(arguments), ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Tool arguments must be JSON values") from exc
        if len(encoded_args.encode("utf-8")) > MAX_TOOL_ARGUMENT_BYTES:
            raise ValueError("Tool arguments exceed the 64 KB flow limit")
        events = []
        timeline = []
        for index, read in enumerate(reads):
            if not isinstance(read, MemoryReadEvent):
                raise ValueError("Tool flow needs ordered memory read events")
            # Only the text a tool actually released is flow evidence.
            released = read.event.model_copy(
                update={"content": normalize_content(read.value, full_text=full_text)}
            )
            event_id = f"read-{index}"
            events.append(cls.item(released, event_id))
            timeline.append({"input": READS, "id": event_id})
        timeline.extend(({"message_index": 0}, {"tool_call_id": tool_call_id}))
        return {
            "memory": {"mutation_requested": [], "read_returned": events},
            "messages": [{
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": encoded_args},
                }],
            }],
            "timeline": timeline,
        }
