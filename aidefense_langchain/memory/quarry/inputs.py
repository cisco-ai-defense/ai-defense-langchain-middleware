"""Bounded, host-stamped declared inputs and flow timelines for Quarry."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MAX_CONTENT_BYTES = 1_048_576
MAX_METADATA_BYTES = 256
MAX_DEPTH = 64
READ_OPERATIONS = frozenset({"read", "get", "search", "checkpoint_read", "history_read"})
_ORIGINS = frozenset({"external", "model", "system", "user", "unknown"})
_TRUST = frozenset({"trusted", "untrusted", "unknown"})
_TARGETS = frozenset({"protected", "ordinary", "unknown"})
_PROVENANCE = frozenset({"verified", "missing", "mismatch", "unknown"})
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

INPUTS = {
    "memory.events": {
        "content": ["content"],
        "facts": {
            "operation": {"type": "string"},
            "namespace": {"type": "list", "items": "string"},
            "key": {"type": "string"},
            "writer_tool": {"type": "string"},
            "reader_tool": {"type": "string"},
            "origin_kind": {"type": "string", "values": sorted(_ORIGINS)},
            "trust_level": {"type": "string", "values": sorted(_TRUST)},
            "target_class": {"type": "string", "values": sorted(_TARGETS)},
            "provenance_status": {"type": "string", "values": sorted(_PROVENANCE)},
            "recipient_matches_to": {"type": "boolean"},
        },
    }
}


@dataclass(frozen=True, slots=True)
class MemoryReadEvent:
    """One host-recorded memory release, in the order it reached the agent."""

    value: Any
    namespace: tuple[str, ...]
    key: str
    writer_tool: str = "unknown"
    reader_tool: str = "unknown"
    origin_kind: str = "unknown"
    trust_level: str = "unknown"
    target_class: str = "unknown"
    provenance_status: str = "missing"
    operation: str = "read"


def readable_content(value: Any) -> str:
    """Include every JSON leaf and key; oversized content fails instead of truncating."""
    lines: list[str] = []
    size = 0
    ancestors: set[int] = set()

    def emit(line: str) -> None:
        nonlocal size
        if len(line) > MAX_CONTENT_BYTES:
            raise ValueError("Memory value exceeds the 1 MiB scan limit")
        size += len(line.encode("utf-8")) + bool(lines)
        if size > MAX_CONTENT_BYTES:
            raise ValueError("Memory value exceeds the 1 MiB scan limit")
        lines.append(line)

    def walk(item: Any, path: str, depth: int) -> None:
        if depth > MAX_DEPTH:
            raise ValueError("Memory value is too deeply nested to scan")
        if isinstance(item, str):
            emit(f"{path}: {item}" if path else item)
        elif item is None or isinstance(item, (bool, int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Memory value contains a non-JSON number")
            text = json.dumps(item, allow_nan=False)
            emit(f"{path}: {text}" if path else text)
        elif isinstance(item, (dict, list)):
            identity = id(item)
            if identity in ancestors:
                raise ValueError("Memory value contains a cycle")
            ancestors.add(identity)
            try:
                if isinstance(item, dict):
                    if not all(isinstance(key, str) for key in item):
                        raise ValueError("Memory object keys must be strings")
                    if not item:
                        emit(f"{path}: {{}}" if path else "{}")
                    for key in item:
                        walk(item[key], f"{path}.{key}" if path else key, depth + 1)
                else:
                    if not item:
                        emit(f"{path}: []" if path else "[]")
                    for index, child in enumerate(item):
                        walk(child, f"{path}[{index}]" if path else f"[{index}]", depth + 1)
            finally:
                ancestors.remove(identity)
        else:
            raise ValueError(f"Unsupported memory value type: {type(item).__name__}")

    walk(value, "", 0)
    return "\n".join(lines)


class InputBuilder:
    """Convert native memory values to the declared sections Quarry validates."""

    @staticmethod
    def bounded_text(value: str, field: str) -> str:
        if not isinstance(value, str) or not value or len(value.encode("utf-8")) > MAX_METADATA_BYTES:
            raise ValueError(f"{field} must be a nonempty string of at most 256 bytes")
        return value

    @classmethod
    def event(
        cls,
        value: Any,
        *,
        event_id: str,
        operation: str,
        namespace: tuple[str, ...],
        key: str,
        writer_tool: str,
        reader_tool: str,
        origin_kind: str,
        trust_level: str,
        target_class: str,
        provenance_status: str,
        recipient_matches_to: bool = False,
    ) -> dict[str, Any]:
        cls.bounded_text(event_id, "event_id")
        cls.bounded_text(operation, "operation")
        if (
            not isinstance(namespace, tuple)
            or len(namespace) > 16
            or not all(isinstance(part, str) and part for part in namespace)
        ):
            raise ValueError("namespace must have at most 16 nonempty string parts")
        for part in namespace:
            cls.bounded_text(part, "namespace part")
        cls.bounded_text(key, "key")
        cls.bounded_text(writer_tool, "writer_tool")
        cls.bounded_text(reader_tool, "reader_tool")
        if (
            origin_kind not in _ORIGINS
            or trust_level not in _TRUST
            or target_class not in _TARGETS
            or provenance_status not in _PROVENANCE
        ):
            raise ValueError("Invalid host-stamped memory provenance")
        return {
            "id": event_id,
            "content": readable_content(value),
            "operation": operation,
            "namespace": list(namespace),
            "key": key,
            "writer_tool": writer_tool,
            "reader_tool": reader_tool,
            "origin_kind": origin_kind,
            "trust_level": trust_level,
            "target_class": target_class,
            "provenance_status": provenance_status,
            "recipient_matches_to": recipient_matches_to,
        }

    @classmethod
    def memory(cls, value: Any, **facts: Any) -> dict[str, Any]:
        return {"memory": {"events": [cls.event(value, event_id="memory-event", **facts)]}}

    @classmethod
    def tool_flow(
        cls,
        reads: Sequence[MemoryReadEvent],
        tool_name: str,
        tool_call_id: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        cls.bounded_text(tool_name, "tool_name")
        cls.bounded_text(tool_call_id, "tool_call_id")
        if len(reads) > 64:
            raise ValueError("Too many memory reads in the tool flow")
        if not isinstance(arguments, Mapping):
            raise ValueError("arguments must be a JSON object")
        readable_content(dict(arguments))  # Reject non-JSON values and keys.
        encoded_args = json.dumps(
            dict(arguments), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        if len(encoded_args.encode("utf-8")) > 64_000:
            raise ValueError("Tool arguments exceed the 64 KB flow limit")
        events = []
        timeline = []
        to_value = arguments.get("to")
        for index, read in enumerate(reads):
            if not isinstance(read, MemoryReadEvent) or read.operation not in READ_OPERATIONS:
                raise ValueError("Tool flow needs ordered memory read events")
            event_id = f"read-{index}"
            events.append(cls.event(
                read.value,
                event_id=event_id,
                operation=read.operation,
                namespace=read.namespace,
                key=read.key,
                writer_tool=read.writer_tool,
                reader_tool=read.reader_tool,
                origin_kind=read.origin_kind,
                trust_level=read.trust_level,
                target_class=read.target_class,
                provenance_status=read.provenance_status,
                recipient_matches_to=(
                    tool_name == "send_report"
                    and isinstance(to_value, str)
                    and any(
                        match.group(0) == to_value
                        for match in EMAIL_PATTERN.finditer(readable_content(read.value))
                    )
                ),
            ))
            timeline.append({"input": "memory.events", "id": event_id})
        timeline.extend(({"message_index": 0}, {"tool_call_id": tool_call_id}))
        return {
            "memory": {"events": events},
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
