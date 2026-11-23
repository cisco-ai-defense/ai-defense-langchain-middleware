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

"""Framework-neutral v1 memory events, content normalization, and value hashing."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MemoryInspectionError(Exception):
    """Inspection could not make a reliable decision."""


SCHEMA_VERSION = 1
PUBLISHED_FRAMEWORKS = frozenset(
    {"langgraph", "langchain", "openai_agents", "google_adk", "crewai", "ag2", "strands"}
)

# Values beyond these caps are inspection errors in every mode.
MAX_VALUE_BYTES = 1024 * 1024
MAX_TEXT_BYTES = 2 * MAX_VALUE_BYTES
# Default (limited) mode keeps only the head and tail of long text, for the boundary
# scan and the run trace alike. Full-text mode keeps everything up to the caps.
LIMITED_HEAD_BYTES = 12 * 1024
LIMITED_TAIL_BYTES = 4 * 1024

_IDENTIFIER = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*\Z")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


ActorType = Literal[
    "tool", "model", "user", "agent", "framework", "application", "unknown"
]
Origin = Literal["user", "external", "model", "tool", "system", "unknown"]
Trust = Literal["trusted", "untrusted", "unknown"]
ProvenanceStatus = Literal["new", "verified", "missing", "mismatch"]
TargetClass = Literal["protected", "ordinary", "unknown"]

_ALLOWED_SCOPES = {
    "write": {"item"},
    "append": {"item"},
    "delete": {"item"},
    "clear": {"collection", "all"},
    "get": {"item"},
    "list": {"item"},
    "search": {"item"},
}
_NO_CONTENT = {"delete", "clear"}


def _check_operation(event: MemoryEventBase, operation: str) -> None:
    """Reject operation, scope, and content combinations that rules would misread."""

    if event.target.scope not in _ALLOWED_SCOPES[operation]:
        raise ValueError(f"{operation} does not allow scope {event.target.scope!r}")
    no_value = event.content.format == "none"
    if operation in _NO_CONTENT:
        if not no_value:
            raise ValueError(f"{operation} must not carry content")
    elif no_value:
        raise ValueError(f"{operation} requires content")


class Actor(_Strict):
    """Who performs an operation, now or at original write time."""

    type: ActorType = "unknown"
    name: str | None = Field(default=None, min_length=1)
    call_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Actor:
        if self.call_id and self.type != "tool":
            raise ValueError("call_id is only valid for tool actors")
        if self.type in ("tool", "agent", "framework") and not self.name:
            raise ValueError(f"{self.type} actors need a name")
        return self


class Target(_Strict):
    """Which memory object and which part of it the operation touches."""

    kind: Literal["store", "session", "checkpoint", "semantic", "other"]
    framework: str
    scope: Literal["item", "collection", "all"] = "item"
    namespace: list[str] = Field(default_factory=list)
    namespace_path: str = ""
    key: str | None = Field(default=None, min_length=1)
    session_id: str | None = Field(default=None, min_length=1)
    item_index: int | None = Field(default=None, ge=0)

    @field_validator("framework")
    @classmethod
    def _framework(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("framework must be a plain identifier")
        return value

    @field_validator("namespace")
    @classmethod
    def _segments(cls, value: list[str]) -> list[str]:
        if any(not segment or "/" in segment for segment in value):
            raise ValueError("namespace segments must be nonempty and contain no '/'")
        return value

    @model_validator(mode="after")
    def _identity(self) -> Target:
        path = f"/{'/'.join(self.namespace)}/" if self.namespace else ""
        if not self.namespace_path:
            self.namespace_path = path
        elif self.namespace_path != path:
            raise ValueError("namespace_path must match namespace")
        if self.scope == "item" and self.key is None and self.item_index is None:
            raise ValueError("Item-scoped memory needs a key or item_index")
        if self.scope == "collection" and not (self.namespace or self.session_id):
            raise ValueError("Collection-scoped memory needs a namespace or session_id")
        return self


class Content(_Strict):
    """The memory value as rules should see it."""

    text: str = ""
    format: Literal["text", "json", "message", "none"] = "text"
    truncated: bool = False

    @model_validator(mode="after")
    def _no_value_is_empty(self) -> Content:
        if self.format == "none" and (self.text or self.truncated):
            raise ValueError("Content with format 'none' must be empty")
        return self


class Provenance(_Strict):
    """Where the stored value originally came from."""

    origin: Origin = "unknown"
    trust: Trust = "unknown"
    status: ProvenanceStatus | None = None
    writer: Actor | None = None
    writer_run_id: str | None = Field(default=None, min_length=1)


class MemoryEventBase(_Strict):
    schema_version: Literal[1]
    target: Target
    source: Actor = Field(default_factory=Actor)
    content: Content = Field(default_factory=Content)
    provenance: Provenance = Field(default_factory=Provenance)
    target_class: TargetClass = "unknown"


class MemoryMutation(MemoryEventBase):
    operation: Literal["write", "append", "delete", "clear"]

    @model_validator(mode="after")
    def _write_provenance(self) -> MemoryMutation:
        _check_operation(self, self.operation)
        provenance = self.provenance
        if provenance.status is None:
            provenance.status = "new"
        elif provenance.status != "new":
            raise ValueError("Mutation provenance status must be 'new'")
        if provenance.writer is None:
            provenance.writer = self.source.model_copy()
        elif provenance.writer != self.source:
            raise ValueError("Mutation writer must match source")
        return self


class MemoryRead(MemoryEventBase):
    operation: Literal["get", "list", "search"]
    query: str | None = None

    @model_validator(mode="after")
    def _read_provenance(self) -> MemoryRead:
        _check_operation(self, self.operation)
        if self.query is not None and self.operation != "search":
            raise ValueError("query is only valid for search")
        provenance = self.provenance
        if provenance.status is None:
            provenance.status = "missing"
        elif provenance.status == "new":
            raise ValueError("Read provenance status cannot be 'new'")
        if provenance.status in ("missing", "mismatch") and (
            provenance.origin != "unknown"
            or provenance.trust != "unknown"
            or provenance.writer is not None
            or provenance.writer_run_id is not None
        ):
            raise ValueError(
                "Unverified provenance cannot carry origin, trust, or writer"
            )
        return self

def _canonical(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite number")
        return int(value) if value.is_integer() else value
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Memory value keys must be strings")
            normalized = unicodedata.normalize("NFC", key)
            if normalized in result:
                raise ValueError("Memory value keys collide after normalization")
            result[normalized] = _canonical(item)
        return result
    raise TypeError("Unsupported memory value type")


def canonical_json(value: Any) -> str:
    """Stable JSON for hashing: sorted keys, compact, NFC strings, integral floats as ints."""

    try:
        text = json.dumps(
            _canonical(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError):
        raise MemoryInspectionError("Memory value must be JSON serializable") from None
    if len(text.encode("utf-8")) > MAX_VALUE_BYTES:
        raise MemoryInspectionError("Memory value exceeds inspection limit")
    return text


def canonical_sha256(value: Any) -> str:
    """Fingerprint of the stored value, never of its normalized text."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _leaves(value: Any, path: str, lines: list[str]) -> None:
    if isinstance(value, dict):
        if not value:
            lines.append(f"{path}: {{}}" if path else "{}")
        for key, item in value.items():
            _leaves(item, f"{path}.{key}" if path else str(key), lines)
    elif isinstance(value, list):
        if not value:
            lines.append(f"{path}: []" if path else "[]")
        for index, item in enumerate(value):
            _leaves(item, f"{path}.{index}" if path else str(index), lines)
    else:
        lines.append(f"{path}: {_scalar(value)}" if path else _scalar(value))


def _message_text(message: dict[str, Any]) -> str:
    role = message.get("role")
    if not isinstance(role, str) or not role:
        raise ValueError("Message needs a role")
    body = message.get("content")
    if body is None:
        head = ""
    elif isinstance(body, str):
        head = body
    else:
        parts: list[str] = []
        _leaves(body, "", parts)
        head = "\n".join(parts)
    # A "system: ..." prefix would read as an injected header to the role
    # spoofing rule, so the real role goes on its own bracketed line.
    lines = [f"[role={role}]", head]
    # Remaining fields, such as tool calls, stay visible so nothing hides beside
    # the text.
    rest = {key: item for key, item in message.items() if key not in ("role", "content")}
    if rest:
        _leaves(rest, "", lines)
    return "\n".join(lines)


def truncate_text(text: str) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= LIMITED_HEAD_BYTES + LIMITED_TAIL_BYTES:
        return text, False
    head = encoded[:LIMITED_HEAD_BYTES].decode("utf-8", errors="ignore")
    tail = encoded[-LIMITED_TAIL_BYTES:].decode("utf-8", errors="ignore")
    omitted = len(encoded) - LIMITED_HEAD_BYTES - LIMITED_TAIL_BYTES
    return f"{head}\n[... {omitted} bytes omitted ...]\n{tail}", True


def no_content() -> Content:
    """Content for operations that carry no value: ``delete`` and ``clear``."""

    return Content(format="none")


def normalize_content(
    value: Any, *, message: bool = False, full_text: bool = False
) -> Content:
    """Render one memory value as readable, unescaped text for rules.

    Pass ``message=True`` when the value is a chat message dict with a ``role``.
    By default long text keeps only its head and tail and is marked ``truncated``;
    ``full_text=True`` keeps all of it. Use the same mode for every scan in a run.
    A stored ``None`` is the JSON value ``null``; use ``no_content()`` for no value.
    """

    canonical_json(value)  # rejects unsupported and oversized values up front
    try:
        if message:
            if not isinstance(value, dict):
                raise TypeError("Message must be a dict")
            text, fmt = _message_text(value), "message"
        elif isinstance(value, str):
            text, fmt = value, "text"
        else:
            lines: list[str] = []
            _leaves(value, "", lines)
            text, fmt = "\n".join(lines), "json"
    except (TypeError, ValueError):
        raise MemoryInspectionError("Memory value could not be normalized") from None
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise MemoryInspectionError("Normalized memory value exceeds inspection limit")
    if full_text:
        return Content(text=text, format=fmt)
    text, truncated = truncate_text(text)
    return Content(text=text, format=fmt, truncated=truncated)
