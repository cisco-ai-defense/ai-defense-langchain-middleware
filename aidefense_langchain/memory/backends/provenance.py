# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Host-asserted origin and value-bound provenance for LangGraph Store items."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Literal

OriginKind = Literal["user", "external", "model", "system", "unknown"]
TrustLevel = Literal["trusted", "untrusted", "unknown"]
ProvenanceStatus = Literal["verified", "missing", "mismatch"]


@dataclass(frozen=True, slots=True)
class MemoryOrigin:
    kind: OriginKind = "unknown"
    trust: TrustLevel = "unknown"


_origin: ContextVar[MemoryOrigin] = ContextVar("aidefense_memory_origin", default=MemoryOrigin())


def current_memory_origin() -> MemoryOrigin:
    return _origin.get()


@contextmanager
def memory_origin(kind: OriginKind, trust: TrustLevel) -> Iterator[None]:
    """Annotate a write from trusted application context, never from saved text."""

    if kind not in ("user", "external", "model", "system", "unknown"):
        raise ValueError("Unsupported memory origin kind")
    if trust not in ("trusted", "untrusted", "unknown"):
        raise ValueError("Unsupported memory trust level")
    token = _origin.set(MemoryOrigin(kind, trust))
    try:
        yield
    finally:
        _origin.reset(token)


def snapshot_json(value: Any) -> Any:
    """Return an independent JSON value for both inspection and persistence."""

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Memory value must be JSON serializable") from exc


def value_digest(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sidecar_key(namespace: tuple[str, ...], key: str) -> str:
    identity = json.dumps([list(namespace), key], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def read_provenance(value: Any, sidecar: Any) -> tuple[MemoryOrigin, str, ProvenanceStatus]:
    """A missing or altered sidecar never upgrades a memory item's trust."""

    if not isinstance(sidecar, dict):
        return MemoryOrigin(), "unknown", "missing"
    try:
        if sidecar.get("version") != 1:
            raise ValueError
        if sidecar.get("value_sha256") != value_digest(value):
            return MemoryOrigin(), "unknown", "mismatch"
        kind, trust, writer = (
            sidecar["origin_kind"], sidecar["trust_level"], sidecar["writer_tool"]
        )
        if kind not in ("user", "external", "model", "system", "unknown"):
            raise ValueError
        if trust not in ("trusted", "untrusted", "unknown") or not isinstance(writer, str):
            raise ValueError
        return MemoryOrigin(kind, trust), writer, "verified"
    except (KeyError, ValueError, TypeError):
        return MemoryOrigin(), "unknown", "mismatch"
