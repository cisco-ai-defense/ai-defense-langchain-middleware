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

"""Host-asserted origin and value-bound provenance for LangGraph Store items."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, get_args

from ..events import Actor, Origin, Provenance, Trust, canonical_sha256

SIDECAR_VERSION = 1


@dataclass(frozen=True, slots=True)
class MemoryOrigin:
    kind: Origin = "unknown"
    trust: Trust = "unknown"


_origin: ContextVar[MemoryOrigin] = ContextVar("aidefense_memory_origin", default=MemoryOrigin())


def current_memory_origin() -> MemoryOrigin:
    return _origin.get()


@contextmanager
def memory_origin(kind: Origin, trust: Trust) -> Iterator[None]:
    """Annotate a write from trusted application context, never from saved text."""

    if kind not in get_args(Origin):
        raise ValueError("Unsupported memory origin kind")
    if trust not in get_args(Trust):
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


def sidecar_key(namespace: tuple[str, ...], key: str) -> str:
    identity = json.dumps([list(namespace), key], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def sidecar_record(value: Any, origin: MemoryOrigin, writer: Actor) -> dict[str, Any]:
    return {
        "version": SIDECAR_VERSION,
        "origin": origin.kind,
        "trust": origin.trust,
        "writer": writer.model_dump(mode="json", exclude_none=True),
        "value_sha256": canonical_sha256(value),
    }


def read_provenance(value: Any, sidecar: Any) -> Provenance:
    """A missing or altered sidecar never upgrades a memory item's trust."""

    if not isinstance(sidecar, dict):
        return Provenance(status="missing")
    try:
        if sidecar.get("version") != SIDECAR_VERSION:
            raise ValueError("Unsupported sidecar version")
        if sidecar.get("value_sha256") != canonical_sha256(value):
            return Provenance(status="mismatch")
        return Provenance(
            origin=sidecar["origin"],
            trust=sidecar["trust"],
            status="verified",
            writer=Actor.model_validate(sidecar["writer"]),
        )
    except (KeyError, ValueError, TypeError):
        return Provenance(status="mismatch")
