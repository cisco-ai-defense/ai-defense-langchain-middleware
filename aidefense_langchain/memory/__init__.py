# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Optional LangChain memory protection backed by local Quarry rules."""

from .middleware import AIDefenseMemoryMiddleware
from .backends.checkpoint import GuardedCheckpointSaver
from .backends.history import GuardedChatMessageHistory
from .backends.provenance import memory_origin
from .backends.store import GuardedMemoryStore, MemoryDecision, MemoryInspectionFailed, MemoryWriteBlocked

__all__ = [
    "AIDefenseMemoryMiddleware",
    "GuardedCheckpointSaver",
    "GuardedChatMessageHistory",
    "GuardedMemoryStore",
    "MemoryDecision",
    "MemoryInspectionFailed",
    "MemoryWriteBlocked",
    "memory_origin",
]
