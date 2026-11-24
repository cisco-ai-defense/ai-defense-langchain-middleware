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

"""Optional LangChain memory protection backed by local Quarry rules."""

from .middleware import AIDefenseMemoryMiddleware
from .backends.checkpoint import GuardedCheckpointSaver
from .backends.history import GuardedChatMessageHistory
from .backends.provenance import memory_origin
from .backends.store import GuardedMemoryStore
from .backends.store_types import MemoryDecision, MemoryInspectionFailed, MemoryWriteBlocked
from .events import MemoryMutation, MemoryRead

__all__ = [
    "AIDefenseMemoryMiddleware",
    "GuardedCheckpointSaver",
    "GuardedChatMessageHistory",
    "GuardedMemoryStore",
    "MemoryDecision",
    "MemoryInspectionFailed",
    "MemoryMutation",
    "MemoryRead",
    "MemoryWriteBlocked",
    "memory_origin",
]
