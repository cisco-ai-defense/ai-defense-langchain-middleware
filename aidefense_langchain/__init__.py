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

"""Cisco AI Defense middleware for LangChain agents.

LLM inspection middleware (``before_model`` / ``after_model``):

1. ``AIDefenseMiddleware`` — built on ``ChatInspectionClient`` (recommended).
   Lightweight, no global state, idiomatic LangChain configuration.

2. ``AIDefenseAgentsecMiddleware`` — built on agentsec's ``LLMInspector``.
   Reuses agentsec's retry / fail-open / config-from-state machinery.

Tool / MCP inspection middleware (``wrap_tool_call``):

3. ``AIDefenseToolMiddleware`` — built on ``MCPInspectionClient``.
   Inspects tool call requests and responses against AI Defense MCP policies.

4. ``AIDefenseAgentsecToolMiddleware`` — built on agentsec's ``MCPInspector``.
   Adds retry, fail-open/closed, and agentsec configuration inheritance.

All middleware enforce *block*, *monitor*, or *off* modes via LangChain's
native ``jump_to`` / state-update patterns.
"""

from .middleware_chat_client import AIDefenseMiddleware
from .middleware_agentsec import AIDefenseAgentsecMiddleware
from .middleware_tool_inspection import AIDefenseToolMiddleware
from .middleware_tool_agentsec import AIDefenseAgentsecToolMiddleware

__all__ = [
    "AIDefenseMiddleware",
    "AIDefenseAgentsecMiddleware",
    "AIDefenseToolMiddleware",
    "AIDefenseAgentsecToolMiddleware",
]
