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

Callback handler (``callbacks=`` parameter — works with ``create_react_agent``):

5. ``AIDefenseCallbackHandler`` — built on ``ChatInspectionClient`` +
   ``MCPInspectionClient``.  Drop-in for any LangChain construct that accepts
   ``callbacks=``, including agents built with ``create_react_agent``.
   Raises ``AIDefenseViolationError`` in enforce mode.

All middleware enforce *block*, *monitor*, or *off* modes via LangChain's
native ``jump_to`` / state-update patterns.
"""

from .middleware_chat_client import AIDefenseMiddleware
from .middleware_agentsec import AIDefenseAgentsecMiddleware
from .middleware_tool_inspection import AIDefenseToolMiddleware
from .middleware_tool_agentsec import AIDefenseAgentsecToolMiddleware
try:
    from .react_agent import (
        AIDefenseHooks,
        AIDefenseToolNode,
        AIDefenseViolationError,
        create_aidefense_react_agent,
    )
except ImportError as _react_err:
    _missing = getattr(_react_err, "name", "") or ""
    if not _missing.startswith("langgraph"):
        raise
    raise ImportError(
        "aidefense_langchain's create_react_agent integration requires "
        "langgraph>=0.2.27 (for pre_model_hook, post_model_hook, and "
        "ToolNode.wrap_tool_call). Upgrade with: "
        "pip install 'langgraph>=0.2.27'"
    ) from _react_err

__all__ = [
    # create_agent middleware
    "AIDefenseMiddleware",
    "AIDefenseAgentsecMiddleware",
    "AIDefenseToolMiddleware",
    "AIDefenseAgentsecToolMiddleware",
    # create_react_agent primitives
    "AIDefenseHooks",
    "AIDefenseToolNode",
    "AIDefenseViolationError",
    # create_react_agent convenience wrapper
    "create_aidefense_react_agent",
]
