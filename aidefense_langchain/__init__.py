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
