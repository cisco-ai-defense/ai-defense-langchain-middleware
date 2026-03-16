"""Cisco AI Defense middleware for LangChain agents.

Two middleware implementations are provided:

1. ``AIDefenseMiddleware`` — built on ``ChatInspectionClient`` (recommended).
   Lightweight, no global state, idiomatic LangChain configuration.

2. ``AIDefenseAgentsecMiddleware`` — built on agentsec's ``LLMInspector``.
   Reuses agentsec's retry / fail-open / config-from-state machinery.

Both inspect LLM inputs and outputs against Cisco AI Defense policies and
enforce *block*, *monitor*, or *off* modes via LangChain's native
``jump_to`` / state-update patterns.
"""

from .middleware_chat_client import AIDefenseMiddleware
from .middleware_agentsec import AIDefenseAgentsecMiddleware

__all__ = [
    "AIDefenseMiddleware",
    "AIDefenseAgentsecMiddleware",
]
