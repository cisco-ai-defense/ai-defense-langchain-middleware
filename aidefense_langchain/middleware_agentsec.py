# Copyright 2025 Cisco Systems, Inc. and its affiliates
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

"""LangChain middleware using agentsec's ``LLMInspector``.

This approach reuses agentsec's retry logic, fail-open handling, and
configuration from ``_state`` / ``protect()``.  It is heavier than the
``ChatInspectionClient`` variant but may be preferred when the rest of the
application already uses ``agentsec.protect()``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

from aidefense.runtime.agentsec.decision import Decision
from aidefense.runtime.agentsec.inspectors.api_llm import LLMInspector

logger = logging.getLogger("aidefense.langchain.agentsec")


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------

_LC_TYPE_TO_ROLE_STR = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
}


def _langchain_messages_to_dicts(lc_messages: list) -> List[Dict[str, Any]]:
    """Convert LangChain message objects to plain dicts for LLMInspector."""
    result: List[Dict[str, Any]] = []
    for msg in lc_messages:
        role = _LC_TYPE_TO_ROLE_STR.get(getattr(msg, "type", ""), "user")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        result.append({"role": role, "content": content})
    return result


def _build_metadata(
    user: Optional[str] = None,
    src_app: Optional[str] = None,
    extra: Optional[dict] = None,
) -> Dict[str, Any]:
    """Build a metadata dict for ``LLMInspector.inspect_conversation``."""
    meta: Dict[str, Any] = {}
    if user:
        meta["user"] = user
    if src_app:
        meta["src_app"] = src_app
    if extra:
        meta.update(extra)
    return meta


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class AIDefenseAgentsecMiddleware(AgentMiddleware):
    """Cisco AI Defense middleware backed by agentsec's ``LLMInspector``.

    This variant reuses agentsec's built-in retry with exponential backoff,
    fail-open/closed semantics, and the ``Decision`` model.  It pulls
    configuration from agentsec's global state when ``protect()`` has been
    called, or accepts explicit parameters.

    Parameters
    ----------
    mode : str
        Enforcement mode: ``"enforce"``, ``"monitor"``, or ``"off"``.
    api_key : str, optional
        AI Defense API key.  Falls back to agentsec state / env var.
    endpoint : str, optional
        AI Defense API endpoint.  Falls back to agentsec state / env var.
    fail_open : bool
        Allow on inspection errors.  Default ``True``.
    timeout_ms : int, optional
        Inspection timeout in milliseconds.
    retry_total : int, optional
        Total retry attempts (default 1 = no retry).
    retry_backoff : float, optional
        Exponential backoff factor in seconds.
    rules : list, optional
        Default rules for inspection.
    user : str, optional
        User identity for every request.
    src_app : str, optional
        Source app name for every request.
    on_violation : callable, optional
        Callback ``(Decision, direction: str) -> None`` on every violation.

    Example
    -------
    ::

        from aidefense.runtime import agentsec
        from aidefense_langchain import AIDefenseAgentsecMiddleware
        from langchain.agents import create_agent

        # Option A: rely on protect() for config
        agentsec.protect(config="agentsec.yaml")
        middleware = AIDefenseAgentsecMiddleware(mode="enforce")

        # Option B: pass config explicitly (no protect() needed)
        middleware = AIDefenseAgentsecMiddleware(
            mode="enforce",
            api_key="...",
            endpoint="https://...",
        )

        agent = create_agent(
            model="openai:gpt-4.1",
            tools=[...],
            middleware=[middleware],
        )
    """

    def __init__(
        self,
        mode: str = "enforce",
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        fail_open: bool = True,
        timeout_ms: Optional[int] = None,
        retry_total: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        rules: Optional[list] = None,
        user: Optional[str] = None,
        src_app: Optional[str] = None,
        on_violation: Optional[Callable[[Decision, str], None]] = None,
    ) -> None:
        super().__init__()
        if mode not in ("enforce", "monitor", "off"):
            raise ValueError(f"mode must be 'enforce', 'monitor', or 'off', got {mode!r}")

        self.mode = mode
        self.on_violation = on_violation
        self._metadata = _build_metadata(user=user, src_app=src_app)

        self.inspector = LLMInspector(
            api_key=api_key,
            endpoint=endpoint,
            default_rules=rules or [],
            fail_open=fail_open,
            timeout_ms=timeout_ms,
            retry_total=retry_total,
            retry_backoff=retry_backoff,
        )

    # -- LangChain hooks ---------------------------------------------------

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: AgentState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Inspect input messages before they reach the LLM."""
        if self.mode == "off":
            return None

        messages = _langchain_messages_to_dicts(state["messages"])
        decision = self.inspector.inspect_conversation(messages, self._metadata)
        return self._process_decision(decision, "input")

    @hook_config(can_jump_to=["end"])
    def after_model(
        self, state: AgentState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Inspect the LLM response after it is received."""
        if self.mode == "off":
            return None

        messages = _langchain_messages_to_dicts(state["messages"])
        decision = self.inspector.inspect_conversation(messages, self._metadata)
        return self._process_decision(decision, "output")

    # -- Internal helpers --------------------------------------------------

    def _process_decision(
        self, decision: Decision, direction: str,
    ) -> dict[str, Any] | None:
        """Map an agentsec ``Decision`` to a LangChain state update."""
        if decision.action == "allow" or decision.action == "monitor_only":
            return None

        # Violation detected (block or sanitize)
        log_parts = [
            f"direction={direction}",
            f"action={decision.action}",
            f"reasons={decision.reasons}",
        ]
        if decision.severity:
            log_parts.append(f"severity={decision.severity}")
        if decision.event_id:
            log_parts.append(f"event_id={decision.event_id}")
        log_msg = f"AI Defense policy violation: {', '.join(log_parts)}"

        if self.on_violation:
            self.on_violation(decision, direction)

        if self.mode == "enforce":
            logger.warning(f"{log_msg} — blocking request")
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "This request was blocked by Cisco AI Defense "
                            f"({direction} policy violation)."
                        )
                    )
                ],
                "jump_to": "end",
            }

        # monitor mode
        logger.warning(f"{log_msg} — monitor only, allowing request")
        return None
