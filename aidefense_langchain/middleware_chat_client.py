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

"""LangChain middleware using ``ChatInspectionClient`` (recommended).

This is the lightweight approach: no global state, no monkey-patching,
explicit configuration at middleware construction time.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, Mapping, Optional

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    hook_config,
)
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

from aidefense.config import Config
from aidefense.runtime import (
    ChatInspectionClient,
    InspectResponse,
    Message,
    Metadata,
    Role,
    InspectionConfig,
    Rule,
    RuleName,
)

from ._env import direct_kwargs_from_env, normalize_region
from ._content import flatten_content_text

logger = logging.getLogger("aidefense.langchain")


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------

_LC_TYPE_TO_ROLE = {
    "human": Role.USER,
    "ai": Role.ASSISTANT,
    "system": Role.SYSTEM,
}


def _langchain_messages_to_aidefense(lc_messages: list) -> List[Message]:
    """Convert LangChain message objects to AI Defense ``Message`` list."""
    result: List[Message] = []
    for msg in lc_messages:
        role = _LC_TYPE_TO_ROLE.get(getattr(msg, "type", ""), Role.USER)
        content = flatten_content_text(msg.content)
        result.append(Message(role=role, content=content))
    return result


def _build_metadata(
    user: Optional[str] = None,
    src_app: Optional[str] = None,
    extra: Optional[dict] = None,
) -> Optional[Metadata]:
    """Build a ``Metadata`` object from optional fields."""
    kwargs: dict[str, Any] = {}
    if user:
        kwargs["user"] = user
    if src_app:
        kwargs["src_app"] = src_app
    if extra:
        for key in ("dst_app", "sni", "dst_ip", "src_ip", "dst_host",
                     "user_agent", "client_transaction_id"):
            if key in extra:
                kwargs[key] = extra[key]
    return Metadata(**kwargs) if kwargs else None


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class AIDefenseMiddleware(AgentMiddleware):
    """Cisco AI Defense middleware for LangChain agents.

    Uses ``ChatInspectionClient`` directly — lightweight, no global state,
    and idiomatic LangChain configuration.

    Parameters
    ----------
    api_key : str
        Cisco AI Defense API key.
    region : str
        AI Defense region (e.g. ``"us-west-2"``, ``"eu-central-1"``,
        ``"ap-northeast-1"``).  Default ``"us-west-2"``.
    mode : str
        Enforcement mode: ``"enforce"`` (block violations), ``"monitor"``
        (log only), or ``"off"`` (skip inspection).  Default ``"enforce"``.
    fail_open : bool
        If ``True`` (default), allow the request when the inspection API is
        unreachable.  If ``False``, block on API errors.
    timeout : int
        Inspection API timeout in seconds.  Default ``30``.
    rules : list, optional
        List of rule names or ``Rule`` objects to enable.
    user : str, optional
        User identity attached to every inspection request.
    src_app : str, optional
        Source application name attached to every inspection request.
    on_violation : callable, optional
        Callback ``(InspectResponse, direction: str) -> None`` invoked on
        every violation (in both enforce and monitor modes).

    Example
    -------
    ::

        from aidefense_langchain import AIDefenseMiddleware
        from langchain.agents import create_agent

        agent = create_agent(
            model="openai:gpt-4.1",
            tools=[...],
            middleware=[
                AIDefenseMiddleware(api_key="...", region="us-west-2", mode="enforce"),
            ],
        )
    """

    def __init__(
        self,
        api_key: str,
        region: str = "us-west-2",
        mode: str = "enforce",
        fail_open: bool = True,
        timeout: int = 30,
        rules: Optional[list] = None,
        user: Optional[str] = None,
        src_app: Optional[str] = None,
        on_violation: Optional[Callable[[InspectResponse, str], None]] = None,
    ) -> None:
        super().__init__()
        if mode not in ("enforce", "monitor", "off"):
            raise ValueError(f"mode must be 'enforce', 'monitor', or 'off', got {mode!r}")

        self.mode = mode
        self.fail_open = fail_open
        self.on_violation = on_violation

        self._metadata = _build_metadata(user=user, src_app=src_app)
        self._inspection_config = self._build_inspection_config(rules)

        config = Config(region=normalize_region(region), timeout=timeout)
        self.client = ChatInspectionClient(api_key=api_key, config=config)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> "AIDefenseMiddleware":
        values = direct_kwargs_from_env(env)
        values.update(kwargs)
        return cls(**values)

    # -- LangChain hooks ---------------------------------------------------

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: AgentState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Inspect input messages before they reach the LLM."""
        if self.mode == "off":
            return None

        messages = _langchain_messages_to_aidefense(state["messages"])
        result = self._safe_inspect(messages)

        if result is not None and not result.is_safe:
            return self._handle_violation(result, "input")
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: AgentState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Inspect input messages before they reach the LLM (async)."""
        if self.mode == "off":
            return None

        messages = _langchain_messages_to_aidefense(state["messages"])
        result = await asyncio.to_thread(self._safe_inspect, messages)

        if result is not None and not result.is_safe:
            return self._handle_violation(result, "input")
        return None

    @hook_config(can_jump_to=["end"])
    def after_model(
        self, state: AgentState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Inspect the LLM response after it is received."""
        if self.mode == "off":
            return None

        messages = _langchain_messages_to_aidefense(state["messages"])
        result = self._safe_inspect(messages)

        if result is not None and not result.is_safe:
            return self._handle_violation(result, "output")
        return None

    @hook_config(can_jump_to=["end"])
    async def aafter_model(
        self, state: AgentState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Inspect the LLM response after it is received (async)."""
        if self.mode == "off":
            return None

        messages = _langchain_messages_to_aidefense(state["messages"])
        result = await asyncio.to_thread(self._safe_inspect, messages)

        if result is not None and not result.is_safe:
            return self._handle_violation(result, "output")
        return None

    # -- Internal helpers --------------------------------------------------

    def _safe_inspect(self, messages: List[Message]) -> Optional[InspectResponse]:
        """Call the inspection API with fail-open protection."""
        try:
            return self.client.inspect_conversation(
                messages=messages,
                metadata=self._metadata,
                config=self._inspection_config,
            )
        except Exception:
            if self.fail_open:
                logger.warning(
                    "AI Defense inspection failed, allowing request (fail_open=True)",
                    exc_info=True,
                )
                return None
            raise

    def _handle_violation(
        self, result: InspectResponse, direction: str,
    ) -> dict[str, Any] | None:
        """Enforce or monitor a policy violation."""
        classifications = (
            [c.value for c in result.classifications] if result.classifications else []
        )
        log_parts = [
            f"direction={direction}",
            f"action={result.action.value}",
            f"classifications={classifications}",
        ]
        if result.severity:
            log_parts.append(f"severity={result.severity.value}")
        if result.event_id:
            log_parts.append(f"event_id={result.event_id}")
        log_msg = f"AI Defense policy violation: {', '.join(log_parts)}"

        if self.on_violation:
            self.on_violation(result, direction)

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

    @staticmethod
    def _build_inspection_config(
        rules: Optional[list],
    ) -> Optional[InspectionConfig]:
        """Convert a list of rule names / Rule objects into InspectionConfig."""
        if not rules:
            return None
        built: List[Rule] = []
        for r in rules:
            if isinstance(r, Rule):
                built.append(r)
            elif isinstance(r, str):
                try:
                    built.append(Rule(rule_name=RuleName(r)))
                except ValueError:
                    built.append(Rule(rule_name=r))
            elif isinstance(r, dict):
                rn = r.get("rule_name")
                if rn and not isinstance(rn, RuleName):
                    try:
                        rn = RuleName(rn)
                    except ValueError:
                        pass
                built.append(Rule(rule_name=rn, entity_types=r.get("entity_types")))
        return InspectionConfig(enabled_rules=built) if built else None
