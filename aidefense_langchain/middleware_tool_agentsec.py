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

"""LangChain middleware for tool/MCP inspection using agentsec MCPInspector.

Uses ``MCPInspector`` which provides retry logic with exponential backoff,
fail-open/closed semantics, and configuration from agentsec global state.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Mapping, Optional, Union

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

from aidefense.runtime.agentsec.inspectors.api_mcp import MCPInspector

from ._env import agentsec_kwargs_from_env

logger = logging.getLogger("aidefense.langchain.tools.agentsec")


class AIDefenseAgentsecToolMiddleware(AgentMiddleware):
    """Cisco AI Defense tool middleware using agentsec's MCPInspector.

    Unlike ``AIDefenseToolMiddleware`` which uses ``MCPInspectionClient``
    directly, this variant delegates to ``MCPInspector`` which provides:

    - Retry with configurable exponential backoff
    - Fail-open / fail-closed semantics
    - ``Decision`` objects (``allow`` / ``block`` / ``sanitize`` / ``monitor_only``)
    - Can inherit configuration from ``agentsec.protect()`` global state
    - Native async inspection support

    Parameters
    ----------
    mode : str
        ``"enforce"`` (block violations), ``"monitor"`` (log only),
        or ``"off"`` (skip inspection).  Default ``"enforce"``.
    api_key : str, optional
        AI Defense API key.  Falls back to ``agentsec`` global state or
        ``AI_DEFENSE_API_MODE_MCP_API_KEY`` env var.
    endpoint : str, optional
        MCP inspection API endpoint.  Falls back to ``agentsec`` global
        state or ``AI_DEFENSE_API_MODE_MCP_ENDPOINT`` env var.
    fail_open : bool
        Allow tool calls when the inspection API is unreachable.
        Default ``True``.
    timeout_ms : int, optional
        Inspection API timeout in milliseconds.
    retry_total : int, optional
        Total retry attempts.
    retry_backoff : float, optional
        Base backoff in seconds between retries.
    inspect_requests : bool
        Inspect tool call requests before execution.  Default ``True``.
    inspect_responses : bool
        Inspect tool results after execution.  Default ``True``.
    on_violation : callable, optional
        ``(Decision, tool_name, direction) -> None`` callback on violations.
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
        inspect_requests: bool = True,
        inspect_responses: bool = True,
        on_violation: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        if mode not in ("enforce", "monitor", "off"):
            raise ValueError(
                f"mode must be 'enforce', 'monitor', or 'off', got {mode!r}"
            )

        self.mode = mode
        self.inspect_requests = inspect_requests
        self.inspect_responses = inspect_responses
        self.on_violation = on_violation

        kwargs: Dict[str, Any] = {"fail_open": fail_open}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if endpoint is not None:
            kwargs["endpoint"] = endpoint
        if timeout_ms is not None:
            kwargs["timeout_ms"] = timeout_ms
        if retry_total is not None:
            kwargs["retry_total"] = retry_total
        if retry_backoff is not None:
            kwargs["retry_backoff"] = retry_backoff

        self.inspector = MCPInspector(**kwargs)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> "AIDefenseAgentsecToolMiddleware":
        values = agentsec_kwargs_from_env(env)
        values.update(kwargs)
        return cls(**values)

    # -- LangChain hook ----------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Union[ToolMessage, Command]
        ],
    ) -> Union[ToolMessage, Command]:
        """Wrap each tool call with pre/post inspection via MCPInspector."""
        if self.mode == "off":
            return handler(request)

        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})
        metadata = request.tool_call.get("metadata", {})

        # --- Pre-call inspection (request) ---
        if self.inspect_requests:
            decision = self.inspector.inspect_request(
                tool_name=tool_name,
                arguments=tool_args,
                metadata=metadata,
                method="tools/call",
            )
            blocked = self._process_decision(
                decision,
                request,
                tool_name,
                "request",
            )
            if blocked is not None:
                return blocked

        # --- Execute the tool ---
        tool_result = handler(request)

        # --- Post-call inspection (response) ---
        if self.inspect_responses:
            result_data = self._extract_result_data(tool_result)
            if result_data is not None:
                decision = self.inspector.inspect_response(
                    tool_name=tool_name,
                    arguments=tool_args,
                    result=result_data,
                    metadata=metadata,
                    method="tools/call",
                )
                blocked = self._process_decision(
                    decision,
                    request,
                    tool_name,
                    "response",
                )
                if blocked is not None:
                    return blocked

        return tool_result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Any
        ],
    ) -> Union[ToolMessage, Command]:
        """Wrap each tool call with pre/post inspection via MCPInspector (async)."""
        if self.mode == "off":
            return await handler(request)

        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})
        metadata = request.tool_call.get("metadata", {})

        if self.inspect_requests:
            decision = await self.inspector.ainspect_request(
                tool_name=tool_name,
                arguments=tool_args,
                metadata=metadata,
                method="tools/call",
            )
            blocked = self._process_decision(
                decision,
                request,
                tool_name,
                "request",
            )
            if blocked is not None:
                return blocked

        tool_result = await handler(request)

        if self.inspect_responses:
            result_data = self._extract_result_data(tool_result)
            if result_data is not None:
                decision = await self.inspector.ainspect_response(
                    tool_name=tool_name,
                    arguments=tool_args,
                    result=result_data,
                    metadata=metadata,
                    method="tools/call",
                )
                blocked = self._process_decision(
                    decision,
                    request,
                    tool_name,
                    "response",
                )
                if blocked is not None:
                    return blocked

        return tool_result

    # -- Internal helpers --------------------------------------------------

    def _process_decision(
        self,
        decision: Any,
        request: ToolCallRequest,
        tool_name: str,
        direction: str,
    ) -> Optional[ToolMessage]:
        """Map a Decision to enforcement or monitoring."""
        action = getattr(decision, "action", "allow")

        if action in ("allow", "monitor_only"):
            if action == "monitor_only":
                logger.info(
                    f"AI Defense monitor: tool={tool_name} "
                    f"direction={direction}"
                )
            return None

        log_msg = (
            f"AI Defense tool policy violation: tool={tool_name}, "
            f"direction={direction}, action={action}"
        )

        if self.on_violation:
            self.on_violation(decision, tool_name, direction)

        if self.mode == "enforce":
            logger.warning(f"{log_msg} — blocking tool call")
            return ToolMessage(
                content=(
                    f"Tool call '{tool_name}' was blocked by Cisco AI Defense "
                    f"({direction} policy violation)."
                ),
                tool_call_id=_tool_call_id(request, tool_name),
            )

        logger.warning(f"{log_msg} — monitor only, allowing tool call")
        return None

    @staticmethod
    def _extract_result_data(
        tool_result: Union[ToolMessage, Command],
    ) -> Optional[Dict[str, Any]]:
        """Extract inspectable data from a tool result."""
        if isinstance(tool_result, ToolMessage):
            content = tool_result.content
            if isinstance(content, str):
                return {"content": [{"type": "text", "text": content}]}
            if isinstance(content, dict):
                return content
            return {"content": str(content)}
        return None

    def close(self) -> None:
        """Release inspector resources."""
        self.inspector.close()


def _tool_call_id(request: ToolCallRequest, tool_name: str) -> str:
    tool_call_id = request.tool_call.get("id")
    if tool_call_id:
        return str(tool_call_id)
    return f"blocked-{tool_name}"
