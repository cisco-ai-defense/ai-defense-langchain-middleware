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

"""LangChain middleware for inspecting tool / MCP calls.

Uses ``MCPInspectionClient`` to inspect tool call requests (name + arguments)
and tool call results against Cisco AI Defense policies.

Can be used standalone or composed with ``AIDefenseMiddleware`` to cover
both LLM and tool inspection in a single middleware stack.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Mapping, Optional, Union

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

from aidefense.config import Config
from aidefense.runtime import MCPInspectionClient
from aidefense.runtime.mcp_models import MCPInspectResponse

from ._env import direct_kwargs_from_env, normalize_region
from ._content import tool_result_payload

logger = logging.getLogger("aidefense.langchain.tools")


class AIDefenseToolMiddleware(AgentMiddleware):
    """Cisco AI Defense middleware for inspecting LangChain tool calls.

    Wraps every tool call with pre-call (request) and post-call (response)
    inspection via the AI Defense MCP Inspection API.

    This covers:
    - LangChain tools (``@tool`` decorated functions)
    - MCP tools registered via LangChain's MCP integration
    - Any tool executed through the agent's tool node

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
        If ``True`` (default), allow the tool call when the inspection API
        is unreachable.  If ``False``, block on API errors.
    timeout : int
        Inspection API timeout in seconds.  Default ``30``.
    inspect_requests : bool
        Inspect tool call requests (name + args) before execution.
        Default ``True``.
    inspect_responses : bool
        Inspect tool call results after execution.  Default ``True``.
    on_violation : callable, optional
        Callback ``(MCPInspectResponse, tool_name, direction) -> None``
        invoked on every violation.

    Example
    -------
    ::

        from aidefense_langchain import AIDefenseMiddleware, AIDefenseToolMiddleware
        from langchain.agents import create_agent

        agent = create_agent(
            model="openai:gpt-4.1",
            tools=[my_tool, mcp_tool],
            middleware=[
                AIDefenseMiddleware(api_key="...", mode="enforce"),
                AIDefenseToolMiddleware(api_key="...", mode="enforce"),
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
        inspect_requests: bool = True,
        inspect_responses: bool = True,
        on_violation: Optional[
            Callable[[MCPInspectResponse, str, str], None]
        ] = None,
    ) -> None:
        super().__init__()
        if mode not in ("enforce", "monitor", "off"):
            raise ValueError(
                f"mode must be 'enforce', 'monitor', or 'off', got {mode!r}"
            )

        self.mode = mode
        self.fail_open = fail_open
        self.inspect_requests = inspect_requests
        self.inspect_responses = inspect_responses
        self.on_violation = on_violation

        config = Config(region=normalize_region(region), timeout=timeout)
        self.client = MCPInspectionClient(api_key=api_key, config=config)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> "AIDefenseToolMiddleware":
        values = direct_kwargs_from_env(env)
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
        """Wrap each tool call with pre/post inspection."""
        if self.mode == "off":
            return handler(request)

        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})

        # --- Pre-call inspection (request) ---
        if self.inspect_requests:
            result = self._safe_inspect_tool_call(tool_name, tool_args)
            if result is not None and not self._is_safe(result):
                blocked = self._handle_violation(
                    result,
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
                result = self._safe_inspect_response(
                    tool_name, tool_args, result_data
                )
                if result is not None and not self._is_safe(result):
                    blocked = self._handle_violation(
                        result,
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
        """Wrap each tool call with pre/post inspection (async)."""
        if self.mode == "off":
            return await handler(request)

        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})

        if self.inspect_requests:
            result = await asyncio.to_thread(
                self._safe_inspect_tool_call,
                tool_name,
                tool_args,
            )
            if result is not None and not self._is_safe(result):
                blocked = self._handle_violation(
                    result,
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
                result = await asyncio.to_thread(
                    self._safe_inspect_response,
                    tool_name,
                    tool_args,
                    result_data,
                )
                if result is not None and not self._is_safe(result):
                    blocked = self._handle_violation(
                        result,
                        request,
                        tool_name,
                        "response",
                    )
                    if blocked is not None:
                        return blocked

        return tool_result

    # -- Internal helpers --------------------------------------------------

    def _safe_inspect_tool_call(
        self, tool_name: str, arguments: Dict[str, Any],
    ) -> Optional[MCPInspectResponse]:
        """Inspect a tool call request with fail-open protection."""
        try:
            return self.client.inspect_tool_call(
                tool_name=tool_name,
                arguments=arguments,
            )
        except Exception:
            if self.fail_open:
                logger.warning(
                    "AI Defense tool inspection failed (fail_open=True), "
                    f"allowing tool call: {tool_name}",
                    exc_info=True,
                )
                return None
            raise

    def _safe_inspect_response(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> Optional[MCPInspectResponse]:
        """Inspect a tool call response with fail-open protection."""
        try:
            return self.client.inspect_response(
                result_data=result_data,
                method="tools/call",
                params={"name": tool_name, "arguments": arguments},
            )
        except Exception:
            if self.fail_open:
                logger.warning(
                    "AI Defense tool response inspection failed "
                    f"(fail_open=True), allowing result: {tool_name}",
                    exc_info=True,
                )
                return None
            raise

    @staticmethod
    def _is_safe(result: MCPInspectResponse) -> bool:
        """Check if an MCP inspection result is safe."""
        if result.result is not None:
            return result.result.is_safe
        if result.error is not None:
            return False
        return True

    def _handle_violation(
        self,
        result: MCPInspectResponse,
        request: ToolCallRequest,
        tool_name: str,
        direction: str,
    ) -> Optional[ToolMessage]:
        """Enforce or monitor a tool call policy violation."""
        inspect_result = result.result
        classifications = []
        severity = None
        event_id = None
        if inspect_result:
            classifications = (
                [c.value for c in inspect_result.classifications]
                if inspect_result.classifications
                else []
            )
            severity = (
                inspect_result.severity.value if inspect_result.severity else None
            )
            event_id = inspect_result.event_id

        log_parts = [
            f"tool={tool_name}",
            f"direction={direction}",
            f"classifications={classifications}",
        ]
        if severity:
            log_parts.append(f"severity={severity}")
        if event_id:
            log_parts.append(f"event_id={event_id}")
        log_msg = f"AI Defense tool policy violation: {', '.join(log_parts)}"

        if self.on_violation:
            self.on_violation(result, tool_name, direction)

        if self.mode == "enforce":
            logger.warning(f"{log_msg} — blocking tool call")
            return ToolMessage(
                content=(
                    f"Tool call '{tool_name}' was blocked by Cisco AI Defense "
                    f"({direction} policy violation)."
                ),
                tool_call_id=_tool_call_id(request, tool_name),
            )

        # monitor mode
        logger.warning(f"{log_msg} — monitor only, allowing tool call")
        return None

    def close(self) -> None:
        """Release underlying HTTP session resources."""
        session = getattr(self.client, "_session", None)
        if session is not None:
            session.close()

    @staticmethod
    def _extract_result_data(
        tool_result: Union[ToolMessage, Command],
    ) -> Optional[Dict[str, Any]]:
        """Extract inspectable data from a tool result."""
        if isinstance(tool_result, ToolMessage):
            return tool_result_payload(tool_result.content)
        if isinstance(tool_result, Command):
            return None
        return None


def _tool_call_id(request: ToolCallRequest, tool_name: str) -> str:
    tool_call_id = request.tool_call.get("id")
    if tool_call_id:
        return str(tool_call_id)
    return f"blocked-{tool_name}"
