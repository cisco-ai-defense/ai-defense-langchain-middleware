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

"""AI Defense integration for ``create_react_agent``.

Provides two composable primitives and a convenience wrapper:

``AIDefenseHooks``
    Supplies ``pre_model_hook`` and ``post_model_hook`` callables that
    inspect LLM input/output via ``create_react_agent``'s native hook API.

``AIDefenseToolNode``
    A ``ToolNode`` subclass that inspects every tool call request and
    response using ``ToolNode``'s native ``wrap_tool_call`` interceptor.

``create_aidefense_react_agent``
    Drop-in replacement for ``create_react_agent`` that wires both
    primitives in automatically — change one function name, add your
    AI Defense config, done.

Usage — primitives::

    from aidefense_langchain import AIDefenseHooks, AIDefenseToolNode
    from langgraph.prebuilt import create_react_agent

    hooks = AIDefenseHooks(api_key="...", mode="enforce")
    tool_node = AIDefenseToolNode(tools, api_key="...", mode="enforce")

    agent = create_react_agent(
        model=llm,
        tools=tool_node,
        pre_model_hook=hooks.pre_model_hook,
        post_model_hook=hooks.post_model_hook,
    )

Usage — convenience wrapper::

    from aidefense_langchain import create_aidefense_react_agent

    agent = create_aidefense_react_agent(
        model=llm,
        tools=tools,
        api_key="...",
        mode="enforce",
    )

In ``"enforce"`` mode both primitives raise ``AIDefenseViolationError`` on
a violation, aborting the agent invocation.  In ``"monitor"`` mode
violations are logged and the ``on_violation`` callback fires, but
execution continues.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any, Callable, List, Mapping, Optional, Sequence, Union

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
try:
    from langgraph.graph import CompiledStateGraph
except ImportError:
    from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent as _create_react_agent
from langgraph.prebuilt.tool_node import ToolCallRequest

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
from aidefense.runtime.mcp_inspect import MCPInspectionClient

from ._content import flatten_content_text, tool_result_payload
from ._env import direct_kwargs_from_env, normalize_region
from .middleware_chat_client import (
    _build_metadata,
    _langchain_messages_to_aidefense,
)
from .middleware_chat_client import AIDefenseMiddleware as _MW

logger = logging.getLogger("aidefense.langchain.react")

_LC_TYPE_TO_ROLE: dict[str, Role] = {
    "human": Role.USER,
    "ai": Role.ASSISTANT,
    "system": Role.SYSTEM,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_mode(mode: str) -> None:
    if mode not in ("enforce", "monitor", "off"):
        raise ValueError(
            f"mode must be 'enforce', 'monitor', or 'off', got {mode!r}"
        )


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class AIDefenseViolationError(RuntimeError):
    """Raised in enforce mode when AI Defense blocks a request or response.

    Attributes
    ----------
    direction : str
        Where the violation was detected: ``"input"``, ``"output"``,
        ``"tool '<name>' input"``, or ``"tool '<name>' output"``.
    response : InspectResponse
        The full inspection result from the AI Defense SDK.
    """

    def __init__(self, direction: str, response: InspectResponse) -> None:
        self.direction = direction
        self.response = response
        explanation = response.explanation or (
            response.action.value if response.action else "policy violation"
        )
        super().__init__(f"AI Defense blocked {direction}: {explanation}")


# ---------------------------------------------------------------------------
# Shared inspection logic
# ---------------------------------------------------------------------------

class _Guard:
    """Holds shared config and inspection methods used by both primitives."""

    def __init__(
        self,
        api_key: str,
        region: str,
        mode: str,
        fail_open: bool,
        timeout: int,
        config: Optional[Config],
        rules: Optional[list],
        user: Optional[str],
        src_app: Optional[str],
        on_violation: Optional[Callable[[InspectResponse, str], None]],
    ) -> None:
        self.mode = mode
        self.fail_open = fail_open
        self.on_violation = on_violation
        self._metadata = _build_metadata(user=user, src_app=src_app)
        self._inspection_config = _MW._build_inspection_config(rules)

        if config is None:
            config = Config(region=normalize_region(region), timeout=timeout)

        self.chat_client = ChatInspectionClient(api_key=api_key, config=config)
        self.mcp_client = MCPInspectionClient(api_key=api_key, config=config)

    def inspect_messages(self, messages: List[BaseMessage], direction: str) -> None:
        # Only inspect human/ai/system messages with non-empty text content.
        # - ToolMessage is skipped: already inspected by AIDefenseToolNode.
        # - AIMessage with tool_calls and empty content is skipped: would fail
        #   SDK validation ("each message must have non-empty string content").
        # - Multimodal content (list of content blocks) is supported via
        #   flatten_content_text, which extracts text from {"type":"text",...}
        #   blocks — so structured content is inspected, not silently skipped.
        inspectable = [
            m for m in messages
            if getattr(m, "type", "") in _LC_TYPE_TO_ROLE
            and flatten_content_text(m.content).strip()
        ]
        ad_messages = _langchain_messages_to_aidefense(inspectable)
        if not ad_messages:
            return
        try:
            result = self.chat_client.inspect_conversation(
                messages=ad_messages,
                metadata=self._metadata,
                config=self._inspection_config,
            )
        except Exception:
            if self.fail_open:
                logger.warning(
                    "AI Defense inspection failed, allowing (fail_open=True)",
                    exc_info=True,
                )
                return
            raise
        if not result.is_safe:
            self._handle_violation(result, direction)

    def inspect_llm_response(self, text: str) -> None:
        try:
            result = self.chat_client.inspect_response(
                response=text,
                metadata=self._metadata,
                config=self._inspection_config,
            )
        except Exception:
            if self.fail_open:
                logger.warning(
                    "AI Defense inspection failed, allowing (fail_open=True)",
                    exc_info=True,
                )
                return
            raise
        if not result.is_safe:
            self._handle_violation(result, "output")

    def inspect_tool_request(self, tool_name: str, arguments: dict) -> None:
        try:
            result = self.mcp_client.inspect_tool_call(tool_name, arguments)
        except Exception:
            if self.fail_open:
                logger.warning(
                    "AI Defense tool-call inspection failed, allowing (fail_open=True)",
                    exc_info=True,
                )
                return
            raise
        if result.result and not result.result.is_safe:
            self._handle_violation(result.result, f"tool '{tool_name}' input")

    def inspect_tool_response(self, tool_name: str, output: Any) -> None:
        payload = tool_result_payload(output)
        try:
            result = self.mcp_client.inspect_response(
                result_data=payload,
                method="tools/call",
                params={"name": tool_name},
            )
        except Exception:
            if self.fail_open:
                logger.warning(
                    "AI Defense tool-response inspection failed, allowing (fail_open=True)",
                    exc_info=True,
                )
                return
            raise
        if result.result and not result.result.is_safe:
            self._handle_violation(result.result, f"tool '{tool_name}' output")

    def _handle_violation(self, result: InspectResponse, direction: str) -> None:
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
            try:
                self.on_violation(result, direction)
            except Exception:
                logger.exception("AI Defense on_violation callback failed")

        if self.mode == "enforce":
            logger.warning("%s — blocking", log_msg)
            raise AIDefenseViolationError(direction, result)

        logger.warning("%s — monitor only, allowing", log_msg)

    def close(self) -> None:
        for client in (self.chat_client, self.mcp_client):
            handler = getattr(client, "_request_handler", None)
            session = getattr(handler, "_session", None) if handler else None
            if session is not None:
                session.close()


# ---------------------------------------------------------------------------
# Primitive 1: AIDefenseHooks
# ---------------------------------------------------------------------------

class AIDefenseHooks:
    """AI Defense inspection hooks for ``create_react_agent``.

    Provides ``pre_model_hook`` and ``post_model_hook`` callables that plug
    directly into ``create_react_agent``'s native hook parameters to inspect
    every LLM input and output.

    In ``"enforce"`` mode violations raise ``AIDefenseViolationError``.
    In ``"monitor"`` mode violations are logged (and ``on_violation`` fires)
    but execution continues.

    Parameters
    ----------
    api_key : str
        Cisco AI Defense API key.
    region : str
        AI Defense region.  Default ``"us-west-2"``.
    mode : str
        ``"enforce"``, ``"monitor"``, or ``"off"``.  Default ``"enforce"``.
    fail_open : bool
        Allow requests on API errors.  Default ``True``.
    timeout : int
        Inspection timeout in seconds.  Default ``30``.
    config : Config, optional
        Pre-built SDK ``Config``.  Shared with ``AIDefenseToolNode`` to
        avoid ``Config`` singleton conflicts.
    rules : list, optional
        Rule names or ``Rule`` objects to enable.
    user : str, optional
        User identity for audit.
    src_app : str, optional
        Source application name for audit.
    on_violation : callable, optional
        ``(InspectResponse, direction: str) -> None``.

    Example
    -------
    ::

        from langgraph.prebuilt import create_react_agent
        from aidefense_langchain import AIDefenseHooks, AIDefenseToolNode

        hooks = AIDefenseHooks(api_key="...", mode="enforce")
        tool_node = AIDefenseToolNode(tools, api_key="...", mode="enforce")

        agent = create_react_agent(
            model=llm,
            tools=tool_node,
            pre_model_hook=hooks.pre_model_hook,
            post_model_hook=hooks.post_model_hook,
        )
    """

    def __init__(
        self,
        api_key: str,
        region: str = "us-west-2",
        mode: str = "enforce",
        fail_open: bool = True,
        timeout: int = 30,
        config: Optional[Config] = None,
        rules: Optional[list] = None,
        user: Optional[str] = None,
        src_app: Optional[str] = None,
        on_violation: Optional[Callable[[InspectResponse, str], None]] = None,
    ) -> None:
        _validate_mode(mode)
        self._guard = _Guard(
            api_key=api_key,
            region=region,
            mode=mode,
            fail_open=fail_open,
            timeout=timeout,
            config=config,
            rules=rules,
            user=user,
            src_app=src_app,
            on_violation=on_violation,
        )

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        **overrides: Any,
    ) -> "AIDefenseHooks":
        """Construct from environment variables (``AIDEFENSE_API_KEY``, etc.)."""
        kwargs = direct_kwargs_from_env(env)
        kwargs.update(overrides)
        return cls(**kwargs)

    def pre_model_hook(self, state: Any) -> dict[str, Any]:
        """Inspect the full message history before each LLM call.

        Pass to ``create_react_agent(pre_model_hook=hooks.pre_model_hook)``.
        """
        if self._guard.mode == "off":
            return {}
        messages = _get_messages(state)
        self._guard.inspect_messages(messages, "input")
        return {}

    def post_model_hook(self, state: Any) -> dict[str, Any]:
        """Inspect the LLM response after each LLM call.

        Pass to ``create_react_agent(post_model_hook=hooks.post_model_hook)``.
        """
        if self._guard.mode == "off":
            return {}
        messages = _get_messages(state)
        last = messages[-1] if messages else None
        if last is None:
            return {}
        text = flatten_content_text(last.content)
        if text:
            self._guard.inspect_llm_response(text)
        return {}

    def close(self) -> None:
        """Release underlying HTTP session resources."""
        self._guard.close()


# ---------------------------------------------------------------------------
# Primitive 2: AIDefenseToolNode
# ---------------------------------------------------------------------------

class AIDefenseToolNode(ToolNode):
    """A ``ToolNode`` with Cisco AI Defense inspection built in.

    Inspects every tool call request (name + arguments) and response using
    ``ToolNode``'s native ``wrap_tool_call`` interceptor.  Pass in place of
    a plain tools list to ``create_react_agent``.

    In ``"enforce"`` mode violations raise ``AIDefenseViolationError``.
    In ``"monitor"`` mode violations are logged but execution continues.

    Parameters
    ----------
    tools : sequence
        Tools to register — same as ``ToolNode``.
    api_key : str
        Cisco AI Defense API key.
    region : str
        AI Defense region.  Default ``"us-west-2"``.
    mode : str
        ``"enforce"``, ``"monitor"``, or ``"off"``.  Default ``"enforce"``.
    fail_open : bool
        Allow requests on API errors.  Default ``True``.
    timeout : int
        Inspection timeout in seconds.  Default ``30``.
    config : Config, optional
        Pre-built SDK ``Config``.  Share with ``AIDefenseHooks`` to avoid
        ``Config`` singleton conflicts.
    rules : list, optional
        Rule names or ``Rule`` objects to enable.
    user : str, optional
        User identity for audit.
    src_app : str, optional
        Source application name for audit.
    on_violation : callable, optional
        ``(InspectResponse, direction: str) -> None``.
    **tool_node_kwargs :
        Forwarded to ``ToolNode`` (``name``, ``tags``, ``messages_key``).
        ``handle_tool_errors`` defaults to ``False`` so ``AIDefenseViolationError``
        always propagates rather than being converted to a ``ToolMessage``.

    Example
    -------
    ::

        from langgraph.prebuilt import create_react_agent
        from aidefense_langchain import AIDefenseHooks, AIDefenseToolNode

        hooks = AIDefenseHooks(api_key="...", mode="enforce")
        tool_node = AIDefenseToolNode(tools, api_key="...", mode="enforce")

        agent = create_react_agent(
            model=llm,
            tools=tool_node,
            pre_model_hook=hooks.pre_model_hook,
            post_model_hook=hooks.post_model_hook,
        )
    """

    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable]],
        *,
        api_key: str,
        region: str = "us-west-2",
        mode: str = "enforce",
        fail_open: bool = True,
        timeout: int = 30,
        config: Optional[Config] = None,
        rules: Optional[list] = None,
        user: Optional[str] = None,
        src_app: Optional[str] = None,
        on_violation: Optional[Callable[[InspectResponse, str], None]] = None,
        handle_tool_errors: bool = False,
        **tool_node_kwargs: Any,
    ) -> None:
        _validate_mode(mode)

        self._aidefense_guard = _Guard(
            api_key=api_key,
            region=region,
            mode=mode,
            fail_open=fail_open,
            timeout=timeout,
            config=config,
            rules=rules,
            user=user,
            src_app=src_app,
            on_violation=on_violation,
        )

        super().__init__(
            tools,
            wrap_tool_call=self._wrap_tool_call,
            awrap_tool_call=self._awrap_tool_call,
            handle_tool_errors=handle_tool_errors,
            **tool_node_kwargs,
        )

    def _wrap_tool_call(
        self,
        request: ToolCallRequest,
        execute: Callable,
    ) -> Any:
        if self._aidefense_guard.mode == "off":
            return execute(request)

        tool_name = request.tool_call["name"]
        arguments = request.tool_call.get("args", {})

        self._aidefense_guard.inspect_tool_request(tool_name, arguments)
        result = execute(request)
        if hasattr(result, "content"):
            self._aidefense_guard.inspect_tool_response(tool_name, result.content)

        return result

    async def _awrap_tool_call(
        self,
        request: ToolCallRequest,
        execute: Callable,
    ) -> Any:
        if self._aidefense_guard.mode == "off":
            return await execute(request)

        tool_name = request.tool_call["name"]
        arguments = request.tool_call.get("args", {})

        # SDK inspection calls are blocking HTTP — offload to a thread so
        # concurrent async tool calls do not serialize on the event loop.
        await asyncio.to_thread(
            self._aidefense_guard.inspect_tool_request, tool_name, arguments
        )
        result = await execute(request)
        if hasattr(result, "content"):
            await asyncio.to_thread(
                self._aidefense_guard.inspect_tool_response, tool_name, result.content
            )

        return result

    def close(self) -> None:
        """Release underlying HTTP session resources."""
        self._aidefense_guard.close()


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def create_aidefense_react_agent(
    model: Any,
    tools: Sequence[Union[BaseTool, Callable, dict]] | ToolNode,
    *,
    api_key: str,
    region: str = "us-west-2",
    mode: str = "enforce",
    fail_open: bool = True,
    timeout: int = 30,
    config: Optional[Config] = None,
    rules: Optional[list] = None,
    user: Optional[str] = None,
    src_app: Optional[str] = None,
    on_violation: Optional[Callable[[InspectResponse, str], None]] = None,
    **agent_kwargs: Any,
) -> CompiledStateGraph:
    """Drop-in replacement for ``create_react_agent`` with AI Defense.

    Wires ``AIDefenseHooks`` (LLM inspection) and ``AIDefenseToolNode``
    (tool inspection) into the agent automatically.  All keyword arguments
    beyond the AI Defense config are forwarded to ``create_react_agent``.

    Parameters
    ----------
    model :
        Same as ``create_react_agent``.
    tools :
        Same as ``create_react_agent``.  A pre-built ``ToolNode`` or
        ``AIDefenseToolNode`` is passed through unchanged.
    api_key : str
        Cisco AI Defense API key.
    region : str
        AI Defense region.  Default ``"us-west-2"``.
    mode : str
        ``"enforce"``, ``"monitor"``, or ``"off"``.  Default ``"enforce"``.
    fail_open : bool
        Allow on API errors.  Default ``True``.
    timeout : int
        Inspection timeout in seconds.  Default ``30``.
    config : Config, optional
        Pre-built SDK ``Config`` instance.
    rules : list, optional
        Rule names or ``Rule`` objects.
    user : str, optional
        User identity for audit.
    src_app : str, optional
        Source application name for audit.
    on_violation : callable, optional
        ``(InspectResponse, direction: str) -> None``.
    **agent_kwargs :
        Forwarded to ``create_react_agent`` (``prompt``, ``checkpointer``,
        ``response_format``, ``state_schema``, etc.).

    Returns
    -------
    CompiledStateGraph

    Example
    -------
    ::

        from langchain_openai import ChatOpenAI
        from aidefense_langchain import create_aidefense_react_agent, AIDefenseViolationError

        agent = create_aidefense_react_agent(
            model=ChatOpenAI(model="gpt-4o-mini"),
            tools=[get_weather, lookup_user],
            api_key="...",
            mode="enforce",
        )

        try:
            result = agent.invoke({"messages": [("user", "Hello")]})
        except AIDefenseViolationError as e:
            print(f"Blocked at {e.direction}: {e}")
    """
    _validate_mode(mode)

    # Shared Config so both hooks and tool node use the same singleton
    if config is None:
        config = Config(region=normalize_region(region), timeout=timeout)

    shared_kwargs = dict(
        api_key=api_key,
        mode=mode,
        fail_open=fail_open,
        timeout=timeout,
        config=config,
        rules=rules,
        user=user,
        src_app=src_app,
        on_violation=on_violation,
    )

    hooks = AIDefenseHooks(**shared_kwargs)

    # If the caller already passed a ToolNode, use it as-is but warn when it
    # is not an AIDefenseToolNode — tool call arguments and responses will not
    # be inspected in that case.
    if isinstance(tools, ToolNode):
        if not isinstance(tools, AIDefenseToolNode):
            warnings.warn(
                "create_aidefense_react_agent received a plain ToolNode — tool call "
                "arguments and responses will NOT be inspected by AI Defense. Pass an "
                "AIDefenseToolNode (or a plain tools list) for full coverage.",
                UserWarning,
                stacklevel=2,
            )
        final_tools: Any = tools
    else:
        tool_list = list(tools)
        # Provider built-in tool dicts (e.g. OpenAI/Anthropic native tools) are
        # not executable by ToolNode — bind them to the model directly so the
        # model knows about them without mixing them into the ToolNode list
        # (LangGraph would try to convert the ToolNode itself into a tool, failing).
        callable_tools = [t for t in tool_list if not isinstance(t, dict)]
        dict_tools = [t for t in tool_list if isinstance(t, dict)]

        if dict_tools:
            model = model.bind_tools(dict_tools)

        if callable_tools:
            final_tools = AIDefenseToolNode(callable_tools, **shared_kwargs)
        else:
            # No executable tools — nothing for ToolNode to do.
            final_tools = []

    return _create_react_agent(
        model,
        final_tools,
        pre_model_hook=hooks.pre_model_hook,
        post_model_hook=hooks.post_model_hook,
        **agent_kwargs,
    )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _get_messages(state: Any) -> List[BaseMessage]:
    if isinstance(state, dict):
        return state.get("messages") or []
    return getattr(state, "messages", []) or []
