# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AIDefenseHooks, AIDefenseToolNode, and create_aidefense_react_agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from aidefense.runtime.models import (
    Action,
    Classification,
    InspectResponse,
    Severity,
)
from aidefense.runtime.mcp_models import MCPInspectResponse

from aidefense_langchain import (
    AIDefenseHooks,
    AIDefenseToolNode,
    AIDefenseViolationError,
    create_aidefense_react_agent,
)

PATCH_CHAT = "aidefense_langchain.react_agent.ChatInspectionClient"
PATCH_MCP = "aidefense_langchain.react_agent.MCPInspectionClient"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hooks(**kwargs) -> AIDefenseHooks:
    with patch(PATCH_CHAT), patch(PATCH_MCP):
        return AIDefenseHooks(api_key="test-key", **kwargs)


def _make_tool_node(tools=None, **kwargs) -> AIDefenseToolNode:
    with patch(PATCH_CHAT), patch(PATCH_MCP):
        return AIDefenseToolNode(tools or [], api_key="test-key", **kwargs)


def _safe() -> InspectResponse:
    return InspectResponse(classifications=[], is_safe=True, action=Action.ALLOW)


def _unsafe() -> InspectResponse:
    return InspectResponse(
        classifications=[Classification.SECURITY_VIOLATION],
        is_safe=False,
        action=Action.BLOCK,
        severity=Severity.HIGH,
        event_id="evt-1",
        explanation="prompt injection",
    )


def _mcp_safe() -> MCPInspectResponse:
    r = MCPInspectResponse()
    r.result = _safe()
    return r


def _mcp_unsafe() -> MCPInspectResponse:
    r = MCPInspectResponse()
    r.result = _unsafe()
    return r


def _state(text="Hello") -> dict:
    return {"messages": [HumanMessage(content=text)]}


def _state_with_ai_response(text="Here is the answer") -> dict:
    return {"messages": [HumanMessage(content="q"), AIMessage(content=text)]}


def _tool_request(name="search", args=None) -> ToolCallRequest:
    req = MagicMock(spec=ToolCallRequest)
    req.tool_call = {"name": name, "args": args or {"query": "test"}, "id": "tc-1"}
    return req


# ---------------------------------------------------------------------------
# AIDefenseViolationError
# ---------------------------------------------------------------------------

class TestAIDefenseViolationError:
    def test_attributes(self):
        err = AIDefenseViolationError("input", _unsafe())
        assert err.direction == "input"
        assert err.response.action == Action.BLOCK
        assert "input" in str(err)
        assert "prompt injection" in str(err)


# ---------------------------------------------------------------------------
# AIDefenseHooks — constructor
# ---------------------------------------------------------------------------

class TestAIDefenseHooksConstructor:
    def test_invalid_mode_raises(self):
        with patch(PATCH_CHAT), patch(PATCH_MCP):
            with pytest.raises(ValueError, match="mode must be"):
                AIDefenseHooks(api_key="key", mode="block")

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("AIDEFENSE_API_KEY", "env-key")
        monkeypatch.setenv("AIDEFENSE_MODE", "monitor")
        with patch(PATCH_CHAT), patch(PATCH_MCP):
            h = AIDefenseHooks.from_env()
        assert h._guard.mode == "monitor"


# ---------------------------------------------------------------------------
# AIDefenseHooks — pre_model_hook
# ---------------------------------------------------------------------------

class TestPreModelHook:
    def test_off_mode_skips_inspection(self):
        h = _make_hooks(mode="off")
        result = h.pre_model_hook(_state())
        h._guard.chat_client.inspect_conversation.assert_not_called()
        assert result == {}

    def test_safe_input_returns_empty_dict(self):
        h = _make_hooks(mode="enforce")
        h._guard.chat_client.inspect_conversation.return_value = _safe()
        assert h.pre_model_hook(_state()) == {}

    def test_unsafe_input_raises_in_enforce(self):
        h = _make_hooks(mode="enforce")
        h._guard.chat_client.inspect_conversation.return_value = _unsafe()
        with pytest.raises(AIDefenseViolationError) as exc:
            h.pre_model_hook(_state("inject me"))
        assert exc.value.direction == "input"

    def test_unsafe_input_does_not_raise_in_monitor(self):
        h = _make_hooks(mode="monitor")
        h._guard.chat_client.inspect_conversation.return_value = _unsafe()
        result = h.pre_model_hook(_state())
        assert result == {}

    def test_on_violation_called_in_monitor(self):
        cb = MagicMock()
        h = _make_hooks(mode="monitor", on_violation=cb)
        h._guard.chat_client.inspect_conversation.return_value = _unsafe()
        h.pre_model_hook(_state())
        cb.assert_called_once()
        assert cb.call_args[0][1] == "input"

    def test_on_violation_called_in_enforce(self):
        cb = MagicMock()
        h = _make_hooks(mode="enforce", on_violation=cb)
        h._guard.chat_client.inspect_conversation.return_value = _unsafe()
        with pytest.raises(AIDefenseViolationError):
            h.pre_model_hook(_state())
        cb.assert_called_once()

    def test_empty_messages_skips_inspection(self):
        h = _make_hooks(mode="enforce")
        h.pre_model_hook({"messages": []})
        h._guard.chat_client.inspect_conversation.assert_not_called()

    def test_system_message_role_mapped(self):
        h = _make_hooks(mode="enforce")
        h._guard.chat_client.inspect_conversation.return_value = _safe()
        state = {"messages": [SystemMessage(content="system"), HumanMessage(content="hi")]}
        h.pre_model_hook(state)
        call_messages = h._guard.chat_client.inspect_conversation.call_args[1]["messages"]
        from aidefense.runtime import Role
        assert call_messages[0].role == Role.SYSTEM
        assert call_messages[1].role == Role.USER

    def test_fail_open_allows_on_api_error(self):
        h = _make_hooks(mode="enforce", fail_open=True)
        h._guard.chat_client.inspect_conversation.side_effect = ConnectionError("timeout")
        assert h.pre_model_hook(_state()) == {}

    def test_fail_closed_raises_on_api_error(self):
        h = _make_hooks(mode="enforce", fail_open=False)
        h._guard.chat_client.inspect_conversation.side_effect = ConnectionError("timeout")
        with pytest.raises(ConnectionError):
            h.pre_model_hook(_state())


# ---------------------------------------------------------------------------
# AIDefenseHooks — post_model_hook
# ---------------------------------------------------------------------------

class TestPostModelHook:
    def test_off_mode_skips_inspection(self):
        h = _make_hooks(mode="off")
        h.post_model_hook(_state_with_ai_response())
        h._guard.chat_client.inspect_response.assert_not_called()

    def test_safe_output_returns_empty_dict(self):
        h = _make_hooks(mode="enforce")
        h._guard.chat_client.inspect_response.return_value = _safe()
        assert h.post_model_hook(_state_with_ai_response()) == {}

    def test_unsafe_output_raises_in_enforce(self):
        h = _make_hooks(mode="enforce")
        h._guard.chat_client.inspect_response.return_value = _unsafe()
        with pytest.raises(AIDefenseViolationError) as exc:
            h.post_model_hook(_state_with_ai_response("my SSN is 123-45-6789"))
        assert exc.value.direction == "output"

    def test_unsafe_output_does_not_raise_in_monitor(self):
        h = _make_hooks(mode="monitor")
        h._guard.chat_client.inspect_response.return_value = _unsafe()
        result = h.post_model_hook(_state_with_ai_response())
        assert result == {}

    def test_empty_message_list_skips_inspection(self):
        h = _make_hooks(mode="enforce")
        h.post_model_hook({"messages": []})
        h._guard.chat_client.inspect_response.assert_not_called()

    def test_fail_open_allows_on_api_error(self):
        h = _make_hooks(mode="enforce", fail_open=True)
        h._guard.chat_client.inspect_response.side_effect = ConnectionError()
        assert h.post_model_hook(_state_with_ai_response()) == {}

    def test_fail_closed_raises_on_api_error(self):
        h = _make_hooks(mode="enforce", fail_open=False)
        h._guard.chat_client.inspect_response.side_effect = ConnectionError()
        with pytest.raises(ConnectionError):
            h.post_model_hook(_state_with_ai_response())


# ---------------------------------------------------------------------------
# AIDefenseToolNode — tool request inspection
# ---------------------------------------------------------------------------

class TestAIDefenseToolNode:
    def test_off_mode_skips_inspection(self):
        node = _make_tool_node(mode="off")
        execute = MagicMock(return_value=MagicMock(content="result"))
        node._wrap_tool_call(_tool_request(), execute)
        node._aidefense_guard.mcp_client.inspect_tool_call.assert_not_called()
        node._aidefense_guard.mcp_client.inspect_response.assert_not_called()

    def test_safe_request_calls_execute(self):
        node = _make_tool_node(mode="enforce")
        node._aidefense_guard.mcp_client.inspect_tool_call.return_value = _mcp_safe()
        node._aidefense_guard.mcp_client.inspect_response.return_value = _mcp_safe()
        execute = MagicMock(return_value=MagicMock(content="result"))
        node._wrap_tool_call(_tool_request(), execute)
        execute.assert_called_once()

    def test_unsafe_request_raises_in_enforce(self):
        node = _make_tool_node(mode="enforce")
        node._aidefense_guard.mcp_client.inspect_tool_call.return_value = _mcp_unsafe()
        execute = MagicMock()
        with pytest.raises(AIDefenseViolationError) as exc:
            node._wrap_tool_call(_tool_request("dangerous_tool"), execute)
        assert "dangerous_tool" in exc.value.direction
        assert "input" in exc.value.direction
        execute.assert_not_called()

    def test_unsafe_response_raises_in_enforce(self):
        node = _make_tool_node(mode="enforce")
        node._aidefense_guard.mcp_client.inspect_tool_call.return_value = _mcp_safe()
        node._aidefense_guard.mcp_client.inspect_response.return_value = _mcp_unsafe()
        execute = MagicMock(return_value=MagicMock(content="SSN 123-45-6789"))
        with pytest.raises(AIDefenseViolationError) as exc:
            node._wrap_tool_call(_tool_request("lookup"), execute)
        assert "lookup" in exc.value.direction
        assert "output" in exc.value.direction

    def test_monitor_mode_does_not_raise(self):
        node = _make_tool_node(mode="monitor")
        node._aidefense_guard.mcp_client.inspect_tool_call.return_value = _mcp_unsafe()
        node._aidefense_guard.mcp_client.inspect_response.return_value = _mcp_unsafe()
        execute = MagicMock(return_value=MagicMock(content="result"))
        node._wrap_tool_call(_tool_request(), execute)  # must not raise

    def test_on_violation_called_for_request(self):
        cb = MagicMock()
        node = _make_tool_node(mode="monitor", on_violation=cb)
        node._aidefense_guard.mcp_client.inspect_tool_call.return_value = _mcp_unsafe()
        node._aidefense_guard.mcp_client.inspect_response.return_value = _mcp_safe()
        execute = MagicMock(return_value=MagicMock(content="ok"))
        node._wrap_tool_call(_tool_request("mytool"), execute)
        cb.assert_called_once()
        assert "mytool" in cb.call_args[0][1]
        assert "input" in cb.call_args[0][1]

    def test_on_violation_called_for_response(self):
        cb = MagicMock()
        node = _make_tool_node(mode="monitor", on_violation=cb)
        node._aidefense_guard.mcp_client.inspect_tool_call.return_value = _mcp_safe()
        node._aidefense_guard.mcp_client.inspect_response.return_value = _mcp_unsafe()
        execute = MagicMock(return_value=MagicMock(content="pii data"))
        node._wrap_tool_call(_tool_request("mytool"), execute)
        cb.assert_called_once()
        assert "mytool" in cb.call_args[0][1]
        assert "output" in cb.call_args[0][1]

    def test_fail_open_allows_on_api_error(self):
        node = _make_tool_node(mode="enforce", fail_open=True)
        node._aidefense_guard.mcp_client.inspect_tool_call.side_effect = ConnectionError()
        execute = MagicMock(return_value=MagicMock(content="ok"))
        node._wrap_tool_call(_tool_request(), execute)  # must not raise

    def test_fail_closed_raises_on_api_error(self):
        node = _make_tool_node(mode="enforce", fail_open=False)
        node._aidefense_guard.mcp_client.inspect_tool_call.side_effect = ConnectionError()
        execute = MagicMock()
        with pytest.raises(ConnectionError):
            node._wrap_tool_call(_tool_request(), execute)

    def test_invalid_mode_raises(self):
        with patch(PATCH_CHAT), patch(PATCH_MCP):
            with pytest.raises(ValueError, match="mode must be"):
                AIDefenseToolNode([], api_key="key", mode="wrong")


# ---------------------------------------------------------------------------
# create_aidefense_react_agent
# ---------------------------------------------------------------------------

class TestCreateAIDefenseReactAgent:
    def test_invalid_mode_raises(self):
        with patch(PATCH_CHAT), patch(PATCH_MCP):
            with pytest.raises(ValueError, match="mode must be"):
                create_aidefense_react_agent(MagicMock(), [], api_key="k", mode="bad")

    def test_returns_compiled_graph(self):
        with patch(PATCH_CHAT), patch(PATCH_MCP), \
             patch("aidefense_langchain.react_agent._create_react_agent") as mock_create:
            mock_create.return_value = MagicMock()
            result = create_aidefense_react_agent(
                MagicMock(), [], api_key="key", mode="enforce"
            )
            mock_create.assert_called_once()
            assert result is mock_create.return_value

    def test_hooks_and_tool_node_passed_to_create_react_agent(self):
        with patch(PATCH_CHAT), patch(PATCH_MCP), \
             patch("aidefense_langchain.react_agent._create_react_agent") as mock_create:
            mock_create.return_value = MagicMock()
            create_aidefense_react_agent(
                MagicMock(), [], api_key="key", mode="enforce"
            )
            _, kwargs = mock_create.call_args
            assert "pre_model_hook" in kwargs
            assert "post_model_hook" in kwargs
            assert callable(kwargs["pre_model_hook"])
            assert callable(kwargs["post_model_hook"])

    def test_prebuilt_tool_node_passed_through_unchanged(self):
        from langgraph.prebuilt import ToolNode
        existing_node = MagicMock(spec=ToolNode)
        with patch(PATCH_CHAT), patch(PATCH_MCP), \
             patch("aidefense_langchain.react_agent._create_react_agent") as mock_create:
            mock_create.return_value = MagicMock()
            create_aidefense_react_agent(
                MagicMock(), existing_node, api_key="key"
            )
            args, _ = mock_create.call_args
            assert args[1] is existing_node

    def test_extra_kwargs_forwarded_to_create_react_agent(self):
        with patch(PATCH_CHAT), patch(PATCH_MCP), \
             patch("aidefense_langchain.react_agent._create_react_agent") as mock_create:
            mock_create.return_value = MagicMock()
            create_aidefense_react_agent(
                MagicMock(), [], api_key="key", prompt="Be helpful"
            )
            _, kwargs = mock_create.call_args
            assert kwargs.get("prompt") == "Be helpful"
