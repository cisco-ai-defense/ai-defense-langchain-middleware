"""Tests for AIDefenseAgentsecToolMiddleware (MCPInspector-based)."""

from unittest.mock import MagicMock, patch
import pytest

from aidefense_langchain.middleware_tool_agentsec import (
    AIDefenseAgentsecToolMiddleware,
)


@pytest.fixture
def mock_inspector():
    with patch(
        "aidefense_langchain.middleware_tool_agentsec.MCPInspector"
    ) as MockInspector:
        inspector = MagicMock()
        MockInspector.return_value = inspector
        yield inspector


def _make_allow_decision():
    d = MagicMock()
    d.action = "allow"
    return d


def _make_block_decision():
    d = MagicMock()
    d.action = "block"
    return d


def _make_monitor_decision():
    d = MagicMock()
    d.action = "monitor_only"
    return d


def _make_tool_request(name="my_tool", args=None):
    req = MagicMock()
    req.tool_call = {"name": name, "args": args or {}, "metadata": {}}
    return req


def _make_tool_message(content="tool result"):
    msg = MagicMock()
    msg.content = content
    type(msg).__name__ = "ToolMessage"
    return msg


class TestAIDefenseAgentsecToolMiddleware:
    def test_off_mode_skips_inspection(self, mock_inspector):
        mw = AIDefenseAgentsecToolMiddleware(mode="off")
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request()

        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once_with(req)
        mock_inspector.inspect_request.assert_not_called()

    def test_allow_decision_passes_through(self, mock_inspector):
        mock_inspector.inspect_request.return_value = _make_allow_decision()
        mock_inspector.inspect_response.return_value = _make_allow_decision()

        mw = AIDefenseAgentsecToolMiddleware(mode="enforce")
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request()

        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once()
        mock_inspector.inspect_request.assert_called_once()

    def test_block_decision_blocks_in_enforce(self, mock_inspector):
        mock_inspector.inspect_request.return_value = _make_block_decision()

        mw = AIDefenseAgentsecToolMiddleware(mode="enforce")
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request(name="risky_tool")

        result = mw.wrap_tool_call(req, handler)

        handler.assert_not_called()
        assert "blocked" in result.content.lower()
        assert "risky_tool" in result.content

    def test_block_decision_allows_in_monitor(self, mock_inspector):
        mock_inspector.inspect_request.return_value = _make_block_decision()
        mock_inspector.inspect_response.return_value = _make_allow_decision()

        mw = AIDefenseAgentsecToolMiddleware(mode="monitor")
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request()

        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once()

    def test_response_block_in_enforce(self, mock_inspector):
        mock_inspector.inspect_request.return_value = _make_allow_decision()
        mock_inspector.inspect_response.return_value = _make_block_decision()

        mw = AIDefenseAgentsecToolMiddleware(mode="enforce")
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request()

        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once()
        assert "blocked" in result.content.lower()

    def test_monitor_only_decision_passes(self, mock_inspector):
        mock_inspector.inspect_request.return_value = _make_monitor_decision()
        mock_inspector.inspect_response.return_value = _make_allow_decision()

        mw = AIDefenseAgentsecToolMiddleware(mode="enforce")
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request()

        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once()

    def test_on_violation_callback(self, mock_inspector):
        mock_inspector.inspect_request.return_value = _make_block_decision()

        callback = MagicMock()
        mw = AIDefenseAgentsecToolMiddleware(
            mode="enforce", on_violation=callback,
        )
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request(name="test_tool")

        mw.wrap_tool_call(req, handler)

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[1] == "test_tool"
        assert args[2] == "request"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            AIDefenseAgentsecToolMiddleware(mode="invalid")

    def test_close_delegates(self, mock_inspector):
        mw = AIDefenseAgentsecToolMiddleware(mode="enforce")
        mw.close()
        mock_inspector.close.assert_called_once()
