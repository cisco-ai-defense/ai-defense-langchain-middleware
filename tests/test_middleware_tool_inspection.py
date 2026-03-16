"""Tests for AIDefenseToolMiddleware (MCPInspectionClient-based)."""

from unittest.mock import MagicMock, patch
import pytest

from aidefense_langchain.middleware_tool_inspection import AIDefenseToolMiddleware


@pytest.fixture
def mock_mcp_client():
    with patch(
        "aidefense_langchain.middleware_tool_inspection.MCPInspectionClient"
    ) as MockClient:
        client = MagicMock()
        MockClient.return_value = client
        yield client


def _make_safe_response():
    resp = MagicMock()
    resp.result = MagicMock()
    resp.result.is_safe = True
    resp.error = None
    return resp


def _make_unsafe_response(classifications=None):
    resp = MagicMock()
    resp.result = MagicMock()
    resp.result.is_safe = False
    resp.result.classifications = classifications or []
    resp.result.severity = None
    resp.result.event_id = None
    resp.error = None
    return resp


def _make_tool_request(name="my_tool", args=None):
    req = MagicMock()
    req.tool_call = {"name": name, "args": args or {}}
    return req


def _make_tool_message(content="tool result"):
    msg = MagicMock()
    msg.content = content
    type(msg).__name__ = "ToolMessage"
    return msg


class TestAIDefenseToolMiddleware:
    def test_off_mode_skips_inspection(self, mock_mcp_client):
        mw = AIDefenseToolMiddleware(api_key="test", mode="off")
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request()

        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once_with(req)
        mock_mcp_client.inspect_tool_call.assert_not_called()

    def test_safe_request_passes_through(self, mock_mcp_client):
        mock_mcp_client.inspect_tool_call.return_value = _make_safe_response()
        mock_mcp_client.inspect_response.return_value = _make_safe_response()

        mw = AIDefenseToolMiddleware(api_key="test", mode="enforce")
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request()

        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once()
        mock_mcp_client.inspect_tool_call.assert_called_once()

    def test_unsafe_request_blocks_in_enforce(self, mock_mcp_client):
        mock_mcp_client.inspect_tool_call.return_value = _make_unsafe_response()

        mw = AIDefenseToolMiddleware(api_key="test", mode="enforce")
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request(name="dangerous_tool")

        result = mw.wrap_tool_call(req, handler)

        handler.assert_not_called()
        assert "blocked" in result.content.lower()
        assert "dangerous_tool" in result.content

    def test_unsafe_request_allows_in_monitor(self, mock_mcp_client):
        mock_mcp_client.inspect_tool_call.return_value = _make_unsafe_response()
        mock_mcp_client.inspect_response.return_value = _make_safe_response()

        mw = AIDefenseToolMiddleware(api_key="test", mode="monitor")
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request()

        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once()

    def test_unsafe_response_blocks_in_enforce(self, mock_mcp_client):
        mock_mcp_client.inspect_tool_call.return_value = _make_safe_response()
        mock_mcp_client.inspect_response.return_value = _make_unsafe_response()

        mw = AIDefenseToolMiddleware(api_key="test", mode="enforce")
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request()

        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once()
        assert "blocked" in result.content.lower()

    def test_fail_open_on_api_error(self, mock_mcp_client):
        mock_mcp_client.inspect_tool_call.side_effect = ConnectionError("timeout")
        mock_mcp_client.inspect_response.return_value = _make_safe_response()

        mw = AIDefenseToolMiddleware(api_key="test", mode="enforce", fail_open=True)
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request()

        result = mw.wrap_tool_call(req, handler)
        handler.assert_called_once()

    def test_fail_closed_on_api_error(self, mock_mcp_client):
        mock_mcp_client.inspect_tool_call.side_effect = ConnectionError("timeout")

        mw = AIDefenseToolMiddleware(api_key="test", mode="enforce", fail_open=False)
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request()

        with pytest.raises(ConnectionError):
            mw.wrap_tool_call(req, handler)

    def test_on_violation_callback(self, mock_mcp_client):
        mock_mcp_client.inspect_tool_call.return_value = _make_unsafe_response()

        callback = MagicMock()
        mw = AIDefenseToolMiddleware(
            api_key="test", mode="enforce", on_violation=callback,
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
            AIDefenseToolMiddleware(api_key="test", mode="invalid")

    def test_request_only_inspection(self, mock_mcp_client):
        mock_mcp_client.inspect_tool_call.return_value = _make_safe_response()

        mw = AIDefenseToolMiddleware(
            api_key="test", mode="enforce",
            inspect_requests=True, inspect_responses=False,
        )
        handler = MagicMock(return_value=_make_tool_message())
        req = _make_tool_request()

        mw.wrap_tool_call(req, handler)

        mock_mcp_client.inspect_tool_call.assert_called_once()
        mock_mcp_client.inspect_response.assert_not_called()
