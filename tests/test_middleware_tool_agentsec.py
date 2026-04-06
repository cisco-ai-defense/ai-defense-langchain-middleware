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

"""Tests for AIDefenseAgentsecToolMiddleware (MCPInspector-based)."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain.messages import ToolMessage

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
    req.tool_call = {"id": f"{name}-id", "name": name, "args": args or {}, "metadata": {}}
    return req


def _make_tool_message(content="tool result"):
    return ToolMessage(content=content, tool_call_id="tool-call-1")


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

    @pytest.mark.asyncio
    async def test_awrap_tool_call_uses_async_inspector(self, mock_inspector):
        mock_inspector.ainspect_request = AsyncMock(return_value=_make_allow_decision())
        mock_inspector.ainspect_response = AsyncMock(return_value=_make_allow_decision())

        mw = AIDefenseAgentsecToolMiddleware(mode="enforce")
        handler = AsyncMock(return_value=_make_tool_message())
        req = _make_tool_request()

        result = await mw.awrap_tool_call(req, handler)

        assert result.content == "tool result"
        mock_inspector.ainspect_request.assert_awaited_once()

    def test_from_env_reads_agentsec_tool_settings(self, mock_inspector):
        with patch(
            "aidefense_langchain.middleware_tool_agentsec.MCPInspector"
        ) as MockInspector:
            AIDefenseAgentsecToolMiddleware.from_env(
                {
                    "AIDEFENSE_API_KEY": "test-key",
                    "AIDEFENSE_ENDPOINT": "https://example.com",
                    "AIDEFENSE_RETRY_BACKOFF": "1.5",
                }
            )

        kwargs = MockInspector.call_args.kwargs
        assert kwargs["api_key"] == "test-key"
        assert kwargs["endpoint"] == "https://example.com"
        assert kwargs["retry_backoff"] == 1.5

    def test_from_env_ignores_user_and_src_app(self, mock_inspector):
        """AIDEFENSE_USER / AIDEFENSE_SRC_APP must not leak into the
        tool middleware constructor (it doesn't accept them)."""
        with patch(
            "aidefense_langchain.middleware_tool_agentsec.MCPInspector"
        ):
            AIDefenseAgentsecToolMiddleware.from_env(
                {
                    "AIDEFENSE_API_KEY": "test-key",
                    "AIDEFENSE_ENDPOINT": "https://example.com",
                    "AIDEFENSE_USER": "alice",
                    "AIDEFENSE_SRC_APP": "myapp",
                }
            )

    def test_structured_tool_result_content_is_flattened(self, mock_inspector):
        mw = AIDefenseAgentsecToolMiddleware(mode="enforce")
        result_data = mw._extract_result_data(
            ToolMessage(
                content=[
                    {"type": "text", "text": "alice@example.com"},
                    {"type": "text", "text": "bob@example.com"},
                ],
                tool_call_id="tool-call-1",
            )
        )

        assert result_data == {
            "content": [
                {
                    "type": "text",
                    "text": "alice@example.com\nbob@example.com",
                }
            ]
        }
