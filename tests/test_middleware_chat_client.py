"""Unit tests for AIDefenseMiddleware (ChatInspectionClient-based).

These tests mock the ChatInspectionClient to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aidefense.runtime.models import (
    Action,
    Classification,
    InspectResponse,
    Severity,
)


class TestAIDefenseMiddleware:
    """Tests for the ChatInspectionClient-based middleware."""

    def _make_middleware(self, **kwargs):
        """Create middleware with a mocked ChatInspectionClient."""
        with patch("aidefense_langchain.middleware_chat_client.ChatInspectionClient"):
            from aidefense_langchain import AIDefenseMiddleware
            mw = AIDefenseMiddleware(api_key="test-key", **kwargs)
        return mw

    @staticmethod
    def _safe_response() -> InspectResponse:
        return InspectResponse(
            classifications=[],
            is_safe=True,
            action=Action.ALLOW,
        )

    @staticmethod
    def _unsafe_response() -> InspectResponse:
        return InspectResponse(
            classifications=[Classification.PRIVACY_VIOLATION],
            is_safe=False,
            action=Action.BLOCK,
            severity=Severity.HIGH,
            event_id="evt-123",
        )

    @staticmethod
    def _fake_state(content: str = "Hello"):
        msg = MagicMock()
        msg.type = "human"
        msg.content = content
        return {"messages": [msg]}

    # -- mode tests --------------------------------------------------------

    def test_off_mode_skips_inspection(self):
        mw = self._make_middleware(mode="off")
        result = mw.before_model(self._fake_state(), MagicMock())
        assert result is None
        mw.client.inspect_conversation.assert_not_called()

    def test_enforce_allows_safe_content(self):
        mw = self._make_middleware(mode="enforce")
        mw.client.inspect_conversation.return_value = self._safe_response()

        result = mw.before_model(self._fake_state(), MagicMock())
        assert result is None

    def test_enforce_blocks_unsafe_content(self):
        mw = self._make_middleware(mode="enforce")
        mw.client.inspect_conversation.return_value = self._unsafe_response()

        result = mw.before_model(self._fake_state(), MagicMock())
        assert result is not None
        assert result["jump_to"] == "end"
        assert "blocked" in result["messages"][0].content.lower()

    def test_monitor_does_not_block(self):
        mw = self._make_middleware(mode="monitor")
        mw.client.inspect_conversation.return_value = self._unsafe_response()

        result = mw.before_model(self._fake_state(), MagicMock())
        assert result is None

    def test_monitor_calls_on_violation(self):
        callback = MagicMock()
        mw = self._make_middleware(mode="monitor", on_violation=callback)
        mw.client.inspect_conversation.return_value = self._unsafe_response()

        mw.before_model(self._fake_state(), MagicMock())
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[1] == "input"

    # -- fail_open tests ---------------------------------------------------

    def test_fail_open_allows_on_error(self):
        mw = self._make_middleware(mode="enforce", fail_open=True)
        mw.client.inspect_conversation.side_effect = ConnectionError("timeout")

        result = mw.before_model(self._fake_state(), MagicMock())
        assert result is None

    def test_fail_closed_raises_on_error(self):
        mw = self._make_middleware(mode="enforce", fail_open=False)
        mw.client.inspect_conversation.side_effect = ConnectionError("timeout")

        with pytest.raises(ConnectionError):
            mw.before_model(self._fake_state(), MagicMock())

    # -- after_model tests -------------------------------------------------

    def test_after_model_blocks_unsafe_response(self):
        mw = self._make_middleware(mode="enforce")
        mw.client.inspect_conversation.return_value = self._unsafe_response()

        ai_msg = MagicMock()
        ai_msg.type = "ai"
        ai_msg.content = "Here is some unsafe content"
        state = {"messages": [MagicMock(type="human", content="Hi"), ai_msg]}

        result = mw.after_model(state, MagicMock())
        assert result is not None
        assert result["jump_to"] == "end"

    @pytest.mark.asyncio
    async def test_abefore_model_uses_async_hook(self):
        mw = self._make_middleware(mode="enforce")
        mw.client.inspect_conversation.return_value = self._safe_response()

        result = await mw.abefore_model(self._fake_state(), MagicMock())

        assert result is None
        mw.client.inspect_conversation.assert_called_once()

    def test_from_env_normalizes_short_region_names(self):
        with patch("aidefense_langchain.middleware_chat_client.ChatInspectionClient"):
            from aidefense_langchain import AIDefenseMiddleware

            mw = AIDefenseMiddleware.from_env(
                {
                    "AIDEFENSE_API_KEY": "test-key",
                    "AIDEFENSE_REGION": "us",
                    "AIDEFENSE_MODE": "monitor",
                }
            )

        assert mw.mode == "monitor"

    # -- validation --------------------------------------------------------

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            self._make_middleware(mode="invalid")
