"""Unit tests for AIDefenseAgentsecMiddleware (LLMInspector-based).

These tests mock the LLMInspector to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aidefense.runtime.agentsec.decision import Decision


class TestAIDefenseAgentsecMiddleware:
    """Tests for the LLMInspector-based middleware."""

    def _make_middleware(self, **kwargs):
        """Create middleware with a mocked LLMInspector."""
        with patch("aidefense_langchain.middleware_agentsec.LLMInspector"):
            from aidefense_langchain import AIDefenseAgentsecMiddleware
            mw = AIDefenseAgentsecMiddleware(**kwargs)
        return mw

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
        mw.inspector.inspect_conversation.assert_not_called()

    def test_enforce_allows_safe_content(self):
        mw = self._make_middleware(mode="enforce")
        mw.inspector.inspect_conversation.return_value = Decision.allow()

        result = mw.before_model(self._fake_state(), MagicMock())
        assert result is None

    def test_enforce_blocks_unsafe_content(self):
        mw = self._make_middleware(mode="enforce")
        mw.inspector.inspect_conversation.return_value = Decision.block(
            reasons=["PII detected"],
            severity="HIGH",
            event_id="evt-456",
        )

        result = mw.before_model(self._fake_state(), MagicMock())
        assert result is not None
        assert result["jump_to"] == "end"
        assert "blocked" in result["messages"][0].content.lower()

    def test_monitor_does_not_block(self):
        mw = self._make_middleware(mode="monitor")
        mw.inspector.inspect_conversation.return_value = Decision.block(
            reasons=["PII detected"],
        )

        result = mw.before_model(self._fake_state(), MagicMock())
        assert result is None

    def test_monitor_calls_on_violation(self):
        callback = MagicMock()
        mw = self._make_middleware(mode="monitor", on_violation=callback)
        mw.inspector.inspect_conversation.return_value = Decision.block(
            reasons=["PII detected"],
        )

        mw.before_model(self._fake_state(), MagicMock())
        callback.assert_called_once()
        decision_arg, direction_arg = callback.call_args[0]
        assert direction_arg == "input"
        assert decision_arg.action == "block"

    def test_monitor_only_decision_is_allowed(self):
        mw = self._make_middleware(mode="enforce")
        mw.inspector.inspect_conversation.return_value = Decision.monitor_only(
            reasons=["Low confidence detection"],
        )

        result = mw.before_model(self._fake_state(), MagicMock())
        assert result is None

    # -- after_model tests -------------------------------------------------

    def test_after_model_blocks_unsafe_response(self):
        mw = self._make_middleware(mode="enforce")
        mw.inspector.inspect_conversation.return_value = Decision.block(
            reasons=["Harmful content"],
        )

        ai_msg = MagicMock()
        ai_msg.type = "ai"
        ai_msg.content = "Some unsafe output"
        state = {"messages": [MagicMock(type="human", content="Hi"), ai_msg]}

        result = mw.after_model(state, MagicMock())
        assert result is not None
        assert result["jump_to"] == "end"

    # -- validation --------------------------------------------------------

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            self._make_middleware(mode="invalid")
