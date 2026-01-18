"""Tests for model utilities."""

from __future__ import annotations

import pytest

from copex.models import Model, ReasoningEffort, supports_reasoning


class TestSupportsReasoning:
    """Tests for supports_reasoning function."""

    def test_gpt_models_support_reasoning(self):
        """All GPT models should support reasoning."""
        gpt_models = [
            Model.GPT_5_2_CODEX,
            Model.GPT_5_1_CODEX,
            Model.GPT_5_1_CODEX_MAX,
            Model.GPT_5_1_CODEX_MINI,
            Model.GPT_5_2,
            Model.GPT_5_1,
            Model.GPT_5,
            Model.GPT_5_MINI,
            Model.GPT_4_1,
        ]
        for model in gpt_models:
            assert supports_reasoning(model) is True, f"{model} should support reasoning"

    def test_claude_models_do_not_support_reasoning(self):
        """Claude models should not support reasoning."""
        claude_models = [
            Model.CLAUDE_SONNET_4_5,
            Model.CLAUDE_SONNET_4,
            Model.CLAUDE_HAIKU_4_5,
            Model.CLAUDE_OPUS_4_5,
        ]
        for model in claude_models:
            assert supports_reasoning(model) is False, f"{model} should not support reasoning"

    def test_gemini_models_do_not_support_reasoning(self):
        """Gemini models should not support reasoning."""
        assert supports_reasoning(Model.GEMINI_3_PRO) is False

    def test_supports_reasoning_with_string(self):
        """Should work with string model names."""
        assert supports_reasoning("gpt-5.2-codex") is True
        assert supports_reasoning("gpt-4.1") is True
        assert supports_reasoning("claude-opus-4.5") is False
        assert supports_reasoning("gemini-3-pro-preview") is False

    def test_model_method_supports_reasoning(self):
        """Model enum should have supports_reasoning method."""
        assert Model.GPT_5_2_CODEX.supports_reasoning() is True
        assert Model.CLAUDE_OPUS_4_5.supports_reasoning() is False


class TestReasoningEffort:
    """Tests for ReasoningEffort enum."""

    def test_all_levels_exist(self):
        """All reasoning levels should be defined."""
        levels = [r.value for r in ReasoningEffort]
        assert "none" in levels
        assert "low" in levels
        assert "medium" in levels
        assert "high" in levels
        assert "xhigh" in levels

    def test_reasoning_effort_from_string(self):
        """Should create from string value."""
        assert ReasoningEffort("xhigh") == ReasoningEffort.XHIGH
        assert ReasoningEffort("none") == ReasoningEffort.NONE
