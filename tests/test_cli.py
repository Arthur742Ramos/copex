"""Tests for CLI components."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from copex.cli import SlashCompleter


class TestSlashCompleter:
    """Tests for slash command completer."""

    def test_no_completions_for_regular_text(self):
        """Should not show completions for regular text."""
        completer = SlashCompleter(["/model", "/help", "/new"])
        document = MagicMock()
        document.text_before_cursor = "hello world"

        completions = list(completer.get_completions(document, None))

        assert completions == []

    def test_no_completions_for_empty_input(self):
        """Should not show completions for empty input."""
        completer = SlashCompleter(["/model", "/help", "/new"])
        document = MagicMock()
        document.text_before_cursor = ""

        completions = list(completer.get_completions(document, None))

        assert completions == []

    def test_shows_all_commands_on_slash(self):
        """Should show all commands when user types just /."""
        completer = SlashCompleter(["/model", "/help", "/new"])
        document = MagicMock()
        document.text_before_cursor = "/"

        completions = list(completer.get_completions(document, None))

        assert len(completions) == 3
        texts = [c.text for c in completions]
        assert "/model" in texts
        assert "/help" in texts
        assert "/new" in texts

    def test_filters_commands_on_partial_input(self):
        """Should filter commands based on partial input."""
        completer = SlashCompleter(["/model", "/models", "/help", "/new"])
        document = MagicMock()
        document.text_before_cursor = "/mo"

        completions = list(completer.get_completions(document, None))

        assert len(completions) == 2
        texts = [c.text for c in completions]
        assert "/model" in texts
        assert "/models" in texts
        assert "/help" not in texts

    def test_case_insensitive_matching(self):
        """Should match commands case-insensitively."""
        completer = SlashCompleter(["/Model", "/HELP"])
        document = MagicMock()
        document.text_before_cursor = "/mo"

        completions = list(completer.get_completions(document, None))

        assert len(completions) == 1
        assert completions[0].text == "/Model"

    def test_handles_leading_whitespace(self):
        """Should handle leading whitespace before slash."""
        completer = SlashCompleter(["/model", "/help"])
        document = MagicMock()
        document.text_before_cursor = "  /mo"

        completions = list(completer.get_completions(document, None))

        assert len(completions) == 1
        assert completions[0].text == "/model"

    def test_start_position_correct(self):
        """Completion start position should replace typed text."""
        completer = SlashCompleter(["/model"])
        document = MagicMock()
        document.text_before_cursor = "/mo"

        completions = list(completer.get_completions(document, None))

        assert len(completions) == 1
        assert completions[0].start_position == -3  # Replace "/mo"

    def test_no_match_returns_empty(self):
        """Should return empty when no commands match."""
        completer = SlashCompleter(["/model", "/help"])
        document = MagicMock()
        document.text_before_cursor = "/xyz"

        completions = list(completer.get_completions(document, None))

        assert completions == []
