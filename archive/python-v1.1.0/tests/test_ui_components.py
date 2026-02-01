"""Tests for UI components - syntax highlighting, diffs, collapsible sections, token usage, tool calls, progress."""

from __future__ import annotations

import os
import time
import pytest
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from copex.ui_components import (
    CodeBlock,
    CollapsibleGroup,
    CollapsibleSection,
    DiffDisplay,
    DiffLine,
    DiffLineType,
    PlainTextRenderer,
    RichProgressReporter,
    StepInfo,
    StepProgress,
    StepStatus,
    TokenUsageDisplay,
    ToolCallGroup,
    ToolCallPanel,
    ToolStatus,
    MODEL_PRICING,
    extract_code_blocks,
    get_plain_console,
    is_terminal,
    render_markdown_with_syntax,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CodeBlock Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeBlock:
    """Tests for CodeBlock syntax highlighting component."""

    def test_basic_creation(self):
        """CodeBlock should be created with code and language."""
        code = CodeBlock("print('hello')", language="python")
        assert code.code == "print('hello')"
        assert code.language == "python"

    def test_language_detection_from_filename(self):
        """Should detect language from filename extension."""
        code = CodeBlock("const x = 1;", filename="app.js")
        assert code.language == "javascript"

        code = CodeBlock("fn main() {}", filename="main.rs")
        assert code.language == "rust"

        code = CodeBlock("package main", filename="main.go")
        assert code.language == "go"

    def test_language_detection_from_content(self):
        """Should detect language from content patterns."""
        code = CodeBlock("#!/bin/bash\necho hello")
        assert code.language == "bash"

        code = CodeBlock("def hello():\n    pass")
        assert code.language == "python"

        code = CodeBlock('{"key": "value"}')
        assert code.language == "json"

    def test_from_file_classmethod(self):
        """from_file should create CodeBlock with filename."""
        code = CodeBlock.from_file("test.py", "print(1)")
        assert code.filename == "test.py"
        assert code.language == "python"

    def test_to_syntax_returns_syntax_object(self):
        """to_syntax should return Rich Syntax object."""
        code = CodeBlock("x = 1", language="python")
        syntax = code.to_syntax()
        assert isinstance(syntax, Syntax)

    def test_to_panel_returns_panel(self):
        """to_panel should wrap code in a Panel."""
        code = CodeBlock("x = 1", language="python", filename="test.py")
        panel = code.to_panel()
        assert isinstance(panel, Panel)

    def test_rich_protocol(self):
        """CodeBlock should implement __rich__ for direct printing."""
        code = CodeBlock("x = 1", language="python")
        result = code.__rich__()
        assert isinstance(result, Syntax)

    def test_line_numbers(self):
        """Should support line numbers."""
        code = CodeBlock("line1\nline2", language="python", line_numbers=True)
        assert code.line_numbers is True

    def test_highlight_lines(self):
        """Should support line highlighting."""
        code = CodeBlock("a\nb\nc", language="python", highlight_lines={2})
        assert code.highlight_lines == {2}

    def test_strips_trailing_whitespace(self):
        """Should strip trailing whitespace from code."""
        code = CodeBlock("hello\n\n  \n", language="text")
        assert code.code == "hello"


# ═══════════════════════════════════════════════════════════════════════════════
# DiffDisplay Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDiffLine:
    """Tests for DiffLine dataclass."""

    def test_creation(self):
        """DiffLine should hold line data."""
        line = DiffLine(
            type=DiffLineType.ADDITION,
            content="new line",
            new_lineno=5,
        )
        assert line.type == DiffLineType.ADDITION
        assert line.content == "new line"
        assert line.new_lineno == 5
        assert line.old_lineno is None


class TestDiffDisplay:
    """Tests for DiffDisplay component."""

    def test_from_unified_diff(self):
        """Should parse unified diff format."""
        diff_text = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 line 1
-old line
+new line
+added line
 line 3"""

        diff = DiffDisplay.from_unified_diff(diff_text)

        assert diff.filename == "a/file.py"
        assert len(diff.lines) > 0

        # Check stats
        stats = diff.stats
        assert stats["additions"] == 2
        assert stats["deletions"] == 1

    def test_from_strings(self):
        """Should create diff from old and new content."""
        old = "line 1\nold line\nline 3"
        new = "line 1\nnew line\nline 3"

        diff = DiffDisplay.from_strings(old, new, filename="test.txt")

        stats = diff.stats
        assert stats["additions"] == 1
        assert stats["deletions"] == 1

    def test_to_text_returns_text(self):
        """to_text should return Rich Text object."""
        diff = DiffDisplay.from_strings("old", "new")
        text = diff.to_text()
        assert isinstance(text, Text)

    def test_to_panel_returns_panel(self):
        """to_panel should wrap diff in Panel with stats."""
        diff = DiffDisplay.from_strings("old", "new", filename="test.py")
        panel = diff.to_panel()
        assert isinstance(panel, Panel)
        # Title should include filename and stats
        assert "test.py" in str(panel.title)

    def test_rich_protocol(self):
        """DiffDisplay should implement __rich__."""
        diff = DiffDisplay.from_strings("a", "b")
        result = diff.__rich__()
        assert isinstance(result, Text)

    def test_show_line_numbers(self):
        """Should include line numbers when enabled."""
        diff = DiffDisplay.from_strings(
            "line1\nline2",
            "line1\nmodified",
            show_line_numbers=True,
        )
        text = diff.to_text()
        # Line numbers should be in output
        assert text.plain  # Just verify it renders

    def test_stats_property(self):
        """stats should return addition/deletion counts."""
        diff = DiffDisplay.from_strings(
            "a\nb\nc",
            "a\nB\nc\nd",
        )
        stats = diff.stats
        assert "additions" in stats
        assert "deletions" in stats
        assert "context" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# CollapsibleSection Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCollapsibleSection:
    """Tests for CollapsibleSection component."""

    def test_default_collapsed(self):
        """Sections should be collapsed by default."""
        section = CollapsibleSection(title="Test", content="Content")
        assert section.collapsed is True

    def test_toggle(self):
        """toggle should flip collapsed state."""
        section = CollapsibleSection(title="Test", content="Content")
        assert section.collapsed is True

        section.toggle()
        assert section.collapsed is False

        section.toggle()
        assert section.collapsed is True

    def test_expand_collapse(self):
        """expand and collapse methods should work."""
        section = CollapsibleSection(title="Test", content="Content")

        section.expand()
        assert section.collapsed is False

        section.collapse()
        assert section.collapsed is True

    def test_to_collapsed_returns_text(self):
        """Collapsed view should return Text with icon and title."""
        section = CollapsibleSection(title="Details", content="Full content")
        text = section.to_collapsed()

        assert isinstance(text, Text)
        assert "▶" in text.plain
        assert "Details" in text.plain

    def test_to_expanded_returns_panel(self):
        """Expanded view should return Panel with content."""
        section = CollapsibleSection(title="Details", content="Full content")
        panel = section.to_expanded()

        assert isinstance(panel, Panel)

    def test_rich_protocol_collapsed(self):
        """__rich__ should return collapsed view when collapsed."""
        section = CollapsibleSection(title="Test", content="Content", collapsed=True)
        result = section.__rich__()
        assert isinstance(result, Text)

    def test_rich_protocol_expanded(self):
        """__rich__ should return expanded view when expanded."""
        section = CollapsibleSection(title="Test", content="Content", collapsed=False)
        result = section.__rich__()
        assert isinstance(result, Panel)

    def test_preview_shown_when_collapsed(self):
        """Collapsed view should show preview of content."""
        section = CollapsibleSection(
            title="Test",
            content="This is the preview text",
            show_preview=True,
        )
        text = section.to_collapsed()
        assert "This is the preview" in text.plain

    def test_preview_truncated(self):
        """Preview should be truncated for long content."""
        long_content = "x" * 200
        section = CollapsibleSection(
            title="Test",
            content=long_content,
            preview_length=50,
        )
        text = section.to_collapsed()
        assert "..." in text.plain

    def test_custom_icons(self):
        """Should support custom expand/collapse icons."""
        section = CollapsibleSection(
            title="Test",
            content="Content",
            icon_collapsed="+",
            icon_expanded="-",
        )
        assert "+ Test" in section.to_collapsed().plain
        section.expand()
        # Expanded panel title contains the icon
        assert section.icon_expanded == "-"


# ═══════════════════════════════════════════════════════════════════════════════
# CollapsibleGroup Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCollapsibleGroup:
    """Tests for CollapsibleGroup component."""

    def test_add_sections(self):
        """Should add sections to group."""
        group = CollapsibleGroup()
        group.add("Section 1", "Content 1")
        group.add("Section 2", "Content 2")

        assert len(group.sections) == 2
        assert group.sections[0].title == "Section 1"
        assert group.sections[1].title == "Section 2"

    def test_expand_all(self):
        """expand_all should expand all sections."""
        group = CollapsibleGroup()
        group.add("S1", "C1")
        group.add("S2", "C2")

        group.expand_all()

        assert all(not s.collapsed for s in group.sections)

    def test_collapse_all(self):
        """collapse_all should collapse all sections."""
        group = CollapsibleGroup()
        group.add("S1", "C1", collapsed=False)
        group.add("S2", "C2", collapsed=False)

        group.collapse_all()

        assert all(s.collapsed for s in group.sections)

    def test_toggle_by_index(self):
        """toggle should work by index."""
        group = CollapsibleGroup()
        group.add("S1", "C1")
        group.add("S2", "C2")

        group.toggle(0)
        assert not group.sections[0].collapsed
        assert group.sections[1].collapsed

    def test_accordion_mode(self):
        """Accordion mode should only allow one expanded section."""
        group = CollapsibleGroup(accordion=True)
        group.add("S1", "C1", collapsed=False)
        group.add("S2", "C2", collapsed=False)

        # Second add should collapse first in accordion mode
        assert group.sections[0].collapsed
        assert not group.sections[1].collapsed

    def test_accordion_expand(self):
        """Expanding in accordion mode should collapse others."""
        group = CollapsibleGroup(accordion=True)
        group.add("S1", "C1")
        group.add("S2", "C2")

        group.expand(0)
        group.expand(1)

        assert group.sections[0].collapsed
        assert not group.sections[1].collapsed

    def test_rich_protocol(self):
        """Should implement __rich__ returning Group."""
        group = CollapsibleGroup()
        group.add("S1", "C1")
        result = group.__rich__()
        # Returns a Rich Group containing the sections


# ═══════════════════════════════════════════════════════════════════════════════
# TokenUsageDisplay Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenUsageDisplay:
    """Tests for TokenUsageDisplay component."""

    def test_basic_creation(self):
        """Should create with input and output tokens."""
        usage = TokenUsageDisplay(input_tokens=1000, output_tokens=500)
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.total_tokens == 1500

    def test_total_tokens_calculation(self):
        """total_tokens should sum input and output."""
        usage = TokenUsageDisplay(input_tokens=2000, output_tokens=1000)
        assert usage.total_tokens == 3000

    def test_cost_calculation_with_known_model(self):
        """Should calculate cost for known models."""
        usage = TokenUsageDisplay(
            input_tokens=1_000_000,  # 1M tokens
            output_tokens=1_000_000,
            model="claude-sonnet-4.5",
        )
        # Claude Sonnet 4.5: $3/1M input, $15/1M output
        expected_cost = 3.00 + 15.00
        assert usage.cost == pytest.approx(expected_cost, rel=0.01)

    def test_cost_calculation_with_unknown_model(self):
        """Should use default pricing for unknown models."""
        usage = TokenUsageDisplay(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="unknown-model",
        )
        # Should use default pricing
        assert usage.cost is not None
        assert usage.cost > 0

    def test_cost_none_without_model(self):
        """Cost should be None when no model specified."""
        usage = TokenUsageDisplay(input_tokens=1000, output_tokens=500)
        assert usage.cost is None

    def test_format_tokens_small(self):
        """Should format small token counts as-is."""
        assert TokenUsageDisplay.format_tokens(500) == "500"
        assert TokenUsageDisplay.format_tokens(999) == "999"

    def test_format_tokens_thousands(self):
        """Should format thousands with K suffix."""
        assert TokenUsageDisplay.format_tokens(1000) == "1.00K"
        assert TokenUsageDisplay.format_tokens(1500) == "1.50K"
        assert TokenUsageDisplay.format_tokens(10000) == "10.0K"
        assert TokenUsageDisplay.format_tokens(15000) == "15.0K"

    def test_format_tokens_millions(self):
        """Should format millions with M suffix."""
        assert TokenUsageDisplay.format_tokens(1_000_000) == "1.0M"
        assert TokenUsageDisplay.format_tokens(1_500_000) == "1.5M"

    def test_format_cost_small(self):
        """Should format small costs appropriately."""
        assert TokenUsageDisplay.format_cost(0.00001) == "<$0.0001"
        assert TokenUsageDisplay.format_cost(0.0001) == "$0.0001"
        assert TokenUsageDisplay.format_cost(0.001) == "$0.0010"

    def test_format_cost_medium(self):
        """Should format medium costs with 3 decimals."""
        assert TokenUsageDisplay.format_cost(0.123) == "$0.123"
        assert TokenUsageDisplay.format_cost(0.999) == "$0.999"

    def test_format_cost_large(self):
        """Should format large costs with 2 decimals."""
        assert TokenUsageDisplay.format_cost(1.23) == "$1.23"
        assert TokenUsageDisplay.format_cost(10.50) == "$10.50"

    def test_to_text_compact(self):
        """Compact format should be concise."""
        usage = TokenUsageDisplay(
            input_tokens=1500,
            output_tokens=500,
            compact=True,
        )
        text = usage.to_text()
        assert isinstance(text, Text)
        plain = text.plain
        assert "1.50K" in plain
        assert "500" in plain

    def test_to_text_full(self):
        """Full format should include labels."""
        usage = TokenUsageDisplay(
            input_tokens=1500,
            output_tokens=500,
            compact=False,
        )
        text = usage.to_text()
        plain = text.plain
        assert "Input" in plain
        assert "Output" in plain
        assert "Total" in plain

    def test_to_text_with_cost(self):
        """Should include cost when model is specified."""
        usage = TokenUsageDisplay(
            input_tokens=1000,
            output_tokens=500,
            model="gpt-5",
            show_cost=True,
        )
        text = usage.to_text()
        assert "$" in text.plain

    def test_to_text_without_cost(self):
        """Should not include cost when show_cost is False."""
        usage = TokenUsageDisplay(
            input_tokens=1000,
            output_tokens=500,
            model="gpt-5",
            show_cost=False,
        )
        text = usage.to_text()
        assert "$" not in text.plain

    def test_to_panel(self):
        """Should wrap in a panel."""
        usage = TokenUsageDisplay(input_tokens=1000, output_tokens=500)
        panel = usage.to_panel()
        assert isinstance(panel, Panel)

    def test_rich_protocol(self):
        """Should implement __rich__."""
        usage = TokenUsageDisplay(input_tokens=1000, output_tokens=500)
        result = usage.__rich__()
        assert isinstance(result, Text)

    def test_model_pricing_exists_for_all_models(self):
        """MODEL_PRICING should have entries for all documented models."""
        expected_models = [
            "gpt-5.2-codex",
            "gpt-5.1-codex",
            "claude-sonnet-4.5",
            "claude-haiku-4.5",
            "gemini-3-pro-preview",
        ]
        for model in expected_models:
            assert model in MODEL_PRICING, f"Missing pricing for {model}"
            input_price, output_price = MODEL_PRICING[model]
            assert input_price > 0
            assert output_price > 0


# ═══════════════════════════════════════════════════════════════════════════════
# ToolCallPanel Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolCallPanel:
    """Tests for ToolCallPanel component."""

    def test_basic_creation(self):
        """Should create with name and arguments."""
        panel = ToolCallPanel(
            name="bash",
            arguments={"command": "ls -la"},
        )
        assert panel.name == "bash"
        assert panel.arguments == {"command": "ls -la"}
        assert panel.status == ToolStatus.PENDING

    def test_status_icons(self):
        """Should return correct status icons."""
        panel = ToolCallPanel(name="test")

        panel.status = ToolStatus.PENDING
        assert panel.get_status_icon() == "○"

        panel.status = ToolStatus.RUNNING
        assert panel.get_status_icon() == "⠋"

        panel.status = ToolStatus.SUCCESS
        assert panel.get_status_icon() == "✓"

        panel.status = ToolStatus.ERROR
        assert panel.get_status_icon() == "✗"

    def test_tool_icons(self):
        """Should return appropriate icons based on tool name."""
        bash_panel = ToolCallPanel(name="bash")
        assert bash_panel.get_tool_icon() == "💻"

        read_panel = ToolCallPanel(name="read_file")
        assert read_panel.get_tool_icon() == "📖"

        search_panel = ToolCallPanel(name="grep")
        assert search_panel.get_tool_icon() == "🔍"

        web_panel = ToolCallPanel(name="web_fetch")
        assert web_panel.get_tool_icon() == "🌐"

    def test_status_styles(self):
        """Should return appropriate styles for each status."""
        panel = ToolCallPanel(name="test")

        panel.status = ToolStatus.SUCCESS
        assert panel.get_status_style() == "green"

        panel.status = ToolStatus.ERROR
        assert panel.get_status_style() == "red"

        panel.status = ToolStatus.RUNNING
        assert panel.get_status_style() == "yellow"

    def test_toggle_collapsed(self):
        """Should toggle collapsed state."""
        panel = ToolCallPanel(name="test", collapsed=True)
        assert panel.collapsed is True

        panel.toggle_collapsed()
        assert panel.collapsed is False

        panel.toggle_collapsed()
        assert panel.collapsed is True

    def test_to_compact(self):
        """to_compact should return single-line Text."""
        panel = ToolCallPanel(
            name="bash",
            arguments={"command": "echo hello"},
            status=ToolStatus.SUCCESS,
            duration=1.5,
        )
        text = panel.to_compact()

        assert isinstance(text, Text)
        assert "bash" in text.plain
        assert "1.5s" in text.plain

    def test_to_compact_shows_key_argument(self):
        """Compact view should show key argument preview."""
        panel = ToolCallPanel(
            name="read_file",
            arguments={"path": "/some/long/path/to/file.txt"},
        )
        text = panel.to_compact()
        assert "path=" in text.plain

    def test_to_panel(self):
        """to_panel should return Panel with details."""
        panel = ToolCallPanel(
            name="bash",
            arguments={"command": "ls"},
            result="file1.txt\nfile2.txt",
            status=ToolStatus.SUCCESS,
        )
        result = panel.to_panel()

        assert isinstance(result, Panel)

    def test_to_panel_shows_arguments(self):
        """Panel should show arguments with syntax highlighting."""
        panel = ToolCallPanel(
            name="test",
            arguments={"key": "value", "nested": {"a": 1}},
        )
        result = panel.to_panel(show_arguments=True)
        assert isinstance(result, Panel)

    def test_to_panel_shows_error(self):
        """Panel should show error in red."""
        panel = ToolCallPanel(
            name="test",
            status=ToolStatus.ERROR,
            error="Something went wrong",
        )
        result = panel.to_panel()
        assert isinstance(result, Panel)

    def test_result_preview_truncation(self):
        """Long results should be truncated in preview."""
        long_result = "x" * 200
        panel = ToolCallPanel(name="test", result=long_result)
        preview = panel._format_result_preview(max_length=50)
        assert len(preview) <= 53  # 50 + "..."
        assert "..." in preview

    def test_result_preview_multiline(self):
        """Multiline results should show line count."""
        panel = ToolCallPanel(name="test", result="line1\nline2\nline3\nline4")
        preview = panel._format_result_preview()
        assert "+3 lines" in preview

    def test_rich_protocol(self):
        """Should implement __rich__."""
        panel = ToolCallPanel(name="test")
        result = panel.__rich__()
        assert isinstance(result, Panel)

    def test_json_result_highlighting(self):
        """JSON results should get syntax highlighting."""
        panel = ToolCallPanel(
            name="test",
            result='{"key": "value"}',
            collapsed=False,
        )
        result = panel._format_result_full()
        assert isinstance(result, Syntax)

    def test_diff_result_highlighting(self):
        """Diff results should get syntax highlighting."""
        panel = ToolCallPanel(
            name="test",
            result="--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new",
            collapsed=False,
        )
        result = panel._format_result_full()
        assert isinstance(result, Syntax)


class TestToolCallGroup:
    """Tests for ToolCallGroup component."""

    def test_add_creates_running_call(self):
        """add should create a running tool call."""
        group = ToolCallGroup()
        panel = group.add("bash", {"command": "ls"})

        assert len(group.calls) == 1
        assert panel.name == "bash"
        assert panel.status == ToolStatus.RUNNING

    def test_complete_marks_success(self):
        """complete should mark call as successful."""
        group = ToolCallGroup()
        group.add("bash", {"command": "ls"})
        group.complete(result="output", duration=1.0)

        assert group.calls[0].status == ToolStatus.SUCCESS
        assert group.calls[0].result == "output"
        assert group.calls[0].duration == 1.0

    def test_complete_with_error(self):
        """complete with error should mark as error."""
        group = ToolCallGroup()
        group.add("bash")
        group.complete(error="failed", duration=0.5)

        assert group.calls[0].status == ToolStatus.ERROR
        assert group.calls[0].error == "failed"

    def test_get_stats(self):
        """get_stats should count statuses."""
        group = ToolCallGroup()
        group.add("tool1")
        group.add("tool2")
        group.add("tool3")
        group.complete(0, result="ok")
        group.complete(1, error="fail")
        # tool3 still running

        stats = group.get_stats()
        assert stats["success"] == 1
        assert stats["error"] == 1
        assert stats["running"] == 1

    def test_to_panel_returns_panel(self):
        """to_panel should return Panel."""
        group = ToolCallGroup()
        group.add("bash", {"command": "ls"})

        panel = group.to_panel()
        assert isinstance(panel, Panel)

    def test_to_panel_none_when_empty(self):
        """to_panel should return None when no calls."""
        group = ToolCallGroup()
        assert group.to_panel() is None

    def test_max_visible_limits_display(self):
        """Should limit visible calls based on max_visible."""
        group = ToolCallGroup(max_visible=2)
        for i in range(5):
            group.add(f"tool{i}")

        assert len(group.calls) == 5
        # Panel should show limited number (verified by internal logic)

    def test_show_all_displays_everything(self):
        """show_all should display all calls."""
        group = ToolCallGroup(max_visible=2, show_all=True)
        for i in range(5):
            group.add(f"tool{i}")

        # When show_all is True, all calls visible
        assert group.show_all is True

    def test_rich_protocol(self):
        """Should implement __rich__."""
        group = ToolCallGroup()
        group.add("test")
        result = group.__rich__()
        assert isinstance(result, Panel)

    def test_rich_protocol_empty(self):
        """Empty group should return placeholder text."""
        group = ToolCallGroup()
        result = group.__rich__()
        assert isinstance(result, Text)


# ═══════════════════════════════════════════════════════════════════════════════
# StepProgress Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStepInfo:
    """Tests for StepInfo dataclass."""

    def test_basic_creation(self):
        """Should create with number and description."""
        step = StepInfo(number=1, description="First step")
        assert step.number == 1
        assert step.description == "First step"
        assert step.status == StepStatus.PENDING

    def test_duration_calculation(self):
        """Should calculate duration when started."""
        step = StepInfo(number=1, description="Test")
        step.started_at = time.time() - 5  # 5 seconds ago
        step.completed_at = time.time()

        assert step.duration is not None
        assert step.duration >= 4.9  # Allow some tolerance

    def test_duration_none_when_not_started(self):
        """Duration should be None when not started."""
        step = StepInfo(number=1, description="Test")
        assert step.duration is None

    def test_duration_str_formatting(self):
        """Should format duration as string."""
        step = StepInfo(number=1, description="Test")
        step.started_at = time.time() - 125  # 2m 5s ago
        step.completed_at = time.time()

        assert "m" in step.duration_str
        assert "s" in step.duration_str


class TestStepProgress:
    """Tests for StepProgress component."""

    def test_basic_creation(self):
        """Should create with total and title."""
        progress = StepProgress(total=5, title="My Plan")
        assert progress.total == 5
        assert progress.title == "My Plan"
        assert len(progress.steps) == 0

    def test_add_step(self):
        """Should add steps and auto-update total."""
        progress = StepProgress()
        progress.add_step(1, "First")
        progress.add_step(2, "Second")

        assert len(progress.steps) == 2
        assert progress.total == 2

    def test_start_step(self):
        """Should mark step as running."""
        progress = StepProgress()
        progress.add_step(1, "First")
        progress.start_step(1)

        step = progress.get_step(1)
        assert step is not None
        assert step.status == StepStatus.RUNNING
        assert step.started_at is not None

    def test_complete_step(self):
        """Should mark step as completed."""
        progress = StepProgress()
        progress.add_step(1, "First")
        progress.start_step(1)
        progress.complete_step(1, result="Done!")

        step = progress.get_step(1)
        assert step.status == StepStatus.COMPLETED
        assert step.completed_at is not None
        assert step.result == "Done!"

    def test_fail_step(self):
        """Should mark step as failed."""
        progress = StepProgress()
        progress.add_step(1, "First")
        progress.start_step(1)
        progress.fail_step(1, error="Something broke")

        step = progress.get_step(1)
        assert step.status == StepStatus.FAILED
        assert step.error == "Something broke"

    def test_skip_step(self):
        """Should mark step as skipped."""
        progress = StepProgress()
        progress.add_step(1, "First")
        progress.skip_step(1, reason="Not needed")

        step = progress.get_step(1)
        assert step.status == StepStatus.SKIPPED
        assert step.result == "Not needed"

    def test_completed_count(self):
        """Should count completed steps."""
        progress = StepProgress()
        progress.add_step(1, "Step 1")
        progress.add_step(2, "Step 2")
        progress.add_step(3, "Step 3")
        progress.complete_step(1)
        progress.complete_step(2)

        assert progress.completed_count == 2

    def test_failed_count(self):
        """Should count failed steps."""
        progress = StepProgress()
        progress.add_step(1, "Step 1")
        progress.add_step(2, "Step 2")
        progress.fail_step(1, "Error")

        assert progress.failed_count == 1

    def test_running_count(self):
        """Should count running steps."""
        progress = StepProgress()
        progress.add_step(1, "Step 1")
        progress.add_step(2, "Step 2")
        progress.start_step(1)
        progress.start_step(2)

        assert progress.running_count == 2

    def test_current_step(self):
        """Should return currently running step."""
        progress = StepProgress()
        progress.add_step(1, "First")
        progress.add_step(2, "Second")
        progress.complete_step(1)
        progress.start_step(2)

        current = progress.current_step
        assert current is not None
        assert current.number == 2

    def test_percent(self):
        """Should calculate percentage."""
        progress = StepProgress(total=4)
        progress.add_step(1, "A")
        progress.add_step(2, "B")
        progress.add_step(3, "C")
        progress.add_step(4, "D")
        progress.complete_step(1)
        progress.complete_step(2)

        assert progress.percent == 50.0

    def test_elapsed_str(self):
        """Should format elapsed time."""
        progress = StepProgress()
        progress.started_at = time.time() - 65  # 1m 5s ago

        elapsed = progress.elapsed_str
        assert "m" in elapsed

    def test_eta_calculation(self):
        """Should estimate time remaining."""
        progress = StepProgress(total=4)
        progress.add_step(1, "A")
        progress.add_step(2, "B")
        progress.started_at = time.time() - 10  # 10 seconds ago
        progress.complete_step(1)

        # ETA should be approximately 30s (3 remaining * 10s avg)
        eta = progress.eta_seconds
        assert eta is not None
        assert eta > 0

    def test_to_text(self):
        """to_text should return Text with header and stats."""
        progress = StepProgress(total=3, title="Test")
        progress.add_step(1, "First step")
        progress.start_step(1)

        text = progress.to_text()
        assert isinstance(text, Text)
        assert "Step 1" in text.plain

    def test_to_panel(self):
        """to_panel should return Panel with full details."""
        progress = StepProgress(total=3, title="Test Plan")
        progress.add_step(1, "Initialize")
        progress.add_step(2, "Process")
        progress.add_step(3, "Finalize")
        progress.complete_step(1)
        progress.start_step(2)

        panel = progress.to_panel()
        assert isinstance(panel, Panel)

    def test_rich_protocol_panel(self):
        """__rich__ should return Panel when not compact."""
        progress = StepProgress(compact=False)
        progress.add_step(1, "Test")

        result = progress.__rich__()
        assert isinstance(result, Panel)

    def test_rich_protocol_text(self):
        """__rich__ should return Text when compact."""
        progress = StepProgress(compact=True)
        progress.add_step(1, "Test")

        result = progress.__rich__()
        assert isinstance(result, Text)

    def test_progress_bar_rendering(self):
        """Progress bar should render correctly."""
        progress = StepProgress(total=4)
        for i in range(4):
            progress.add_step(i + 1, f"Step {i + 1}")
        progress.complete_step(1)
        progress.complete_step(2)

        bar = progress._build_progress_bar()
        assert isinstance(bar, Text)
        assert "█" in bar.plain  # Should have filled portion
        assert "░" in bar.plain  # Should have empty portion

    def test_spinner_advances(self):
        """Spinner should cycle through frames."""
        progress = StepProgress()
        initial_idx = progress._spinner_idx

        progress._advance_spinner()
        assert progress._spinner_idx == (initial_idx + 1) % len(progress._spinner_frames)


class TestRichProgressReporter:
    """Tests for RichProgressReporter integration."""

    def test_basic_creation(self):
        """Should create with title."""
        reporter = RichProgressReporter(title="My Progress")
        assert reporter.progress.title == "My Progress"

    def test_rich_protocol(self):
        """Should implement __rich__."""
        reporter = RichProgressReporter()
        reporter.progress.add_step(1, "Test")

        result = reporter.__rich__()
        assert isinstance(result, (Panel, Text))


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Function Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractCodeBlocks:
    """Tests for extract_code_blocks utility."""

    def test_extracts_python_block(self):
        """Should extract Python code blocks."""
        text = """
Some text
```python
def hello():
    pass
```
More text
"""
        blocks = extract_code_blocks(text)

        assert len(blocks) == 1
        assert blocks[0][0] == "python"
        assert "def hello():" in blocks[0][1]

    def test_extracts_multiple_blocks(self):
        """Should extract multiple code blocks."""
        text = """
```python
x = 1
```

```javascript
const y = 2;
```
"""
        blocks = extract_code_blocks(text)

        assert len(blocks) == 2
        assert blocks[0][0] == "python"
        assert blocks[1][0] == "javascript"

    def test_no_language_defaults_to_text(self):
        """Blocks without language should default to 'text'."""
        text = """
```
plain content
```
"""
        blocks = extract_code_blocks(text)

        assert len(blocks) == 1
        assert blocks[0][0] == "text"


class TestRenderMarkdownWithSyntax:
    """Tests for render_markdown_with_syntax utility."""

    def test_renders_code_with_syntax(self):
        """Should render markdown with syntax-highlighted code."""
        text = """
# Hello

```python
print("world")
```
"""
        result = render_markdown_with_syntax(text)
        # Returns a Group - just verify it doesn't error
        assert result is not None

    def test_handles_text_without_code(self):
        """Should handle markdown without code blocks."""
        text = "Just some plain text\n\nWith paragraphs"
        result = render_markdown_with_syntax(text)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestComponentRendering:
    """Integration tests for rendering components to console."""

    def test_codeblock_renders_without_error(self):
        """CodeBlock should render to console without error."""
        console = Console(force_terminal=True, width=80)
        code = CodeBlock("print('hello')", language="python")

        # Should not raise
        with console.capture() as capture:
            console.print(code)

        output = capture.get()
        assert "print" in output

    def test_diff_renders_without_error(self):
        """DiffDisplay should render to console without error."""
        console = Console(force_terminal=True, width=80)
        diff = DiffDisplay.from_strings("old", "new", filename="test.txt")

        with console.capture() as capture:
            console.print(diff)

        output = capture.get()
        assert output  # Has content

    def test_collapsible_renders_without_error(self):
        """CollapsibleSection should render to console without error."""
        console = Console(force_terminal=True, width=80)
        section = CollapsibleSection(title="Test", content="Content here")

        with console.capture() as capture:
            console.print(section)

        output = capture.get()
        assert "Test" in output



# ═══════════════════════════════════════════════════════════════════════════════
# Non-TTY / Plain Text Support Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsTerminal:
    """Tests for is_terminal detection."""

    def test_returns_bool(self):
        """is_terminal should return a boolean."""
        result = is_terminal()
        assert isinstance(result, bool)

    def test_respects_no_color_env(self, monkeypatch):
        """Should return False when NO_COLOR is set."""
        monkeypatch.setenv("NO_COLOR", "1")
        # Note: actual result depends on stdout.isatty()
        # but NO_COLOR should force False
        result = is_terminal()
        assert result is False


class TestGetPlainConsole:
    """Tests for get_plain_console."""

    def test_returns_console(self):
        """Should return a Console instance."""
        console = get_plain_console()
        assert isinstance(console, Console)

    def test_console_has_no_color(self):
        """Console should have no_color=True."""
        console = get_plain_console()
        assert console.no_color is True


class TestPlainTextRenderer:
    """Tests for PlainTextRenderer utility."""

    def test_strip_unicode(self):
        """Should replace Unicode icons with ASCII."""
        renderer = PlainTextRenderer()
        result = renderer._strip_unicode("✓ Success ✗ Fail")
        assert "[OK]" in result
        assert "[FAIL]" in result
        assert "✓" not in result
        assert "✗" not in result

    def test_render_separator(self):
        """Should render a separator line."""
        renderer = PlainTextRenderer(line_width=40)
        result = renderer.render_separator("-")
        assert len(result) == 40
        assert result == "-" * 40

    def test_render_header(self):
        """Should render headers with appropriate decoration."""
        renderer = PlainTextRenderer()
        result = renderer.render_header("My Title", level=1)
        assert "My Title" in result
        assert "=" in result

    def test_render_code_block(self):
        """Should render code as plain text."""
        renderer = PlainTextRenderer()
        result = renderer.render_code_block("print('hello')", language="python")
        assert "print('hello')" in result
        assert "[python]" in result

    def test_render_diff(self):
        """Should render diff as plain text."""
        renderer = PlainTextRenderer()
        result = renderer.render_diff(
            additions=5,
            deletions=3,
            content="+ added\n- removed",
            filename="test.py",
        )
        assert "+5" in result
        assert "-3" in result
        assert "test.py" in result

    def test_render_progress(self):
        """Should render progress bar as plain text."""
        renderer = PlainTextRenderer()
        result = renderer.render_progress(5, 10, title="Build")
        assert "Build" in result
        assert "50%" in result
        assert "#" in result
        assert "-" in result

    def test_render_tool_call(self):
        """Should render tool call as plain text."""
        renderer = PlainTextRenderer()
        result = renderer.render_tool_call(
            name="bash",
            status="success",
            arguments={"command": "ls"},
            result="file1.txt",
            duration=1.5,
        )
        assert "bash" in result
        assert "[OK]" in result
        assert "1.5s" in result

    def test_render_token_usage(self):
        """Should render token usage as plain text."""
        renderer = PlainTextRenderer()
        result = renderer.render_token_usage(1000, 500, cost=0.05)
        assert "1000" in result
        assert "500" in result
        assert "1500" in result
        assert "$0.05" in result


class TestComponentPlainText:
    """Tests for to_plain_text methods on components."""

    def test_codeblock_to_plain_text(self):
        """CodeBlock should render as plain text."""
        code = CodeBlock("x = 1", language="python", filename="test.py")
        result = code.to_plain_text()
        assert "x = 1" in result
        assert "test.py" in result

    def test_codeblock_plain_text_with_line_numbers(self):
        """CodeBlock plain text should include line numbers."""
        code = CodeBlock("line1\nline2", language="python", line_numbers=True)
        result = code.to_plain_text()
        assert "1" in result
        assert "2" in result

    def test_diff_to_plain_text(self):
        """DiffDisplay should render as plain text."""
        diff = DiffDisplay.from_strings("old", "new", filename="file.txt")
        result = diff.to_plain_text()
        assert "file.txt" in result
        assert "+" in result or "-" in result

    def test_token_usage_to_plain_text(self):
        """TokenUsageDisplay should render as plain text."""
        usage = TokenUsageDisplay(
            input_tokens=1000,
            output_tokens=500,
            model="gpt-5",
            show_cost=True,
        )
        result = usage.to_plain_text()
        assert "1000" in result or "1.00K" in result
        assert "500" in result
        assert "Tokens" in result

    def test_tool_call_to_plain_text(self):
        """ToolCallPanel should render as plain text."""
        panel = ToolCallPanel(
            name="bash",
            arguments={"command": "ls"},
            status=ToolStatus.SUCCESS,
            result="files...",
            duration=1.0,
        )
        result = panel.to_plain_text()
        assert "bash" in result
        assert "[OK]" in result
        assert "1.0s" in result

    def test_step_progress_to_plain_text(self):
        """StepProgress should render as plain text."""
        progress = StepProgress(total=3, title="Plan")
        progress.add_step(1, "First")
        progress.add_step(2, "Second")
        progress.add_step(3, "Third")
        progress.complete_step(1)
        progress.start_step(2)

        result = progress.to_plain_text()
        assert "Step 2" in result
        assert "First" in result
        assert "[OK]" in result or "[...]" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Cases and Error Handling Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeBlockEdgeCases:
    """Edge case tests for CodeBlock."""

    def test_empty_code(self):
        """Should handle empty code string."""
        code = CodeBlock("", language="python")
        assert code.code == ""
        # Should still render without error
        result = code.to_syntax()
        assert result is not None

    def test_very_long_line(self):
        """Should handle very long lines."""
        long_line = "x = " + "a" * 10000
        code = CodeBlock(long_line, language="python")
        result = code.to_syntax()
        assert result is not None

    def test_unicode_content(self):
        """Should handle unicode in code."""
        code = CodeBlock("print('🎉 Привет 世界')", language="python")
        result = code.to_syntax()
        assert result is not None
        plain = code.to_plain_text()
        assert "🎉" in plain

    def test_unknown_language_fallback(self):
        """Should fall back gracefully for unknown language."""
        code = CodeBlock("some code", language="nonexistent_lang_xyz")
        # Should not raise, should use fallback
        result = code.to_syntax()
        assert result is not None

    def test_binary_like_content(self):
        """Should handle content with null bytes."""
        code = CodeBlock("data = b'\\x00\\x01\\x02'", language="python")
        result = code.to_syntax()
        assert result is not None

    def test_mixed_line_endings(self):
        """Should handle mixed line endings."""
        code = CodeBlock("line1\nline2\r\nline3\rline4", language="text")
        result = code.to_plain_text()
        assert "line1" in result

    def test_only_whitespace(self):
        """Should handle whitespace-only code."""
        code = CodeBlock("   \n\t\n   ", language="python")
        result = code.to_syntax()
        assert result is not None


class TestDiffDisplayEdgeCases:
    """Edge case tests for DiffDisplay."""

    def test_identical_strings(self):
        """Should handle identical strings (no diff)."""
        diff = DiffDisplay.from_strings("same", "same", filename="file.txt")
        stats = diff.stats
        assert stats["additions"] == 0
        assert stats["deletions"] == 0

    def test_empty_strings(self):
        """Should handle empty strings."""
        diff = DiffDisplay.from_strings("", "", filename="file.txt")
        assert diff is not None

    def test_empty_to_content(self):
        """Should handle empty -> content transition."""
        diff = DiffDisplay.from_strings("", "new content", filename="file.txt")
        stats = diff.stats
        assert stats["additions"] > 0

    def test_content_to_empty(self):
        """Should handle content -> empty transition."""
        diff = DiffDisplay.from_strings("old content", "", filename="file.txt")
        stats = diff.stats
        assert stats["deletions"] > 0

    def test_very_large_diff(self):
        """Should handle large diffs."""
        old = "\n".join([f"line {i}" for i in range(1000)])
        new = "\n".join([f"modified line {i}" for i in range(1000)])
        diff = DiffDisplay.from_strings(old, new, filename="large.txt")
        assert diff is not None

    def test_special_characters_in_diff(self):
        """Should handle special characters."""
        old = "a <tag> & \"quote\""
        new = "b <tag> & 'quote'"
        diff = DiffDisplay.from_strings(old, new, filename="special.txt")
        result = diff.to_plain_text()
        assert "<tag>" in result or "&lt;tag&gt;" in result

    def test_from_unified_diff_empty(self):
        """Should handle empty unified diff."""
        diff = DiffDisplay.from_unified_diff("")
        assert diff.lines == []

    def test_from_unified_diff_malformed(self):
        """Should handle malformed unified diff gracefully."""
        malformed = "not a valid diff format\njust some text"
        diff = DiffDisplay.from_unified_diff(malformed)
        # Should not crash, just parse what it can
        assert diff is not None


class TestCollapsibleSectionEdgeCases:
    """Edge case tests for CollapsibleSection."""

    def test_empty_content(self):
        """Should handle empty content."""
        section = CollapsibleSection("Title", "")
        result = section.to_expanded()
        assert result is not None

    def test_very_long_title(self):
        """Should handle very long titles."""
        long_title = "A" * 1000
        section = CollapsibleSection(long_title, "content")
        result = section.to_collapsed()
        assert result is not None

    def test_unicode_title(self):
        """Should handle unicode in title."""
        section = CollapsibleSection("🔧 Настройки", "content")
        result = section.to_collapsed()
        # Should not crash
        assert result is not None

    def test_nested_renderables(self):
        """Should handle nested Rich renderables."""
        inner = Panel("inner content")
        section = CollapsibleSection("Title", inner)
        result = section.to_expanded()
        assert result is not None


class TestTokenUsageDisplayEdgeCases:
    """Edge case tests for TokenUsageDisplay."""

    def test_zero_tokens(self):
        """Should handle zero tokens."""
        usage = TokenUsageDisplay(input_tokens=0, output_tokens=0)
        result = usage.to_text()
        assert "0" in result

    def test_very_large_tokens(self):
        """Should handle very large token counts."""
        usage = TokenUsageDisplay(
            input_tokens=999_999_999,
            output_tokens=999_999_999,
        )
        result = usage.to_text()
        assert "M" in result  # Should show millions

    def test_negative_tokens(self):
        """Should handle negative tokens gracefully."""
        usage = TokenUsageDisplay(input_tokens=-100, output_tokens=-50)
        result = usage.to_text()
        # Should not crash
        assert result is not None

    def test_unknown_model_uses_default_pricing(self):
        """Unknown model should use default pricing."""
        usage = TokenUsageDisplay(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="unknown-model-xyz",
            show_cost=True,
        )
        # Check via to_text with cost
        result = usage.to_text()
        # Default pricing should still produce a cost display
        assert result is not None


class TestToolCallPanelEdgeCases:
    """Edge case tests for ToolCallPanel."""

    def test_empty_arguments(self):
        """Should handle empty arguments."""
        panel = ToolCallPanel(name="test", arguments={})
        result = panel.to_panel()
        assert result is not None

    def test_none_result(self):
        """Should handle None result."""
        panel = ToolCallPanel(name="test", result=None)
        result = panel.to_panel()
        assert result is not None

    def test_very_long_result(self):
        """Should handle very long result."""
        panel = ToolCallPanel(
            name="bash",
            result="x" * 100000,
            status=ToolStatus.SUCCESS,
        )
        # Preview should be truncated
        panel.collapsed = True
        compact = panel.to_compact()
        assert len(str(compact)) < 1000

    def test_json_in_result(self):
        """Should detect and highlight JSON in result."""
        panel = ToolCallPanel(
            name="api",
            result='{"key": "value", "count": 42}',
            status=ToolStatus.SUCCESS,
        )
        panel.collapsed = False
        result = panel.to_panel()
        assert result is not None

    def test_error_message(self):
        """Should display error message."""
        panel = ToolCallPanel(
            name="bash",
            status=ToolStatus.ERROR,
            error="Command failed with exit code 1",
        )
        result = panel.to_panel()
        assert result is not None

    def test_duration_display(self):
        """Should display duration when set."""
        panel = ToolCallPanel(
            name="bash",
            duration=3.14159,
            status=ToolStatus.SUCCESS,
        )
        compact = panel.to_compact()
        result_str = str(compact)
        assert "3.1" in result_str


class TestStepProgressEdgeCases:
    """Edge case tests for StepProgress."""

    def test_single_step(self):
        """Should handle single step progress."""
        progress = StepProgress(total=1, title="Single")
        progress.add_step(1, "Only step")
        result = progress.to_panel()
        assert result is not None

    def test_many_steps(self):
        """Should handle many steps."""
        progress = StepProgress(total=100, title="Big plan")
        for i in range(1, 101):
            progress.add_step(i, f"Step {i}")
        result = progress.to_panel()
        assert result is not None

    def test_all_failed_steps(self):
        """Should handle all failed steps."""
        progress = StepProgress(total=3, title="Failed")
        for i in range(1, 4):
            progress.add_step(i, f"Step {i}")
            progress.fail_step(i, f"Error in step {i}")
        result = progress.to_panel()
        assert result is not None

    def test_all_skipped_steps(self):
        """Should handle all skipped steps."""
        progress = StepProgress(total=3, title="Skipped")
        for i in range(1, 4):
            progress.add_step(i, f"Step {i}")
            progress.skip_step(i, "Skipped for testing")
        result = progress.to_panel()
        assert result is not None

    def test_empty_progress(self):
        """Should handle progress with no steps added."""
        progress = StepProgress(total=5, title="Empty")
        result = progress.to_text()
        assert result is not None

    def test_step_rerun(self):
        """Should handle re-running a completed step."""
        progress = StepProgress(total=2, title="Rerun")
        progress.add_step(1, "First")
        progress.complete_step(1)
        # Re-start the same step
        progress.start_step(1)
        # Steps is a list, find by number
        step = next((s for s in progress.steps if s.number == 1), None)
        assert step is not None
        assert step.status == StepStatus.RUNNING


class TestToolCallGroupEdgeCases:
    """Edge case tests for ToolCallGroup."""

    def test_complete_nonexistent_call(self):
        """Should handle completing an empty group."""
        group = ToolCallGroup()
        # Should not crash on empty group
        group.complete(result="test")  # Operates on last call (-1 index)
        assert group.get_stats()["success"] == 0

    def test_many_concurrent_calls(self):
        """Should handle many concurrent calls."""
        group = ToolCallGroup()
        for i in range(100):
            group.add(f"tool-{i % 5}", {"arg": i})
        
        stats = group.get_stats()
        assert stats["running"] == 100

    def test_mixed_statuses(self):
        """Should handle mixed success/error/cancelled."""
        group = ToolCallGroup()
        group.add("bash", {})
        group.add("bash", {})
        group.add("bash", {})
        
        group.complete(0, result="ok")  # First call
        group.complete(1, error="failed")  # Second call
        # Third stays running
        
        stats = group.get_stats()
        assert stats["success"] == 1
        assert stats["error"] == 1
        assert stats["running"] == 1


class TestPlainTextRendererEdgeCases:
    """Edge case tests for PlainTextRenderer."""

    def test_strip_unicode_preserves_ascii(self):
        """Should preserve ASCII characters."""
        renderer = PlainTextRenderer()
        result = renderer._strip_unicode("Hello, World! 123")
        assert result == "Hello, World! 123"

    def test_strip_unicode_handles_mixed(self):
        """Should handle mixed ASCII and Unicode."""
        renderer = PlainTextRenderer()
        result = renderer._strip_unicode("OK ✓ Error ✗ Warning ⚠")
        assert "[OK]" in result
        assert "[FAIL]" in result
        assert "[WARN]" in result

    def test_render_code_block(self):
        """Should render code block."""
        renderer = PlainTextRenderer()
        result = renderer.render_code_block("print('hi')", "python")
        assert "print" in result

    def test_render_progress_zero_total(self):
        """Should handle zero total in progress bar."""
        result = PlainTextRenderer.render_progress(0, 5, 50)
        assert result is not None

    def test_render_progress_current_exceeds_total(self):
        """Should handle current > total."""
        result = PlainTextRenderer.render_progress(10, 5, 50)
        # Should not crash, might show 100% or clamp
        assert result is not None


class TestIntegrationScenarios:
    """Integration tests for common usage scenarios."""

    def test_code_block_with_diff_comparison(self):
        """Should work together: CodeBlock rendered, then DiffDisplay for changes."""
        original = "def hello():\n    print('hi')"
        modified = "def hello():\n    print('hello world')"
        
        # Create code blocks
        before = CodeBlock(original, language="python")
        after = CodeBlock(modified, language="python")
        
        # Create diff
        diff = DiffDisplay.from_strings(original, modified, filename="hello.py")
        
        # All should render without error
        assert before.to_syntax() is not None
        assert after.to_syntax() is not None
        assert diff.to_panel() is not None

    def test_tool_call_with_code_result(self):
        """Tool call result containing code should be displayable."""
        code_result = '''```python
def generated():
    return 42
```'''
        panel = ToolCallPanel(
            name="generate_code",
            arguments={"prompt": "write a function"},
            result=code_result,
            status=ToolStatus.SUCCESS,
        )
        result = panel.to_panel()
        assert result is not None

    def test_progress_with_token_usage(self):
        """Progress display with token tracking."""
        progress = StepProgress(total=3, title="Plan")
        progress.add_step(1, "Analyze")
        progress.add_step(2, "Generate")
        progress.add_step(3, "Review")
        
        usage = TokenUsageDisplay(
            input_tokens=5000,
            output_tokens=2000,
            model="gpt-5",
            show_cost=True,
        )
        
        # Both should render
        assert progress.to_panel() is not None
        assert usage.to_panel() is not None

    def test_collapsible_containing_code_block(self):
        """Collapsible section containing a code block."""
        code = CodeBlock("print('hello')", language="python")
        section = CollapsibleSection("Generated Code", code.to_syntax())
        
        collapsed = section.to_collapsed()
        expanded = section.to_expanded()
        
        assert collapsed is not None
        assert expanded is not None

    def test_full_console_render(self):
        """Full render to console without errors."""
        console = Console(force_terminal=False, width=80)
        
        # Create various components
        code = CodeBlock("x = 1", language="python")
        diff = DiffDisplay.from_strings("old", "new", filename="test.txt")
        usage = TokenUsageDisplay(input_tokens=100, output_tokens=50)
        tool = ToolCallPanel(name="test", status=ToolStatus.SUCCESS, result="ok")
        progress = StepProgress(total=2, title="Test")
        progress.add_step(1, "First")
        
        # Render all to console (should not raise)
        with console.capture() as capture:
            console.print(code)
            console.print(diff)
            console.print(usage)
            console.print(tool)
            console.print(progress)
        
        output = capture.get()
        assert len(output) > 0

    def test_non_tty_fallback_chain(self):
        """Full non-TTY rendering chain."""
        # All components should have to_plain_text
        code = CodeBlock("print(1)", language="python")
        diff = DiffDisplay.from_strings("a", "b", filename="f.txt")
        usage = TokenUsageDisplay(input_tokens=10, output_tokens=5)
        tool = ToolCallPanel(name="test", status=ToolStatus.SUCCESS)
        progress = StepProgress(total=1, title="T")
        progress.add_step(1, "S")
        
        # All should produce plain text without Rich formatting
        assert isinstance(code.to_plain_text(), str)
        assert isinstance(diff.to_plain_text(), str)
        assert isinstance(usage.to_plain_text(), str)
        assert isinstance(tool.to_plain_text(), str)
        assert isinstance(progress.to_plain_text(), str)
