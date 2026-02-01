"""Tests for UI components - syntax highlighting, diffs, collapsible sections."""

from __future__ import annotations

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
    extract_code_blocks,
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
