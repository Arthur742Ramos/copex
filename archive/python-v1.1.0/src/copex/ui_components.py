"""Core UI components for Copex - syntax highlighting, diffs, collapsible sections."""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rich.box import ROUNDED
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.panel import Panel
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text


# ═══════════════════════════════════════════════════════════════════════════════
# Non-TTY / Plain Text Support
# ═══════════════════════════════════════════════════════════════════════════════


def is_terminal() -> bool:
    """
    Check if we're running in a terminal (TTY) environment.
    
    Returns False when output is piped, redirected, or in non-interactive environments.
    """
    # Check if stdout is a TTY
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False
    
    # Check for explicit NO_COLOR environment variable (https://no-color.org/)
    if os.environ.get("NO_COLOR"):
        return False
    
    # Check for CI/CD environments that might not support rich output
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        # Still allow terminal features in CI if TERM is set
        if not os.environ.get("TERM"):
            return False
    
    return True


def get_plain_console() -> Console:
    """Get a Console configured for plain text output (no colors, no markup)."""
    return Console(
        force_terminal=False,
        no_color=True,
        markup=False,
        highlight=False,
        width=120,
    )


@dataclass
class PlainTextRenderer:
    """
    Utility for rendering UI components as plain text.
    
    Used for non-TTY environments (pipes, redirects, CI/CD, etc.)
    where Rich formatting would create noise or break parsing.
    
    Usage:
        if not is_terminal():
            renderer = PlainTextRenderer()
            print(renderer.render_code_block(code_block))
    """
    
    # ASCII substitutes for Unicode icons
    ASCII_ICONS: dict[str, str] = field(default_factory=lambda: {
        "✓": "[OK]",
        "✗": "[FAIL]",
        "⠋": "[...]",
        "⠙": "[...]",
        "⠹": "[...]",
        "○": "[ ]",
        "●": "[*]",
        "⊘": "[SKIP]",
        "⚠": "[WARN]",
        "→": "->",
        "←": "<-",
        "▶": ">",
        "▼": "v",
        "█": "#",
        "░": "-",
        "│": "|",
        "💻": "$",
        "📖": "R",
        "📝": "W",
        "📄": "+",
        "🔍": "?",
        "🌐": "@",
        "⚡": "*",
        "🧠": "!",
        "🤖": "AI",
        "⏱": "t",
        "✨": "*",
    })
    
    line_width: int = 80
    
    def _strip_unicode(self, text: str) -> str:
        """Replace Unicode icons with ASCII equivalents."""
        result = text
        for unicode_char, ascii_char in self.ASCII_ICONS.items():
            result = result.replace(unicode_char, ascii_char)
        return result
    
    def render_separator(self, char: str = "-") -> str:
        """Render a separator line."""
        return char * self.line_width
    
    def render_header(self, title: str, level: int = 1) -> str:
        """Render a section header."""
        title = self._strip_unicode(title)
        if level == 1:
            return f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}"
        elif level == 2:
            return f"\n{title}\n{'-' * len(title)}"
        return f"\n{title}"
    
    def render_code_block(self, code: str, language: str = "", filename: str = "") -> str:
        """Render a code block as plain text."""
        lines = []
        header = filename or f"[{language}]" if language else "[code]"
        lines.append(f"--- {header} ---")
        lines.append(code)
        lines.append("-" * (len(header) + 8))
        return "\n".join(lines)
    
    def render_diff(
        self,
        additions: int,
        deletions: int,
        content: str,
        filename: str = "",
    ) -> str:
        """Render a diff as plain text."""
        lines = []
        header = f"--- {filename} ---" if filename else "--- diff ---"
        stats = f"+{additions} -{deletions}"
        lines.append(f"{header} ({stats})")
        lines.append(content)
        return "\n".join(lines)
    
    def render_progress(
        self,
        current: int,
        total: int,
        title: str = "Progress",
        eta: str | None = None,
    ) -> str:
        """Render a progress indicator as plain text."""
        percent = (current / total * 100) if total > 0 else 0
        bar_width = 30
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "#" * filled + "-" * (bar_width - filled)
        
        line = f"{title}: [{bar}] {current}/{total} ({percent:.0f}%)"
        if eta:
            line += f" ETA: {eta}"
        return line
    
    def render_tool_call(
        self,
        name: str,
        status: str,
        arguments: dict[str, Any] | None = None,
        result: str | None = None,
        duration: float | None = None,
    ) -> str:
        """Render a tool call as plain text."""
        status_icons = {
            "pending": "[ ]",
            "running": "[...]",
            "success": "[OK]",
            "error": "[FAIL]",
        }
        icon = status_icons.get(status, "[ ]")
        
        line = f"{icon} {name}"
        if duration is not None:
            line += f" ({duration:.1f}s)"
        
        lines = [line]
        
        if arguments:
            import json
            try:
                args_str = json.dumps(arguments, indent=2)
                for arg_line in args_str.split("\n"):
                    lines.append(f"    {arg_line}")
            except (TypeError, ValueError):
                lines.append(f"    args: {arguments}")
        
        if result:
            # Truncate long results
            preview = result[:200]
            if len(result) > 200:
                preview += "..."
            lines.append(f"    -> {preview}")
        
        return "\n".join(lines)
    
    def render_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cost: float | None = None,
    ) -> str:
        """Render token usage as plain text."""
        total = input_tokens + output_tokens
        line = f"Tokens: {input_tokens} in / {output_tokens} out / {total} total"
        if cost is not None:
            line += f" (${cost:.4f})"
        return line
    
    def render_step_progress(
        self,
        steps: list[tuple[int, str, str]],  # (number, description, status)
        current: int,
        total: int,
    ) -> str:
        """Render step-by-step progress as plain text."""
        lines = []
        lines.append(self.render_progress(current, total, "Plan Execution"))
        lines.append("")
        
        for number, description, status in steps:
            status_icons = {
                "pending": "[ ]",
                "running": "[...]",
                "completed": "[OK]",
                "failed": "[FAIL]",
                "skipped": "[SKIP]",
            }
            icon = status_icons.get(status, "[ ]")
            lines.append(f"  {icon} Step {number}: {description}")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Syntax Highlighting Component
# ═══════════════════════════════════════════════════════════════════════════════


class CodeBlock:
    """
    Syntax-highlighted code block with automatic language detection.
    
    Usage:
        code = CodeBlock("def hello(): pass", language="python")
        console.print(code)
        
        # Or with auto-detection from filename
        code = CodeBlock.from_file("main.py", content)
    """
    
    # Map file extensions to Pygments lexer names
    EXTENSION_MAP: dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".sql": "sql",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "zsh",
        ".fish": "fish",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".kt": "kotlin",
        ".swift": "swift",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
        ".lua": "lua",
        ".r": "r",
        ".R": "r",
        ".ex": "elixir",
        ".exs": "elixir",
        ".erl": "erlang",
        ".hs": "haskell",
        ".ml": "ocaml",
        ".scala": "scala",
        ".clj": "clojure",
        ".vim": "vim",
        ".dockerfile": "dockerfile",
        ".tf": "terraform",
        ".graphql": "graphql",
        ".proto": "protobuf",
        ".xml": "xml",
        ".csv": "text",
        ".txt": "text",
        ".log": "text",
        ".ini": "ini",
        ".cfg": "ini",
        ".conf": "nginx",
        ".env": "bash",
    }
    
    # Patterns for detecting language from content
    CONTENT_PATTERNS: list[tuple[str, str]] = [
        (r"^#!/usr/bin/env python", "python"),
        (r"^#!/usr/bin/python", "python"),
        (r"^#!/bin/bash", "bash"),
        (r"^#!/bin/sh", "bash"),
        (r"^#!/usr/bin/env node", "javascript"),
        (r"^<\?php", "php"),
        (r"^package\s+\w+", "go"),
        (r"^import\s+\w+\s+from", "javascript"),
        (r"^from\s+\w+\s+import", "python"),
        (r"^def\s+\w+\s*\(", "python"),
        (r"^class\s+\w+", "python"),
        (r"^function\s+\w+", "javascript"),
        (r"^const\s+\w+\s*=", "javascript"),
        (r"^let\s+\w+\s*=", "javascript"),
        (r"^var\s+\w+\s*=", "javascript"),
        (r"^\s*fn\s+\w+", "rust"),
        (r"^\s*func\s+\w+", "go"),
        (r"^SELECT\s+", "sql"),
        (r"^CREATE\s+TABLE", "sql"),
        (r"^INSERT\s+INTO", "sql"),
        (r"^\s*<html", "html"),
        (r"^\s*<!DOCTYPE", "html"),
        (r"^\{[\s\n]*\"", "json"),
        (r"^\[[\s\n]*\{", "json"),
    ]
    
    def __init__(
        self,
        code: str,
        *,
        language: str | None = None,
        filename: str | None = None,
        line_numbers: bool = False,
        start_line: int = 1,
        highlight_lines: set[int] | None = None,
        theme: str = "monokai",
        word_wrap: bool = True,
        background_color: str | None = None,
    ):
        self.code = code.rstrip()
        self.language = language or self._detect_language(code, filename)
        self.filename = filename
        self.line_numbers = line_numbers
        self.start_line = start_line
        self.highlight_lines = highlight_lines
        self.theme = theme
        self.word_wrap = word_wrap
        self.background_color = background_color
    
    @classmethod
    def from_file(cls, filename: str, content: str, **kwargs) -> "CodeBlock":
        """Create a CodeBlock from a filename and its content."""
        return cls(content, filename=filename, **kwargs)
    
    def _detect_language(self, code: str, filename: str | None) -> str:
        """Detect language from filename extension or content patterns."""
        # Try filename first
        if filename:
            for ext, lang in self.EXTENSION_MAP.items():
                if filename.endswith(ext):
                    return lang
            # Check for Dockerfile
            if filename.lower() == "dockerfile":
                return "dockerfile"
            if filename.lower() == "makefile":
                return "makefile"
        
        # Try content patterns
        first_lines = "\n".join(code.split("\n")[:5])
        for pattern, lang in self.CONTENT_PATTERNS:
            if re.search(pattern, first_lines, re.IGNORECASE | re.MULTILINE):
                return lang
        
        return "text"
    
    def to_syntax(self) -> Syntax:
        """Convert to Rich Syntax object."""
        return Syntax(
            self.code,
            self.language,
            theme=self.theme,
            line_numbers=self.line_numbers,
            start_line=self.start_line,
            highlight_lines=self.highlight_lines,
            word_wrap=self.word_wrap,
            background_color=self.background_color,
        )
    
    def to_panel(
        self,
        title: str | None = None,
        border_style: str = "bright_black",
    ) -> Panel:
        """Wrap the code in a panel."""
        display_title = title or self.filename or f"[{self.language}]"
        return Panel(
            self.to_syntax(),
            title=display_title,
            title_align="left",
            border_style=border_style,
            box=ROUNDED,
            padding=(0, 1),
        )
    
    def to_plain_text(self) -> str:
        """Render as plain text for non-TTY environments."""
        lines = []
        header = self.filename or f"[{self.language}]"
        separator = "-" * (len(header) + 8)
        lines.append(f"--- {header} ---")
        
        # Add line numbers if requested
        if self.line_numbers:
            code_lines = self.code.split("\n")
            for i, line in enumerate(code_lines, start=self.start_line):
                lines.append(f"{i:4d} | {line}")
        else:
            lines.append(self.code)
        
        lines.append(separator)
        return "\n".join(lines)
    
    def __rich__(self) -> Syntax:
        """Allow direct printing with Rich console."""
        return self.to_syntax()


# ═══════════════════════════════════════════════════════════════════════════════
# Diff Display Component
# ═══════════════════════════════════════════════════════════════════════════════


class DiffLineType(str, Enum):
    """Type of diff line."""
    CONTEXT = "context"
    ADDITION = "addition"
    DELETION = "deletion"
    HEADER = "header"
    HUNK = "hunk"


@dataclass
class DiffLine:
    """A single line in a diff."""
    type: DiffLineType
    content: str
    old_lineno: int | None = None
    new_lineno: int | None = None


class DiffDisplay:
    """
    Beautiful diff display with colors and line numbers.
    
    Supports unified diff format and custom before/after comparison.
    
    Usage:
        # From unified diff string
        diff = DiffDisplay.from_unified_diff(diff_text)
        console.print(diff)
        
        # From before/after content
        diff = DiffDisplay.from_strings(old_content, new_content, filename="main.py")
        console.print(diff)
    """
    
    # Style configuration
    STYLES = {
        DiffLineType.ADDITION: Style(color="green"),
        DiffLineType.DELETION: Style(color="red"),
        DiffLineType.CONTEXT: Style(color="white", dim=True),
        DiffLineType.HEADER: Style(color="cyan", bold=True),
        DiffLineType.HUNK: Style(color="magenta"),
    }
    
    # Background styles for better visibility
    BACKGROUND_STYLES = {
        DiffLineType.ADDITION: Style(color="green", bgcolor="grey11"),
        DiffLineType.DELETION: Style(color="red", bgcolor="grey11"),
    }
    
    def __init__(
        self,
        lines: list[DiffLine],
        *,
        filename: str | None = None,
        show_line_numbers: bool = True,
        use_background: bool = False,
        context_lines: int = 3,
    ):
        self.lines = lines
        self.filename = filename
        self.show_line_numbers = show_line_numbers
        self.use_background = use_background
        self.context_lines = context_lines
    
    @classmethod
    def from_unified_diff(cls, diff_text: str, **kwargs) -> "DiffDisplay":
        """Parse a unified diff string into DiffDisplay."""
        lines: list[DiffLine] = []
        old_lineno = 0
        new_lineno = 0
        filename = None
        
        for line in diff_text.split("\n"):
            if line.startswith("---"):
                # Old file header
                filename = line[4:].split("\t")[0].strip()
                lines.append(DiffLine(DiffLineType.HEADER, line))
            elif line.startswith("+++"):
                # New file header
                if not filename:
                    filename = line[4:].split("\t")[0].strip()
                lines.append(DiffLine(DiffLineType.HEADER, line))
            elif line.startswith("@@"):
                # Hunk header - parse line numbers
                match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                if match:
                    old_lineno = int(match.group(1))
                    new_lineno = int(match.group(2))
                lines.append(DiffLine(DiffLineType.HUNK, line))
            elif line.startswith("+"):
                lines.append(DiffLine(
                    DiffLineType.ADDITION,
                    line[1:],
                    new_lineno=new_lineno,
                ))
                new_lineno += 1
            elif line.startswith("-"):
                lines.append(DiffLine(
                    DiffLineType.DELETION,
                    line[1:],
                    old_lineno=old_lineno,
                ))
                old_lineno += 1
            elif line.startswith(" "):
                lines.append(DiffLine(
                    DiffLineType.CONTEXT,
                    line[1:],
                    old_lineno=old_lineno,
                    new_lineno=new_lineno,
                ))
                old_lineno += 1
                new_lineno += 1
            elif line.strip():
                # Other content (like "\ No newline at end of file")
                lines.append(DiffLine(DiffLineType.CONTEXT, line))
        
        return cls(lines, filename=filename, **kwargs)
    
    @classmethod
    def from_strings(
        cls,
        old: str,
        new: str,
        *,
        filename: str = "file",
        context_lines: int = 3,
        **kwargs,
    ) -> "DiffDisplay":
        """Create a diff from two strings."""
        import difflib
        
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            n=context_lines,
        )
        
        diff_text = "".join(diff)
        return cls.from_unified_diff(diff_text, context_lines=context_lines, **kwargs)
    
    def _get_style(self, line_type: DiffLineType) -> Style:
        """Get style for a line type."""
        if self.use_background and line_type in self.BACKGROUND_STYLES:
            return self.BACKGROUND_STYLES[line_type]
        return self.STYLES.get(line_type, Style())
    
    def _format_line_number(self, lineno: int | None, width: int = 4) -> str:
        """Format a line number with padding."""
        if lineno is None:
            return " " * width
        return str(lineno).rjust(width)
    
    def to_text(self) -> Text:
        """Convert diff to Rich Text object."""
        result = Text()
        
        for line in self.lines:
            style = self._get_style(line.type)
            
            # Add prefix based on type
            if line.type == DiffLineType.ADDITION:
                prefix = "+ "
            elif line.type == DiffLineType.DELETION:
                prefix = "- "
            elif line.type in (DiffLineType.HEADER, DiffLineType.HUNK):
                prefix = ""
            else:
                prefix = "  "
            
            # Add line numbers if enabled
            if self.show_line_numbers and line.type not in (
                DiffLineType.HEADER, DiffLineType.HUNK
            ):
                old_num = self._format_line_number(line.old_lineno)
                new_num = self._format_line_number(line.new_lineno)
                result.append(f"{old_num} {new_num} ", style="dim")
            
            result.append(f"{prefix}{line.content}\n", style=style)
        
        return result
    
    def to_panel(
        self,
        title: str | None = None,
        border_style: str = "bright_black",
    ) -> Panel:
        """Wrap diff in a panel."""
        display_title = title or self.filename or "Diff"
        
        # Add stats to title
        additions = sum(1 for l in self.lines if l.type == DiffLineType.ADDITION)
        deletions = sum(1 for l in self.lines if l.type == DiffLineType.DELETION)
        stats = f" [green]+{additions}[/green] [red]-{deletions}[/red]"
        
        return Panel(
            self.to_text(),
            title=f"{display_title}{stats}",
            title_align="left",
            border_style=border_style,
            box=ROUNDED,
            padding=(0, 1),
        )
    
    def to_plain_text(self) -> str:
        """Render as plain text for non-TTY environments."""
        lines = []
        stats = self.stats
        header = self.filename or "diff"
        lines.append(f"--- {header} (+{stats['additions']} -{stats['deletions']}) ---")
        
        for line in self.lines:
            # Add prefix based on type
            if line.type == DiffLineType.ADDITION:
                prefix = "+ "
            elif line.type == DiffLineType.DELETION:
                prefix = "- "
            elif line.type in (DiffLineType.HEADER, DiffLineType.HUNK):
                prefix = ""
            else:
                prefix = "  "
            
            # Add line numbers if enabled
            if self.show_line_numbers and line.type not in (
                DiffLineType.HEADER, DiffLineType.HUNK
            ):
                old_num = self._format_line_number(line.old_lineno)
                new_num = self._format_line_number(line.new_lineno)
                lines.append(f"{old_num} {new_num} {prefix}{line.content}")
            else:
                lines.append(f"{prefix}{line.content}")
        
        return "\n".join(lines)
    
    @property
    def stats(self) -> dict[str, int]:
        """Get diff statistics."""
        return {
            "additions": sum(1 for l in self.lines if l.type == DiffLineType.ADDITION),
            "deletions": sum(1 for l in self.lines if l.type == DiffLineType.DELETION),
            "context": sum(1 for l in self.lines if l.type == DiffLineType.CONTEXT),
        }
    
    def __rich__(self) -> Text:
        """Allow direct printing with Rich console."""
        return self.to_text()


# ═══════════════════════════════════════════════════════════════════════════════
# Collapsible Section Component
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CollapsibleSection:
    """
    A collapsible/expandable section for verbose output.
    
    In terminal output, shows either:
    - Collapsed: Just the title with indicator (▶ Section Title)
    - Expanded: Full content in a panel
    
    Usage:
        section = CollapsibleSection(
            title="Verbose Output",
            content="Long detailed content...",
            collapsed=True,
        )
        console.print(section)
        
        # Toggle state
        section.toggle()
        console.print(section)
    """
    
    title: str
    content: Any  # Can be str, Text, or any Rich renderable
    collapsed: bool = True
    icon_collapsed: str = "▶"
    icon_expanded: str = "▼"
    border_style: str = "bright_black"
    title_style: str = "bold cyan"
    preview_length: int = 80
    show_preview: bool = True
    
    def toggle(self) -> None:
        """Toggle collapsed state."""
        self.collapsed = not self.collapsed
    
    def expand(self) -> None:
        """Expand the section."""
        self.collapsed = False
    
    def collapse(self) -> None:
        """Collapse the section."""
        self.collapsed = True
    
    def _get_preview(self) -> str:
        """Get a preview of the content when collapsed."""
        if isinstance(self.content, str):
            text = self.content
        elif isinstance(self.content, Text):
            text = self.content.plain
        else:
            text = str(self.content)
        
        # Get first line or truncate
        first_line = text.split("\n")[0]
        if len(first_line) > self.preview_length:
            return first_line[:self.preview_length] + "..."
        if "\n" in text:
            return first_line + " ..."
        return first_line
    
    def to_collapsed(self) -> Text:
        """Render collapsed view."""
        result = Text()
        result.append(f"{self.icon_collapsed} ", style=self.title_style)
        result.append(self.title, style=self.title_style)
        
        if self.show_preview:
            preview = self._get_preview()
            result.append(f" — {preview}", style="dim")
        
        return result
    
    def to_expanded(self) -> Panel:
        """Render expanded view."""
        # Wrap content appropriately
        if isinstance(self.content, str):
            renderable = Text(self.content)
        else:
            renderable = self.content
        
        title = f"{self.icon_expanded} {self.title}"
        
        return Panel(
            renderable,
            title=f"[{self.title_style}]{title}[/{self.title_style}]",
            title_align="left",
            border_style=self.border_style,
            box=ROUNDED,
            padding=(0, 1),
        )
    
    def __rich__(self) -> Text | Panel:
        """Allow direct printing with Rich console."""
        if self.collapsed:
            return self.to_collapsed()
        return self.to_expanded()


@dataclass
class CollapsibleGroup:
    """
    A group of collapsible sections with coordinated behavior.
    
    Usage:
        group = CollapsibleGroup()
        group.add("Details", detailed_content)
        group.add("Logs", log_content, collapsed=False)
        console.print(group)
        
        # Expand all / collapse all
        group.expand_all()
        group.collapse_all()
    """
    
    sections: list[CollapsibleSection] = field(default_factory=list)
    accordion: bool = False  # If True, only one section can be expanded at a time
    
    def add(
        self,
        title: str,
        content: Any,
        *,
        collapsed: bool = True,
        **kwargs,
    ) -> CollapsibleSection:
        """Add a section to the group."""
        section = CollapsibleSection(
            title=title,
            content=content,
            collapsed=collapsed,
            **kwargs,
        )
        self.sections.append(section)
        
        # Accordion mode: collapse others if this one is expanded
        if self.accordion and not collapsed:
            for other in self.sections[:-1]:
                other.collapse()
        
        return section
    
    def expand(self, index: int) -> None:
        """Expand a specific section by index."""
        if 0 <= index < len(self.sections):
            if self.accordion:
                self.collapse_all()
            self.sections[index].expand()
    
    def collapse(self, index: int) -> None:
        """Collapse a specific section by index."""
        if 0 <= index < len(self.sections):
            self.sections[index].collapse()
    
    def toggle(self, index: int) -> None:
        """Toggle a specific section by index."""
        if 0 <= index < len(self.sections):
            if self.accordion and self.sections[index].collapsed:
                self.collapse_all()
            self.sections[index].toggle()
    
    def expand_all(self) -> None:
        """Expand all sections."""
        for section in self.sections:
            section.expand()
    
    def collapse_all(self) -> None:
        """Collapse all sections."""
        for section in self.sections:
            section.collapse()
    
    def __rich__(self) -> Group:
        """Allow direct printing with Rich console."""
        return Group(*self.sections)


# ═══════════════════════════════════════════════════════════════════════════════
# Token Usage Display Component
# ═══════════════════════════════════════════════════════════════════════════════


# Model pricing per 1M tokens (input, output) in USD
# Prices are approximate and may change
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # GPT models
    "gpt-5.2-codex": (5.00, 15.00),
    "gpt-5.1-codex": (3.00, 12.00),
    "gpt-5.1-codex-max": (10.00, 30.00),
    "gpt-5.1-codex-mini": (0.50, 1.50),
    "gpt-5.2": (5.00, 15.00),
    "gpt-5.1": (3.00, 12.00),
    "gpt-5": (2.50, 10.00),
    "gpt-5-mini": (0.30, 1.00),
    "gpt-4.1": (2.00, 8.00),
    # Claude models
    "claude-sonnet-4.5": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4.5": (0.80, 4.00),
    "claude-opus-4.5": (15.00, 75.00),
    # Gemini models
    "gemini-3-pro-preview": (1.25, 5.00),
}

# Default pricing for unknown models
DEFAULT_PRICING: tuple[float, float] = (2.00, 8.00)


@dataclass
class TokenUsageDisplay:
    """
    Display component for token usage with cost calculation.
    
    Shows input/output/total tokens and calculates cost based on model pricing.
    
    Usage:
        usage = TokenUsageDisplay(
            input_tokens=1500,
            output_tokens=500,
            model="claude-sonnet-4.5",
        )
        console.print(usage)
        
        # Or from TokenUsage object
        usage = TokenUsageDisplay.from_token_usage(token_usage, model="gpt-5")
    """
    
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None
    show_cost: bool = True
    compact: bool = False
    
    # Icons
    icon_input: str = "→"
    icon_output: str = "←"
    icon_total: str = "Σ"
    icon_cost: str = "$"
    
    @classmethod
    def from_token_usage(
        cls,
        usage: Any,  # TokenUsage from models.py
        *,
        model: str | None = None,
        **kwargs,
    ) -> "TokenUsageDisplay":
        """Create from a TokenUsage object."""
        input_tokens = usage.prompt or 0
        output_tokens = usage.completion or 0
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            **kwargs,
        )
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens
    
    @property
    def cost(self) -> float | None:
        """Calculate cost in USD based on model pricing."""
        if not self.model:
            return None
        
        pricing = MODEL_PRICING.get(self.model, DEFAULT_PRICING)
        input_price, output_price = pricing
        
        # Pricing is per 1M tokens
        input_cost = (self.input_tokens / 1_000_000) * input_price
        output_cost = (self.output_tokens / 1_000_000) * output_price
        
        return input_cost + output_cost
    
    @staticmethod
    def format_tokens(count: int) -> str:
        """Format token count with K/M suffix for large numbers."""
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        if count >= 10_000:
            return f"{count / 1_000:.1f}K"
        if count >= 1_000:
            return f"{count / 1_000:.2f}K"
        return str(count)
    
    @staticmethod
    def format_cost(cost: float) -> str:
        """Format cost in USD."""
        if cost < 0.0001:
            return "<$0.0001"
        if cost < 0.01:
            return f"${cost:.4f}"
        if cost < 1.00:
            return f"${cost:.3f}"
        return f"${cost:.2f}"
    
    def to_text(self) -> Text:
        """Convert to Rich Text object."""
        result = Text()
        
        if self.compact:
            # Compact format: "1.2K → 0.5K ← (1.7K) $0.012"
            result.append(self.format_tokens(self.input_tokens), style="cyan")
            result.append(f" {self.icon_input} ", style="dim")
            result.append(self.format_tokens(self.output_tokens), style="green")
            result.append(f" {self.icon_output} ", style="dim")
            result.append("(", style="dim")
            result.append(self.format_tokens(self.total_tokens), style="bold white")
            result.append(")", style="dim")
            
            if self.show_cost and self.cost is not None:
                result.append(" ", style="dim")
                result.append(self.format_cost(self.cost), style="yellow")
        else:
            # Full format with labels
            result.append(f"{self.icon_input} Input: ", style="dim")
            result.append(self.format_tokens(self.input_tokens), style="cyan")
            result.append(" │ ", style="dim")
            result.append(f"{self.icon_output} Output: ", style="dim")
            result.append(self.format_tokens(self.output_tokens), style="green")
            result.append(" │ ", style="dim")
            result.append(f"{self.icon_total} Total: ", style="dim")
            result.append(self.format_tokens(self.total_tokens), style="bold white")
            
            if self.show_cost and self.cost is not None:
                result.append(" │ ", style="dim")
                result.append(f"{self.icon_cost} Cost: ", style="dim")
                result.append(self.format_cost(self.cost), style="yellow")
        
        return result
    
    def to_panel(
        self,
        title: str = "Token Usage",
        border_style: str = "bright_black",
    ) -> Panel:
        """Wrap in a panel."""
        return Panel(
            self.to_text(),
            title=title,
            title_align="left",
            border_style=border_style,
            box=ROUNDED,
            padding=(0, 1),
        )
    
    def to_plain_text(self) -> str:
        """Render as plain text for non-TTY environments."""
        parts = [
            f"Tokens: {self.format_tokens(self.input_tokens)} in",
            f"{self.format_tokens(self.output_tokens)} out",
            f"{self.format_tokens(self.total_tokens)} total",
        ]
        if self.show_cost and self.cost is not None:
            parts.append(f"({self.format_cost(self.cost)})")
        return " / ".join(parts[:3]) + (" " + parts[3] if len(parts) > 3 else "")
    
    def __rich__(self) -> Text:
        """Allow direct printing with Rich console."""
        return self.to_text()


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Call Panel Component
# ═══════════════════════════════════════════════════════════════════════════════


class ToolStatus(str, Enum):
    """Status of a tool call."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ToolCallPanel:
    """
    Enhanced tool call display with syntax highlighting and collapsible results.
    
    Features:
    - Tool name with appropriate icon
    - Arguments displayed with JSON syntax highlighting
    - Results in collapsible format
    - Status icons (spinner, checkmark, X)
    - Duration tracking
    
    Usage:
        panel = ToolCallPanel(
            name="bash",
            arguments={"command": "ls -la"},
            status=ToolStatus.SUCCESS,
            result="file1.txt\\nfile2.txt",
            duration=1.5,
        )
        console.print(panel)
    """
    
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    status: ToolStatus = ToolStatus.PENDING
    duration: float | None = None
    error: str | None = None
    collapsed: bool = True
    
    # Icons
    TOOL_ICONS: dict[str, str] = field(default_factory=lambda: {
        "bash": "💻",
        "shell": "💻",
        "powershell": "💻",
        "read": "📖",
        "view": "📖",
        "write": "📝",
        "edit": "📝",
        "create": "📄",
        "search": "🔍",
        "grep": "🔍",
        "glob": "🔍",
        "find": "🔍",
        "web": "🌐",
        "fetch": "🌐",
        "http": "🌐",
        "api": "🌐",
        "git": "📦",
        "npm": "📦",
        "pip": "📦",
        "default": "⚡",
    })
    
    STATUS_ICONS: dict[ToolStatus, str] = field(default_factory=lambda: {
        ToolStatus.PENDING: "○",
        ToolStatus.RUNNING: "⠋",
        ToolStatus.SUCCESS: "✓",
        ToolStatus.ERROR: "✗",
        ToolStatus.CANCELLED: "⊘",
    })
    
    STATUS_STYLES: dict[ToolStatus, str] = field(default_factory=lambda: {
        ToolStatus.PENDING: "dim",
        ToolStatus.RUNNING: "yellow",
        ToolStatus.SUCCESS: "green",
        ToolStatus.ERROR: "red",
        ToolStatus.CANCELLED: "dim",
    })
    
    def get_tool_icon(self) -> str:
        """Get icon for this tool based on name."""
        name_lower = self.name.lower()
        for key, icon in self.TOOL_ICONS.items():
            if key in name_lower:
                return icon
        return self.TOOL_ICONS["default"]
    
    def get_status_icon(self) -> str:
        """Get status icon."""
        return self.STATUS_ICONS.get(self.status, "○")
    
    def get_status_style(self) -> str:
        """Get style for current status."""
        return self.STATUS_STYLES.get(self.status, "dim")
    
    def toggle_collapsed(self) -> None:
        """Toggle collapsed state."""
        self.collapsed = not self.collapsed
    
    def _format_arguments(self) -> Syntax | Text:
        """Format arguments with JSON syntax highlighting."""
        if not self.arguments:
            return Text("(no arguments)", style="dim italic")
        
        import json
        try:
            formatted = json.dumps(self.arguments, indent=2, ensure_ascii=False)
            return Syntax(
                formatted,
                "json",
                theme="monokai",
                word_wrap=True,
                background_color=None,
            )
        except (TypeError, ValueError):
            # Fallback for non-JSON-serializable arguments
            return Text(str(self.arguments), style="dim")
    
    def _format_result_preview(self, max_length: int = 100) -> str:
        """Get a preview of the result."""
        if not self.result:
            return "(no output)"
        
        # Get first line or truncate
        first_line = self.result.split("\n")[0]
        if len(first_line) > max_length:
            return first_line[:max_length] + "..."
        if "\n" in self.result:
            return first_line + f" (+{self.result.count(chr(10))} lines)"
        return first_line
    
    def _format_result_full(self) -> Text | Syntax:
        """Format the full result with appropriate highlighting."""
        if not self.result:
            return Text("(no output)", style="dim italic")
        
        result = self.result
        
        # Try to detect if result looks like code/structured output
        if result.strip().startswith("{") or result.strip().startswith("["):
            try:
                import json
                json.loads(result)
                return Syntax(result, "json", theme="monokai", word_wrap=True)
            except json.JSONDecodeError:
                pass
        
        # Check for diff output
        if result.startswith("---") or result.startswith("@@"):
            return Syntax(result, "diff", theme="monokai", word_wrap=True)
        
        return Text(result)
    
    def _build_header(self) -> Text:
        """Build the tool call header line."""
        header = Text()
        
        # Status icon
        status_style = self.get_status_style()
        header.append(f"{self.get_status_icon()} ", style=f"bold {status_style}")
        
        # Tool icon and name
        header.append(f"{self.get_tool_icon()} ", style=status_style)
        header.append(self.name, style=f"bold {status_style}")
        
        # Duration
        if self.duration is not None:
            header.append(f" ({self.duration:.1f}s)", style="dim")
        elif self.status == ToolStatus.RUNNING:
            header.append(" (running...)", style="dim italic")
        
        return header
    
    def to_compact(self) -> Text:
        """Render compact single-line view."""
        line = self._build_header()
        
        # Add key argument preview
        if self.arguments:
            for key in ("path", "command", "pattern", "query", "file", "url"):
                if key in self.arguments:
                    val = str(self.arguments[key])[:40]
                    if len(str(self.arguments[key])) > 40:
                        val += "..."
                    line.append(f" {key}=", style="dim")
                    line.append(val, style="dim italic")
                    break
        
        return line
    
    def to_panel(self, show_arguments: bool = True, show_result: bool = True) -> Panel:
        """Render as a full panel with details."""
        elements = []
        
        # Header
        elements.append(self._build_header())
        elements.append(Text())
        
        # Arguments
        if show_arguments and self.arguments:
            elements.append(Text("Arguments:", style="bold dim"))
            elements.append(self._format_arguments())
            elements.append(Text())
        
        # Result or error
        if show_result:
            if self.error:
                elements.append(Text("Error:", style="bold red"))
                elements.append(Text(self.error, style="red"))
            elif self.result is not None:
                if self.collapsed:
                    elements.append(Text("Result: ", style="bold dim", end=""))
                    elements.append(Text(self._format_result_preview(), style="dim"))
                else:
                    elements.append(Text("Result:", style="bold dim"))
                    elements.append(self._format_result_full())
        
        # Build panel
        status_style = self.get_status_style()
        border_style = "bright_black"
        if self.status == ToolStatus.RUNNING:
            border_style = "yellow"
        elif self.status == ToolStatus.ERROR:
            border_style = "red"
        elif self.status == ToolStatus.SUCCESS:
            border_style = "green"
        
        title = f"[{status_style}]{self.get_tool_icon()} {self.name}[/{status_style}]"
        
        return Panel(
            Group(*elements),
            title=title,
            title_align="left",
            border_style=border_style,
            box=ROUNDED,
            padding=(0, 1),
        )
    
    def to_plain_text(self) -> str:
        """Render as plain text for non-TTY environments."""
        status_icons = {
            ToolStatus.PENDING: "[ ]",
            ToolStatus.RUNNING: "[...]",
            ToolStatus.SUCCESS: "[OK]",
            ToolStatus.ERROR: "[FAIL]",
            ToolStatus.CANCELLED: "[SKIP]",
        }
        icon = status_icons.get(self.status, "[ ]")
        
        lines = []
        line = f"{icon} {self.name}"
        if self.duration is not None:
            line += f" ({self.duration:.1f}s)"
        lines.append(line)
        
        if self.arguments:
            import json
            try:
                args_str = json.dumps(self.arguments, indent=2)
                for arg_line in args_str.split("\n"):
                    lines.append(f"    {arg_line}")
            except (TypeError, ValueError):
                lines.append(f"    args: {self.arguments}")
        
        if self.error:
            lines.append(f"    ERROR: {self.error}")
        elif self.result:
            preview = self._format_result_preview()
            lines.append(f"    -> {preview}")
        
        return "\n".join(lines)
    
    def __rich__(self) -> Panel:
        """Allow direct printing with Rich console."""
        return self.to_panel()


@dataclass
class ToolCallGroup:
    """
    Group of tool calls with summary header.
    
    Usage:
        group = ToolCallGroup()
        group.add("bash", {"command": "ls"})
        group.complete("bash_1", result="files...")
        console.print(group)
    """
    
    calls: list[ToolCallPanel] = field(default_factory=list)
    max_visible: int = 5
    show_all: bool = False
    compact: bool = True
    
    def add(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        call_id: str | None = None,
    ) -> ToolCallPanel:
        """Add a new tool call."""
        panel = ToolCallPanel(
            name=name,
            arguments=arguments or {},
            status=ToolStatus.RUNNING,
        )
        self.calls.append(panel)
        return panel
    
    def complete(
        self,
        index: int = -1,
        *,
        result: str | None = None,
        error: str | None = None,
        duration: float | None = None,
    ) -> None:
        """Mark a tool call as complete."""
        if not self.calls:
            return
        
        panel = self.calls[index]
        panel.result = result
        panel.error = error
        panel.duration = duration
        panel.status = ToolStatus.ERROR if error else ToolStatus.SUCCESS
    
    def get_stats(self) -> dict[str, int]:
        """Get status counts."""
        stats = {"running": 0, "success": 0, "error": 0, "pending": 0}
        for call in self.calls:
            if call.status == ToolStatus.RUNNING:
                stats["running"] += 1
            elif call.status == ToolStatus.SUCCESS:
                stats["success"] += 1
            elif call.status == ToolStatus.ERROR:
                stats["error"] += 1
            else:
                stats["pending"] += 1
        return stats
    
    def _build_summary(self) -> Text:
        """Build summary line."""
        stats = self.get_stats()
        summary = Text()
        summary.append("⚡ Tools: ", style="bold yellow")
        
        parts = []
        if stats["running"]:
            parts.append(f"[yellow]{stats['running']} running[/yellow]")
        if stats["success"]:
            parts.append(f"[green]{stats['success']} ok[/green]")
        if stats["error"]:
            parts.append(f"[red]{stats['error']} failed[/red]")
        
        summary.append(Text.from_markup(" • ".join(parts)))
        return summary
    
    def to_panel(self) -> Panel | None:
        """Render as a panel."""
        if not self.calls:
            return None
        
        from rich.tree import Tree
        
        stats = self.get_stats()
        
        # Build tree
        tree = Tree("")
        visible_calls = self.calls if self.show_all else self.calls[-self.max_visible:]
        
        for call in visible_calls:
            if self.compact:
                tree.add(call.to_compact())
            else:
                # For non-compact, show more details
                branch = tree.add(call.to_compact())
                if call.result and call.status != ToolStatus.RUNNING:
                    preview = call._format_result_preview(60)
                    branch.add(Text(preview, style="dim"))
        
        # Show count of hidden calls
        hidden = len(self.calls) - len(visible_calls)
        if hidden > 0:
            tree.add(Text(f"... +{hidden} more", style="dim italic"))
        
        # Determine border style
        border_style = "bright_black"
        if stats["running"]:
            border_style = "yellow"
        elif stats["error"]:
            border_style = "red"
        
        # Build title
        title_parts = ["⚡ Tools"]
        if stats["running"]:
            title_parts.append(f"{stats['running']} running")
        if stats["success"]:
            title_parts.append(f"{stats['success']} ok")
        if stats["error"]:
            title_parts.append(f"{stats['error']} failed")
        title = f"[yellow]{' • '.join(title_parts)}[/yellow]"
        
        return Panel(
            tree,
            title=title,
            title_align="left",
            border_style=border_style,
            box=ROUNDED,
            padding=(0, 1),
        )
    
    def __rich__(self) -> Panel | Text:
        """Allow direct printing with Rich console."""
        panel = self.to_panel()
        return panel if panel else Text("(no tool calls)", style="dim")


# ═══════════════════════════════════════════════════════════════════════════════
# Step Progress Component
# ═══════════════════════════════════════════════════════════════════════════════


class StepStatus(str, Enum):
    """Status of a plan step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepInfo:
    """Information about a single step."""
    number: int
    description: str
    status: StepStatus = StepStatus.PENDING
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    result: str | None = None
    
    @property
    def duration(self) -> float | None:
        """Get duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at
    
    @property
    def duration_str(self) -> str:
        """Format duration as string."""
        duration = self.duration
        if duration is None:
            return ""
        if duration < 60:
            return f"{duration:.1f}s"
        minutes = int(duration // 60)
        seconds = duration % 60
        return f"{minutes}m {seconds:.0f}s"


@dataclass
class StepProgress:
    """
    Progress display for multi-step plan execution.
    
    Features:
    - "Step X of Y" header with progress bar
    - Per-step status icons (pending, running, completed, failed)
    - ETA estimation
    - Duration tracking per step
    - Collapsible step details
    
    Usage:
        progress = StepProgress(total=5, title="Plan Execution")
        progress.add_step(1, "Initialize project")
        progress.add_step(2, "Install dependencies")
        progress.start_step(1)
        progress.complete_step(1)
        console.print(progress)
    """
    
    total: int = 0
    title: str = "Progress"
    steps: list[StepInfo] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    show_eta: bool = True
    show_steps: bool = True
    max_visible_steps: int = 10
    compact: bool = False
    
    # Spinner frames for running status
    _spinner_frames: list[str] = field(default_factory=lambda: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    _spinner_idx: int = 0
    
    STATUS_ICONS: dict[StepStatus, str] = field(default_factory=lambda: {
        StepStatus.PENDING: "○",
        StepStatus.RUNNING: "⠋",
        StepStatus.COMPLETED: "✓",
        StepStatus.FAILED: "✗",
        StepStatus.SKIPPED: "⊘",
    })
    
    STATUS_STYLES: dict[StepStatus, str] = field(default_factory=lambda: {
        StepStatus.PENDING: "dim",
        StepStatus.RUNNING: "yellow bold",
        StepStatus.COMPLETED: "green",
        StepStatus.FAILED: "red",
        StepStatus.SKIPPED: "dim italic",
    })
    
    def add_step(self, number: int, description: str) -> StepInfo:
        """Add a step to track."""
        step = StepInfo(number=number, description=description)
        self.steps.append(step)
        if self.total < len(self.steps):
            self.total = len(self.steps)
        return step
    
    def get_step(self, number: int) -> StepInfo | None:
        """Get step by number."""
        for step in self.steps:
            if step.number == number:
                return step
        return None
    
    def start_step(self, number: int) -> None:
        """Mark a step as running."""
        step = self.get_step(number)
        if step:
            step.status = StepStatus.RUNNING
            step.started_at = time.time()
    
    def complete_step(self, number: int, result: str | None = None) -> None:
        """Mark a step as completed."""
        step = self.get_step(number)
        if step:
            step.status = StepStatus.COMPLETED
            step.completed_at = time.time()
            step.result = result
    
    def fail_step(self, number: int, error: str) -> None:
        """Mark a step as failed."""
        step = self.get_step(number)
        if step:
            step.status = StepStatus.FAILED
            step.completed_at = time.time()
            step.error = error
    
    def skip_step(self, number: int, reason: str | None = None) -> None:
        """Mark a step as skipped."""
        step = self.get_step(number)
        if step:
            step.status = StepStatus.SKIPPED
            step.completed_at = time.time()
            if reason:
                step.result = reason
    
    def _advance_spinner(self) -> None:
        """Advance spinner animation."""
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
    
    def _get_spinner(self) -> str:
        """Get current spinner frame."""
        return self._spinner_frames[self._spinner_idx]
    
    @property
    def completed_count(self) -> int:
        """Count of completed steps."""
        return sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
    
    @property
    def failed_count(self) -> int:
        """Count of failed steps."""
        return sum(1 for s in self.steps if s.status == StepStatus.FAILED)
    
    @property
    def running_count(self) -> int:
        """Count of running steps."""
        return sum(1 for s in self.steps if s.status == StepStatus.RUNNING)
    
    @property
    def current_step(self) -> StepInfo | None:
        """Get the currently running step."""
        for step in self.steps:
            if step.status == StepStatus.RUNNING:
                return step
        return None
    
    @property
    def percent(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.completed_count / self.total) * 100
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.started_at
    
    @property
    def elapsed_str(self) -> str:
        """Format elapsed time."""
        elapsed = self.elapsed
        if elapsed < 60:
            return f"{elapsed:.0f}s"
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        if minutes < 60:
            return f"{minutes}m {seconds:02d}s"
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours}h {mins:02d}m"
    
    @property
    def eta_seconds(self) -> float | None:
        """Estimate time remaining in seconds."""
        if self.completed_count == 0:
            return None
        avg_time = self.elapsed / self.completed_count
        remaining = self.total - self.completed_count
        return avg_time * remaining
    
    @property
    def eta_str(self) -> str:
        """Format ETA as string."""
        eta = self.eta_seconds
        if eta is None:
            return "calculating..."
        if eta < 60:
            return f"~{eta:.0f}s"
        minutes = int(eta // 60)
        if minutes < 60:
            return f"~{minutes}m"
        hours = int(minutes // 60)
        return f"~{hours}h {minutes % 60}m"
    
    def _build_progress_bar(self, width: int = 30) -> Text:
        """Build a visual progress bar."""
        filled = int(width * self.percent / 100)
        bar = Text()
        bar.append("│", style="dim")
        bar.append("█" * filled, style="green")
        bar.append("░" * (width - filled), style="dim")
        bar.append("│", style="dim")
        return bar
    
    def _build_header(self) -> Text:
        """Build the progress header line."""
        self._advance_spinner()
        header = Text()
        
        # Title and step count
        current = self.current_step
        if current:
            header.append(f"{self._get_spinner()} ", style="yellow bold")
            header.append(f"Step {current.number} of {self.total}", style="bold")
            header.append(f" — {current.description}", style="white")
        elif self.completed_count == self.total and self.total > 0:
            header.append("✓ ", style="green bold")
            header.append(f"All {self.total} steps completed", style="green")
        elif self.failed_count > 0:
            header.append("✗ ", style="red bold")
            header.append(f"{self.failed_count} step(s) failed", style="red")
        else:
            header.append("○ ", style="dim")
            header.append(f"0 of {self.total} steps", style="dim")
        
        return header
    
    def _build_stats_line(self) -> Text:
        """Build the stats line with progress bar and ETA."""
        stats = Text()
        
        # Progress bar
        stats.append_text(self._build_progress_bar())
        stats.append(" ", style="default")
        
        # Percentage
        stats.append(f"{self.percent:.0f}%", style="bold")
        
        # Elapsed
        stats.append(f" • {self.elapsed_str} elapsed", style="dim")
        
        # ETA
        if self.show_eta and self.running_count > 0:
            stats.append(f" • ETA: {self.eta_str}", style="dim")
        
        # Counts
        if self.failed_count > 0:
            stats.append(f" • ", style="dim")
            stats.append(f"{self.failed_count} failed", style="red")
        
        return stats
    
    def _build_step_line(self, step: StepInfo) -> Text:
        """Build a single step line."""
        line = Text()
        
        # Status icon
        if step.status == StepStatus.RUNNING:
            icon = self._get_spinner()
        else:
            icon = self.STATUS_ICONS.get(step.status, "○")
        style = self.STATUS_STYLES.get(step.status, "dim")
        
        line.append(f"  {icon} ", style=style)
        
        # Step number and description
        line.append(f"Step {step.number}: ", style=style)
        line.append(step.description, style=style if step.status != StepStatus.RUNNING else "white")
        
        # Duration
        if step.duration is not None:
            line.append(f" ({step.duration_str})", style="dim")
        elif step.status == StepStatus.RUNNING:
            elapsed = time.time() - (step.started_at or time.time())
            line.append(f" ({elapsed:.0f}s...)", style="dim italic")
        
        # Error
        if step.error:
            line.append(f" — {step.error[:50]}", style="red dim")
        
        return line
    
    def to_text(self) -> Text:
        """Render as plain text (compact mode)."""
        text = Text()
        text.append_text(self._build_header())
        text.append("\n")
        text.append_text(self._build_stats_line())
        return text
    
    def to_panel(self) -> Panel:
        """Render as a panel with step details."""
        elements = []
        
        # Header
        elements.append(self._build_header())
        elements.append(Text())
        
        # Stats line with progress bar
        elements.append(self._build_stats_line())
        
        # Step list
        if self.show_steps and self.steps:
            elements.append(Text())
            
            # Determine which steps to show
            if len(self.steps) <= self.max_visible_steps:
                visible_steps = self.steps
            else:
                # Show running step and surrounding context
                current_idx = None
                for i, step in enumerate(self.steps):
                    if step.status == StepStatus.RUNNING:
                        current_idx = i
                        break
                
                if current_idx is not None:
                    # Show context around current step
                    start = max(0, current_idx - 2)
                    end = min(len(self.steps), current_idx + 3)
                    visible_steps = self.steps[start:end]
                else:
                    # Show last N steps
                    visible_steps = self.steps[-self.max_visible_steps:]
            
            for step in visible_steps:
                elements.append(self._build_step_line(step))
            
            # Show hidden count
            hidden = len(self.steps) - len(visible_steps)
            if hidden > 0:
                elements.append(Text(f"  ... +{hidden} more steps", style="dim italic"))
        
        # Build panel
        border_style = "bright_black"
        if self.running_count > 0:
            border_style = "yellow"
        elif self.failed_count > 0:
            border_style = "red"
        elif self.completed_count == self.total and self.total > 0:
            border_style = "green"
        
        return Panel(
            Group(*elements),
            title=f"[cyan]{self.title}[/cyan]",
            title_align="left",
            border_style=border_style,
            box=ROUNDED,
            padding=(0, 1),
        )
    
    def to_plain_text(self) -> str:
        """Render as plain text for non-TTY environments."""
        lines = []
        
        # Header
        current = self.current_step
        if current:
            lines.append(f"[...] Step {current.number} of {self.total}: {current.description}")
        elif self.completed_count == self.total and self.total > 0:
            lines.append(f"[OK] All {self.total} steps completed")
        elif self.failed_count > 0:
            lines.append(f"[FAIL] {self.failed_count} step(s) failed")
        else:
            lines.append(f"[ ] 0 of {self.total} steps")
        
        # Progress bar
        bar_width = 30
        filled = int(bar_width * self.percent / 100)
        bar = "#" * filled + "-" * (bar_width - filled)
        progress_line = f"[{bar}] {self.percent:.0f}% | {self.elapsed_str} elapsed"
        if self.running_count > 0 and self.eta_seconds is not None:
            progress_line += f" | ETA: {self.eta_str}"
        lines.append(progress_line)
        
        # Steps
        if self.show_steps and self.steps:
            lines.append("")
            status_icons = {
                StepStatus.PENDING: "[ ]",
                StepStatus.RUNNING: "[...]",
                StepStatus.COMPLETED: "[OK]",
                StepStatus.FAILED: "[FAIL]",
                StepStatus.SKIPPED: "[SKIP]",
            }
            for step in self.steps:
                icon = status_icons.get(step.status, "[ ]")
                step_line = f"  {icon} Step {step.number}: {step.description}"
                if step.duration is not None:
                    step_line += f" ({step.duration_str})"
                lines.append(step_line)
        
        return "\n".join(lines)
    
    def __rich__(self) -> Panel | Text:
        """Allow direct printing with Rich console."""
        if self.compact:
            return self.to_text()
        return self.to_panel()


class RichProgressReporter:
    """
    Rich-based progress reporter that integrates with ProgressReporter.
    
    Provides a bridge between the base ProgressReporter and Rich display.
    
    Usage:
        from copex.progress import ProgressReporter
        
        reporter = ProgressReporter(total=5)
        rich_reporter = RichProgressReporter(reporter)
        
        with Live(rich_reporter, refresh_per_second=10):
            for i in range(5):
                reporter.start_item(i, f"Step {i+1}")
                # do work
                reporter.complete_item(i)
    """
    
    def __init__(
        self,
        state: Any = None,  # ProgressState
        *,
        title: str = "Progress",
        show_steps: bool = True,
    ):
        self.progress = StepProgress(title=title, show_steps=show_steps)
        if state:
            self.sync_state(state)
    
    def sync_state(self, state: Any) -> None:
        """Sync with a ProgressState object."""
        self.progress.total = state.total
        self.progress.started_at = state.started_at
        
        # Sync items
        self.progress.steps = []
        for item in state.items:
            step = StepInfo(
                number=item.id if isinstance(item.id, int) else len(self.progress.steps) + 1,
                description=item.description,
                started_at=item.started_at,
                completed_at=item.completed_at,
                error=item.error,
            )
            # Map status
            if item.status.value == "running":
                step.status = StepStatus.RUNNING
            elif item.status.value == "completed":
                step.status = StepStatus.COMPLETED
            elif item.status.value == "failed":
                step.status = StepStatus.FAILED
            elif item.status.value == "skipped":
                step.status = StepStatus.SKIPPED
            else:
                step.status = StepStatus.PENDING
            
            self.progress.steps.append(step)
    
    def __rich__(self) -> Panel | Text:
        """Allow direct printing with Rich console."""
        return self.progress.__rich__()


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════


def extract_code_blocks(text: str) -> list[tuple[str, str, str]]:
    """
    Extract fenced code blocks from markdown text.
    
    Returns list of (language, code, full_match) tuples.
    """
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    results = []
    for lang, code in matches:
        full_match = f"```{lang}\n{code}```"
        results.append((lang or "text", code.strip(), full_match))
    
    return results


def render_markdown_with_syntax(
    text: str,
    *,
    theme: str = "monokai",
    line_numbers: bool = False,
) -> Group:
    """
    Render markdown text with syntax-highlighted code blocks.
    
    Replaces fenced code blocks with Rich Syntax objects.
    """
    from rich.markdown import Markdown
    
    elements: list[Any] = []
    last_end = 0
    
    pattern = r"```(\w*)\n(.*?)```"
    for match in re.finditer(pattern, text, re.DOTALL):
        # Add text before code block
        if match.start() > last_end:
            before_text = text[last_end:match.start()]
            if before_text.strip():
                elements.append(Markdown(before_text))
        
        # Add syntax-highlighted code block
        lang = match.group(1) or "text"
        code = match.group(2).strip()
        code_block = CodeBlock(
            code,
            language=lang,
            line_numbers=line_numbers,
            theme=theme,
        )
        elements.append(code_block.to_panel(title=lang))
        elements.append(Text())  # Spacer
        
        last_end = match.end()
    
    # Add remaining text
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining.strip():
            elements.append(Markdown(remaining))
    
    return Group(*elements)
