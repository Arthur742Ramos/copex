"""Core UI components for Copex - syntax highlighting, diffs, collapsible sections."""

from __future__ import annotations

import re
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
    
    def __rich__(self) -> Text:
        """Allow direct printing with Rich console."""
        return self.to_text()


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
