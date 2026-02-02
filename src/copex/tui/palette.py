"""Command palette with fuzzy matching for Copex TUI."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

# NOTE:
# Avoid __future__.annotations here; these modules are imported in tests via
# importlib.util without being inserted into sys.modules, which breaks
# dataclasses on Python 3.14 when annotations are strings.


class CommandCategory(str, Enum):
    """Command categories for organization."""
    MODEL = "model"
    REASONING = "reasoning"
    SESSION = "session"
    EXPORT = "export"
    VIEW = "view"
    HELP = "help"


@dataclass
class PaletteCommand:
    """A command in the palette."""
    id: str
    label: str
    description: str
    category: CommandCategory
    shortcut: str | None = None
    action: Callable[[], Any] | None = None
    # For subcommands (e.g., model selection)
    subcommands: list["PaletteCommand"] = field(default_factory=list)
    # Value for commands that set something
    value: Any = None


def fuzzy_match(pattern: str, text: str) -> tuple[bool, int]:
    """
    Fuzzy match pattern against text.
    
    Returns (is_match, score) where higher score is better match.
    Uses a simple algorithm that prefers:
    - Consecutive character matches
    - Matches at word boundaries
    - Matches at the start
    """
    if not pattern:
        return True, 0
    
    pattern_lower = pattern.lower()
    text_lower = text.lower()
    
    # Exact substring match gets highest score
    if pattern_lower in text_lower:
        # Bonus for match at start
        start_bonus = 100 if text_lower.startswith(pattern_lower) else 0
        return True, 1000 + start_bonus - text_lower.index(pattern_lower)
    
    # Fuzzy matching
    pattern_idx = 0
    score = 0
    last_match_idx = -1
    word_boundary = True
    
    for i, char in enumerate(text_lower):
        if pattern_idx < len(pattern_lower) and char == pattern_lower[pattern_idx]:
            pattern_idx += 1
            
            # Consecutive match bonus
            if last_match_idx == i - 1:
                score += 10
            
            # Word boundary bonus
            if word_boundary:
                score += 15
            
            # Start bonus
            if i == 0:
                score += 20
            
            last_match_idx = i
            score += 5
        
        # Track word boundaries
        word_boundary = char in " _-./\\"
    
    if pattern_idx == len(pattern_lower):
        return True, score
    
    return False, 0


def filter_commands(
    commands: list[PaletteCommand],
    query: str,
) -> list[tuple[PaletteCommand, int]]:
    """
    Filter and rank commands by fuzzy matching.
    
    Returns list of (command, score) tuples, sorted by score descending.
    """
    if not query:
        return [(cmd, 0) for cmd in commands]
    
    results: list[tuple[PaletteCommand, int]] = []
    
    for cmd in commands:
        # Match against label, description, and id
        label_match, label_score = fuzzy_match(query, cmd.label)
        desc_match, desc_score = fuzzy_match(query, cmd.description)
        id_match, id_score = fuzzy_match(query, cmd.id)
        
        # Also match against category
        cat_match, cat_score = fuzzy_match(query, cmd.category.value)
        
        if label_match or desc_match or id_match or cat_match:
            # Prefer label matches, then id, then description, then category
            total_score = (
                label_score * 3 +
                id_score * 2 +
                desc_score +
                cat_score // 2
            )
            results.append((cmd, total_score))
    
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


class CommandPalette:
    """Command palette for the TUI."""
    
    REASONING_EFFORTS = ["none", "low", "medium", "high", "xhigh"]
    
    def __init__(self) -> None:
        self.commands: list[PaletteCommand] = []
        self._base_commands: list[PaletteCommand] = []
        self._command_stack: list[list[PaletteCommand]] = []
        self._build_default_commands()
    
    def _build_default_commands(self) -> None:
        """Build the default set of commands."""
        # Dynamic model list from the main copex Model enum.
        from copex.models import Model
        models = [m.value for m in Model]
        models.sort()
        # Model commands
        model_subcommands = [
            PaletteCommand(
                id=f"model:{model_id}",
                label=model_id,
                description=f"Switch to {model_id}",
                category=CommandCategory.MODEL,
                value=model_id,
            )
            for model_id in models
        ]
        
        self.commands.append(PaletteCommand(
            id="model",
            label="Change Model",
            description="Switch to a different AI model",
            category=CommandCategory.MODEL,
            shortcut="Ctrl+M",
            subcommands=model_subcommands,
        ))
        
        # Reasoning commands
        reasoning_subcommands = [
            PaletteCommand(
                id=f"reasoning:{r}",
                label=r,
                description=f"Set reasoning effort to {r}",
                category=CommandCategory.REASONING,
                value=r,
            )
            for r in self.REASONING_EFFORTS
        ]
        reasoning_subcommands.sort(key=lambda cmd: self.REASONING_EFFORTS.index(cmd.label))
        
        self.commands.append(PaletteCommand(
            id="reasoning",
            label="Set Reasoning Effort",
            description="Adjust model reasoning level",
            category=CommandCategory.REASONING,
            shortcut="Ctrl+E",
            subcommands=reasoning_subcommands,
        ))
        
        # Session commands
        self.commands.extend([
            PaletteCommand(
                id="session:new",
                label="New Session",
                description="Start a fresh conversation",
                category=CommandCategory.SESSION,
                shortcut="Ctrl+N",
            ),
            PaletteCommand(
                id="session:clear",
                label="Clear Screen",
                description="Clear the screen output",
                category=CommandCategory.SESSION,
                shortcut="Ctrl+L",
            ),
            PaletteCommand(
                id="session:status",
                label="Show Status",
                description="Show current session metrics",
                category=CommandCategory.SESSION,
            ),
        ])
        
        # Export commands
        self.commands.extend([
            PaletteCommand(
                id="export:json",
                label="Export as JSON",
                description="Export conversation to JSON",
                category=CommandCategory.EXPORT,
            ),
            PaletteCommand(
                id="export:markdown",
                label="Export as Markdown",
                description="Export conversation to Markdown",
                category=CommandCategory.EXPORT,
            ),
            PaletteCommand(
                id="export:metrics",
                label="Export Metrics",
                description="Export usage metrics (CSV)",
                category=CommandCategory.EXPORT,
            ),
        ])
        
        # View commands
        self.commands.extend([
            PaletteCommand(
                id="view:tools:toggle",
                label="Toggle Tool Calls",
                description="Expand/collapse tool call panels",
                category=CommandCategory.VIEW,
                shortcut="Ctrl+T",
            ),
            PaletteCommand(
                id="view:reasoning:toggle",
                label="Toggle Reasoning",
                description="Show or hide reasoning panel",
                category=CommandCategory.VIEW,
                shortcut="Ctrl+G",
            ),
            PaletteCommand(
                id="view:statusbar:toggle",
                label="Toggle Status Bar",
                description="Show or hide status bar",
                category=CommandCategory.VIEW,
            ),
        ])
        
        # Help commands
        self.commands.extend([
            PaletteCommand(
                id="help:shortcuts",
                label="Keyboard Shortcuts",
                description="Show all keyboard shortcuts",
                category=CommandCategory.HELP,
                shortcut="?",
            ),
            PaletteCommand(
                id="help:about",
                label="About Copex",
                description="Show version and information",
                category=CommandCategory.HELP,
            ),
        ])
        self._base_commands = list(self.commands)
    
    def search(self, query: str) -> list[tuple[PaletteCommand, int]]:
        """Search commands with fuzzy matching."""
        return filter_commands(self.commands, query)

    @property
    def has_parent(self) -> bool:
        """Return True if palette is showing subcommands."""
        return bool(self._command_stack)

    def reset(self) -> None:
        """Reset palette to the top-level commands."""
        self.commands = list(self._base_commands)
        self._command_stack = []

    def push_commands(self, commands: list[PaletteCommand]) -> None:
        """Show a subcommand list."""
        self._command_stack.append(self.commands)
        self.commands = list(commands)

    def pop_commands(self) -> bool:
        """Return to the previous command list."""
        if not self._command_stack:
            return False
        self.commands = self._command_stack.pop()
        return True
    
    def get_command(self, command_id: str) -> PaletteCommand | None:
        """Get a command by ID."""
        for cmd in self.commands:
            if cmd.id == command_id:
                return cmd
            # Check subcommands
            for sub in cmd.subcommands:
                if sub.id == command_id:
                    return sub
        return None
    
    def get_commands_by_category(
        self,
        category: CommandCategory,
    ) -> list[PaletteCommand]:
        """Get all commands in a category."""
        return [cmd for cmd in self.commands if cmd.category == category]
    
    def register_command(self, command: PaletteCommand) -> None:
        """Register a new command."""
        # Remove existing command with same ID
        self.commands = [c for c in self.commands if c.id != command.id]
        self.commands.append(command)
    
    def set_action(self, command_id: str, action: Callable[[], Any]) -> bool:
        """Set the action for a command. Returns True if found."""
        cmd = self.get_command(command_id)
        if cmd:
            cmd.action = action
            return True
        return False
