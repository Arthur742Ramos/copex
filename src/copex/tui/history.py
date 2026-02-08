"""Prompt history and stash management for Copex TUI."""

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class HistoryEntry:
    """A single history entry."""

    content: str
    timestamp: float = field(default_factory=time.time)
    model: str | None = None
    tags: list[str] = field(default_factory=list)

    def matches(self, query: str) -> bool:
        """Check if entry matches a search query."""
        query_lower = query.lower()
        return query_lower in self.content.lower() or any(
            query_lower in tag.lower() for tag in self.tags
        )


@dataclass
class StashEntry:
    """A stashed prompt draft."""

    content: str
    cursor_position: int
    timestamp: float = field(default_factory=time.time)
    name: str | None = None  # Optional name for the stash


class PromptHistory:
    """
    Manages prompt history with persistence.

    Provides:
    - Persistent history across sessions
    - Search/filter capabilities
    - Deduplication
    - Maximum size limits
    """

    def __init__(
        self,
        history_file: Path | None = None,
        max_entries: int = 1000,
    ) -> None:
        self.history_file = history_file or (Path.home() / ".copex" / "history.json")
        self.max_entries = max_entries
        self._entries: list[HistoryEntry] = []
        self._position: int = -1  # Current navigation position
        self._load()

    def _load(self) -> None:
        """Load history from file."""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, encoding="utf-8") as f:
                data = json.load(f)

            self._entries = [
                HistoryEntry(
                    content=e["content"],
                    timestamp=e.get("timestamp", 0),
                    model=e.get("model"),
                    tags=e.get("tags", []),
                )
                for e in data.get("entries", [])
            ]
        except (json.JSONDecodeError, OSError):
            self._entries = []

    def _save(self) -> None:
        """Save history to file."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        data = {"entries": [asdict(e) for e in self._entries[-self.max_entries :]]}

        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass  # Silently fail on write errors

    def add(
        self,
        content: str,
        model: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Add an entry to history."""
        content = content.strip()
        if not content:
            return

        # Deduplicate: remove existing entry with same content
        self._entries = [e for e in self._entries if e.content != content]

        # Add new entry
        entry = HistoryEntry(
            content=content,
            model=model,
            tags=tags or [],
        )
        self._entries.append(entry)

        # Trim to max size
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

        # Reset navigation position
        self._position = -1

        self._save()

    def search(self, query: str) -> list[HistoryEntry]:
        """Search history for matching entries."""
        if not query:
            return list(reversed(self._entries))

        return [e for e in reversed(self._entries) if e.matches(query)]

    def get_previous(self, current: str = "") -> str | None:
        """Get the previous history entry (up arrow)."""
        if not self._entries:
            return None

        # First navigation - start from the end
        if self._position == -1:
            self._position = len(self._entries)

        # Move up
        if self._position > 0:
            self._position -= 1
            return self._entries[self._position].content

        return None

    def get_next(self, current: str = "") -> str | None:
        """Get the next history entry (down arrow)."""
        if not self._entries or self._position == -1:
            return None

        # Move down
        if self._position < len(self._entries) - 1:
            self._position += 1
            return self._entries[self._position].content

        # At the end - return to empty
        self._position = -1
        return ""

    def reset_position(self) -> None:
        """Reset navigation position."""
        self._position = -1

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[HistoryEntry]:
        return iter(reversed(self._entries))

    def clear(self) -> None:
        """Clear all history."""
        self._entries = []
        self._position = -1
        self._save()


class PromptStash:
    """
    Manages prompt drafts (stash).

    Provides:
    - Save current draft
    - Restore drafts
    - Cycle through stash
    - Optional persistence
    """

    def __init__(
        self,
        stash_file: Path | None = None,
        max_entries: int = 10,
        persist: bool = True,
    ) -> None:
        self.stash_file = stash_file or (Path.home() / ".copex" / "stash.json")
        self.max_entries = max_entries
        self.persist = persist
        self._entries: list[StashEntry] = []
        self._position: int = -1

        if persist:
            self._load()

    def _load(self) -> None:
        """Load stash from file."""
        if not self.stash_file.exists():
            return

        try:
            with open(self.stash_file, encoding="utf-8") as f:
                data = json.load(f)

            self._entries = [
                StashEntry(
                    content=e["content"],
                    cursor_position=e.get("cursor_position", 0),
                    timestamp=e.get("timestamp", 0),
                    name=e.get("name"),
                )
                for e in data.get("entries", [])
            ]
        except (json.JSONDecodeError, OSError):
            self._entries = []

    def _save(self) -> None:
        """Save stash to file."""
        if not self.persist:
            return

        self.stash_file.parent.mkdir(parents=True, exist_ok=True)

        data = {"entries": [asdict(e) for e in self._entries]}

        try:
            with open(self.stash_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def push(
        self,
        content: str,
        cursor_position: int = 0,
        name: str | None = None,
    ) -> bool:
        """
        Push a draft to the stash.

        Returns True if saved, False if empty.
        """
        content = content.strip()
        if not content:
            return False

        entry = StashEntry(
            content=content,
            cursor_position=cursor_position,
            name=name,
        )
        self._entries.append(entry)

        # Trim to max size
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

        self._position = len(self._entries) - 1
        self._save()
        return True

    def pop(self) -> StashEntry | None:
        """Pop the most recent entry from stash."""
        if not self._entries:
            return None

        entry = self._entries.pop()
        self._position = len(self._entries) - 1
        self._save()
        return entry

    def peek(self, index: int | None = None) -> StashEntry | None:
        """Peek at a stash entry without removing it."""
        if not self._entries:
            return None

        if index is None:
            index = len(self._entries) - 1

        if 0 <= index < len(self._entries):
            return self._entries[index]

        return None

    def cycle_next(self) -> StashEntry | None:
        """Cycle to the next stash entry."""
        if not self._entries:
            return None

        self._position = (self._position + 1) % len(self._entries)
        return self._entries[self._position]

    def cycle_prev(self) -> StashEntry | None:
        """Cycle to the previous stash entry."""
        if not self._entries:
            return None

        self._position = (self._position - 1) % len(self._entries)
        return self._entries[self._position]

    def get_current(self) -> StashEntry | None:
        """Get the current stash entry."""
        if not self._entries or self._position < 0:
            return None

        if 0 <= self._position < len(self._entries):
            return self._entries[self._position]

        return None

    @property
    def current_index(self) -> int:
        """Get the current position in the stash."""
        return self._position

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[StashEntry]:
        return iter(self._entries)

    def clear(self) -> None:
        """Clear all stashed entries."""
        self._entries = []
        self._position = -1
        self._save()

    def remove(self, index: int) -> bool:
        """Remove a specific entry by index."""
        if 0 <= index < len(self._entries):
            self._entries.pop(index)
            if self._position >= len(self._entries):
                self._position = len(self._entries) - 1
            self._save()
            return True
        return False


class CombinedHistoryManager:
    """
    Combined manager for history and stash.

    Provides a unified interface for the TUI.
    """

    def __init__(
        self,
        history_dir: Path | None = None,
        max_history: int = 1000,
        max_stash: int = 10,
    ) -> None:
        base_dir = history_dir or (Path.home() / ".copex")

        self.history = PromptHistory(
            history_file=base_dir / "history.json",
            max_entries=max_history,
        )
        self.stash = PromptStash(
            stash_file=base_dir / "stash.json",
            max_entries=max_stash,
        )

    def add_to_history(
        self,
        content: str,
        model: str | None = None,
    ) -> None:
        """Add a sent prompt to history."""
        self.history.add(content, model=model)

    def stash_draft(
        self,
        content: str,
        cursor_position: int = 0,
    ) -> bool:
        """Stash current draft."""
        return self.stash.push(content, cursor_position)

    def restore_draft(self) -> tuple[str, int] | None:
        """Restore most recent draft."""
        entry = self.stash.pop()
        if entry:
            return entry.content, entry.cursor_position
        return None

    def history_up(self, current: str = "") -> str | None:
        """Navigate up in history."""
        return self.history.get_previous(current)

    def history_down(self, current: str = "") -> str | None:
        """Navigate down in history."""
        return self.history.get_next(current)

    def cycle_stash(self) -> tuple[str, int] | None:
        """Cycle through stash."""
        entry = self.stash.cycle_prev()
        if entry:
            return entry.content, entry.cursor_position
        return None

    @property
    def has_stash(self) -> bool:
        """Check if there are stashed drafts."""
        return len(self.stash) > 0

    @property
    def stash_count(self) -> int:
        """Get number of stashed drafts."""
        return len(self.stash)

    @property
    def stash_position(self) -> int:
        """Get current stash position."""
        return self.stash.current_index
