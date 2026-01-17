"""
Session Persistence - Save and restore conversation history.

Allows saving sessions to disk and resuming later, useful for:
- Long-running tasks that span multiple sessions
- Crash recovery
- Sharing context between different runs
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from copex.models import Model, ReasoningEffort


@dataclass
class Message:
    """A message in the conversation history."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionData:
    """Persistent session data."""

    id: str
    created_at: str
    updated_at: str
    model: str
    reasoning_effort: str
    messages: list[Message]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "messages": [asdict(m) for m in self.messages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionData":
        """Create from dictionary."""
        messages = [Message(**m) for m in data.get("messages", [])]
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            model=data["model"],
            reasoning_effort=data["reasoning_effort"],
            messages=messages,
            metadata=data.get("metadata", {}),
        )


class SessionStore:
    """
    Persistent storage for Copex sessions.

    Usage:
        store = SessionStore()
        
        # Save a session
        store.save(session_id, messages, model, reasoning)
        
        # Load a session
        data = store.load(session_id)
        
        # List all sessions
        sessions = store.list_sessions()
    """

    def __init__(self, base_dir: Path | str | None = None):
        """
        Initialize session store.

        Args:
            base_dir: Directory for session files. Defaults to ~/.copex/sessions
        """
        if base_dir is None:
            base_dir = Path.home() / ".copex" / "sessions"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Get path for a session file."""
        # Sanitize session ID for filesystem
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self.base_dir / f"{safe_id}.json"

    def save(
        self,
        session_id: str,
        messages: list[Message],
        model: Model | str,
        reasoning_effort: ReasoningEffort | str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save a session to disk.

        Returns:
            Path to the saved session file
        """
        path = self._session_path(session_id)
        now = datetime.now().isoformat()

        # Check if session exists
        if path.exists():
            existing = self.load(session_id)
            created_at = existing.created_at if existing else now
        else:
            created_at = now

        data = SessionData(
            id=session_id,
            created_at=created_at,
            updated_at=now,
            model=model.value if isinstance(model, Model) else model,
            reasoning_effort=(
                reasoning_effort.value
                if isinstance(reasoning_effort, ReasoningEffort)
                else reasoning_effort
            ),
            messages=messages,
            metadata=metadata or {},
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data.to_dict(), f, indent=2, ensure_ascii=False)

        return path

    def load(self, session_id: str) -> SessionData | None:
        """Load a session from disk."""
        path = self._session_path(session_id)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return SessionData.from_dict(data)

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all saved sessions.

        Returns:
            List of session summaries (id, created_at, updated_at, message_count)
        """
        sessions = []
        for path in self.base_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "id": data["id"],
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "model": data["model"],
                    "message_count": len(data.get("messages", [])),
                })
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by updated_at descending
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        return sessions

    def export(self, session_id: str, format: str = "json") -> str:
        """
        Export a session to a string.

        Args:
            session_id: Session to export
            format: Export format ("json" or "markdown")

        Returns:
            Exported session as string
        """
        data = self.load(session_id)
        if not data:
            raise ValueError(f"Session not found: {session_id}")

        if format == "json":
            return json.dumps(data.to_dict(), indent=2, ensure_ascii=False)

        elif format == "markdown":
            lines = [
                f"# Session: {data.id}",
                f"",
                f"- **Model**: {data.model}",
                f"- **Reasoning**: {data.reasoning_effort}",
                f"- **Created**: {data.created_at}",
                f"- **Updated**: {data.updated_at}",
                f"",
                "---",
                "",
            ]

            for msg in data.messages:
                role_label = {"user": "👤 User", "assistant": "🤖 Assistant", "system": "⚙️ System"}.get(
                    msg.role, msg.role
                )
                lines.append(f"### {role_label}")
                lines.append(f"*{msg.timestamp}*")
                lines.append("")
                lines.append(msg.content)
                lines.append("")
                lines.append("---")
                lines.append("")

            return "\n".join(lines)

        else:
            raise ValueError(f"Unknown format: {format}")


class PersistentSession:
    """
    A session wrapper that auto-saves to disk.

    Usage:
        session = PersistentSession("my-project", store)
        session.add_user_message("Hello")
        session.add_assistant_message("Hi there!")
        # Automatically saved after each message
    """

    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        model: Model | str = Model.GPT_5_2_CODEX,
        reasoning_effort: ReasoningEffort | str = ReasoningEffort.XHIGH,
        auto_save: bool = True,
    ):
        self.session_id = session_id
        self.store = store
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.auto_save = auto_save
        self.messages: list[Message] = []
        self.metadata: dict[str, Any] = {}

        # Load existing session if it exists
        existing = store.load(session_id)
        if existing:
            self.messages = existing.messages
            self.metadata = existing.metadata
            self.model = existing.model
            self.reasoning_effort = existing.reasoning_effort

    def add_message(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a message to the session."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        if self.auto_save:
            self.save()

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str, reasoning: str | None = None) -> None:
        """Add an assistant message."""
        metadata = {"reasoning": reasoning} if reasoning else {}
        self.add_message("assistant", content, metadata)

    def save(self) -> Path:
        """Save the session to disk."""
        return self.store.save(
            self.session_id,
            self.messages,
            self.model,
            self.reasoning_effort,
            self.metadata,
        )

    def clear(self) -> None:
        """Clear all messages (keeps session file)."""
        self.messages = []
        if self.auto_save:
            self.save()

    def get_context(self, max_messages: int | None = None) -> list[dict[str, str]]:
        """
        Get messages formatted for API context.

        Args:
            max_messages: Limit to last N messages (None = all)

        Returns:
            List of {"role": ..., "content": ...} dicts
        """
        messages = self.messages[-max_messages:] if max_messages else self.messages
        return [{"role": m.role, "content": m.content} for m in messages]
