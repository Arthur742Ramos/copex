"""
Stats tracking for Copex runs.

Provides:
- StatsTracker: append-only JSONL logger for per-run statistics
- Helper to read/aggregate stats for display
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _copex_dir() -> Path:
    return Path.home() / ".copex"


def _stats_path() -> Path:
    return _copex_dir() / "stats.jsonl"


def _state_path() -> Path:
    return _copex_dir() / "state.json"


@dataclass
class RunStats:
    """Statistics for a single copex run."""

    timestamp: str
    model: str
    reasoning_effort: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    duration_ms: float = 0.0
    command: str = ""  # chat / plan / fleet / ralph / etc.
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StatsTracker:
    """Append-only stats logger backed by ``~/.copex/stats.jsonl``."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _stats_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log(self, stats: RunStats) -> None:
        """Append a run record."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats.to_dict()) + "\n")

        # Also update state.json with latest run info
        self._update_state(stats)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def all_runs(self) -> list[dict[str, Any]]:
        """Return every logged run."""
        if not self.path.exists():
            return []
        runs: list[dict[str, Any]] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return runs

    def runs_today(self) -> list[dict[str, Any]]:
        """Return runs whose timestamp falls on today (UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return [r for r in self.all_runs() if r.get("timestamp", "").startswith(today)]

    def last_run(self) -> dict[str, Any] | None:
        """Return the most recent run, or None."""
        runs = self.all_runs()
        return runs[-1] if runs else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_state(self, stats: RunStats) -> None:
        state_path = _state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {}
        if state_path.exists():
            try:
                with open(state_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

        data["last_run"] = stats.to_dict()

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# ------------------------------------------------------------------
# State helpers for diff tracking
# ------------------------------------------------------------------


def save_start_commit(commit_hash: str) -> None:
    """Store the HEAD commit hash at the start of a copex run."""
    state_path = _state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}
    if state_path.exists():
        try:
            with open(state_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    data["last_start_commit"] = commit_hash
    data["last_start_time"] = datetime.now(timezone.utc).isoformat()

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_start_commit() -> str | None:
    """Load the stored start commit hash."""
    state_path = _state_path()
    if not state_path.exists():
        return None
    try:
        with open(state_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("last_start_commit")
    except (OSError, json.JSONDecodeError):
        return None


def load_state() -> dict[str, Any]:
    """Load the full state dict."""
    state_path = _state_path()
    if not state_path.exists():
        return {}
    try:
        with open(state_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
