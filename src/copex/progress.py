"""
Progress Reporting - Real-time progress tracking for long operations.

Provides:
- Progress bars for multi-step operations
- ETA estimation
- Status callbacks for integration with external tools
- JSON/structured output for automation
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TextIO


class ProgressStatus(str, Enum):
    """Status of a progress item."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProgressItem:
    """A single progress item (e.g., a plan step)."""

    id: str | int
    description: str
    status: ProgressStatus = ProgressStatus.PENDING
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        """Get duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at


@dataclass
class ProgressState:
    """Overall progress state."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    running: int = 0
    items: list[ProgressItem] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    message: str = ""

    @property
    def percent(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.started_at

    @property
    def eta_seconds(self) -> float | None:
        """Estimate time remaining in seconds."""
        if self.completed == 0:
            return None
        avg_time = self.elapsed / self.completed
        remaining = self.total - self.completed
        return avg_time * remaining

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "running": self.running,
            "percent": round(self.percent, 1),
            "elapsed_seconds": round(self.elapsed, 1),
            "eta_seconds": round(self.eta_seconds, 1) if self.eta_seconds else None,
            "message": self.message,
            "items": [
                {
                    "id": item.id,
                    "description": item.description,
                    "status": item.status.value,
                    "duration": round(item.duration, 1) if item.duration else None,
                    "error": item.error,
                }
                for item in self.items
            ],
        }


class ProgressReporter:
    """
    Reports progress for long-running operations.

    Supports multiple output formats:
    - Terminal progress bar
    - JSON lines for automation
    - Callbacks for custom handling

    Usage:
        reporter = ProgressReporter(total=10)

        for i, item in enumerate(items):
            reporter.start_item(i, f"Processing {item}")
            try:
                process(item)
                reporter.complete_item(i)
            except Exception as e:
                reporter.fail_item(i, str(e))

        reporter.finish()
    """

    def __init__(
        self,
        total: int = 0,
        *,
        output: TextIO | None = None,
        format: str = "terminal",  # "terminal", "json", "quiet"
        on_update: Callable[[ProgressState], None] | None = None,
        title: str = "Progress",
    ):
        """
        Initialize progress reporter.

        Args:
            total: Total number of items
            output: Output stream (default: stderr)
            format: Output format ("terminal", "json", "quiet")
            on_update: Callback on each state change
            title: Title for the progress display
        """
        self.state = ProgressState(total=total)
        self.output = output or sys.stderr
        self.format = format
        self.on_update = on_update
        self.title = title
        self._last_render = 0.0
        self._render_interval = 0.1  # seconds

    def set_total(self, total: int) -> None:
        """Set total item count."""
        self.state.total = total
        self._update()

    def set_message(self, message: str) -> None:
        """Set status message."""
        self.state.message = message
        self._update()

    def add_item(self, id: str | int, description: str) -> ProgressItem:
        """Add a progress item."""
        item = ProgressItem(id=id, description=description)
        self.state.items.append(item)
        return item

    def start_item(self, id: str | int, description: str | None = None) -> None:
        """Mark an item as started."""
        item = self._get_or_create_item(id, description or f"Item {id}")
        item.status = ProgressStatus.RUNNING
        item.started_at = time.time()
        self.state.running += 1
        self._update()

    def complete_item(self, id: str | int, message: str | None = None) -> None:
        """Mark an item as completed."""
        item = self._get_item(id)
        if item:
            prev_status = item.status
            item.status = ProgressStatus.COMPLETED
            item.completed_at = time.time()
            if message:
                item.metadata["message"] = message
            # Only decrement running if it was previously RUNNING
            if prev_status == ProgressStatus.RUNNING:
                self.state.running = max(0, self.state.running - 1)
            self.state.completed += 1
            self._update()

    def fail_item(self, id: str | int, error: str) -> None:
        """Mark an item as failed."""
        item = self._get_item(id)
        if item:
            prev_status = item.status
            item.status = ProgressStatus.FAILED
            item.completed_at = time.time()
            item.error = error
            # Only decrement running if it was previously RUNNING
            if prev_status == ProgressStatus.RUNNING:
                self.state.running = max(0, self.state.running - 1)
            self.state.failed += 1
            self._update()

    def skip_item(self, id: str | int, reason: str | None = None) -> None:
        """Mark an item as skipped."""
        item = self._get_item(id)
        if item:
            item.status = ProgressStatus.SKIPPED
            item.completed_at = time.time()
            if reason:
                item.metadata["skip_reason"] = reason
            self.state.completed += 1  # Count as complete for progress
            self._update()

    def finish(self, message: str | None = None) -> None:
        """Finish progress reporting."""
        if message:
            self.state.message = message
        self._render(force=True)
        if self.format == "terminal":
            self.output.write("\n")
            self.output.flush()

    def _get_or_create_item(self, id: str | int, description: str) -> ProgressItem:
        """Get existing item or create new one."""
        item = self._get_item(id)
        if item:
            return item
        return self.add_item(id, description)

    def _get_item(self, id: str | int) -> ProgressItem | None:
        """Get item by ID."""
        for item in self.state.items:
            if item.id == id:
                return item
        return None

    def _update(self) -> None:
        """Handle state update."""
        if self.on_update:
            self.on_update(self.state)
        self._render()

    def _render(self, force: bool = False) -> None:
        """Render progress output."""
        now = time.time()
        if not force and (now - self._last_render) < self._render_interval:
            return
        self._last_render = now

        if self.format == "terminal":
            self._render_terminal()
        elif self.format == "json":
            self._render_json()
        # "quiet" format: no output

    def _render_terminal(self) -> None:
        """Render terminal progress bar."""
        bar_width = 30
        filled = int(bar_width * self.state.percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        elapsed_str = self._format_time(self.state.elapsed)
        eta_str = self._format_time(self.state.eta_seconds) if self.state.eta_seconds else "?"

        status = f"\r{self.title}: [{bar}] {self.state.completed}/{self.state.total}"
        status += f" ({self.state.percent:.0f}%) | {elapsed_str} elapsed | ETA: {eta_str}"

        if self.state.failed:
            status += f" | {self.state.failed} failed"

        if self.state.message:
            status += f" | {self.state.message}"

        # Pad to overwrite previous output
        status = status.ljust(120)

        self.output.write(status)
        self.output.flush()

    def _render_json(self) -> None:
        """Render JSON lines output."""
        output = {
            "type": "progress",
            "timestamp": datetime.now().isoformat(),
            **self.state.to_dict(),
        }
        self.output.write(json.dumps(output) + "\n")
        self.output.flush()

    @staticmethod
    def _format_time(seconds: float | None) -> str:
        """Format seconds as human-readable time."""
        if seconds is None:
            return "?"
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours}h {mins}m"


class PlanProgressReporter(ProgressReporter):
    """
    Specialized progress reporter for plan execution.

    Usage:
        from copex.plan import Plan, PlanExecutor
        from copex.progress import PlanProgressReporter

        reporter = PlanProgressReporter(plan)

        await executor.execute_plan(
            plan,
            on_step_start=reporter.on_step_start,
            on_step_complete=reporter.on_step_complete,
            on_error=reporter.on_error,
        )
    """

    def __init__(
        self,
        plan: Any,  # Plan type, avoiding circular import
        **kwargs,
    ):
        """Initialize from a plan."""
        super().__init__(total=len(plan.steps), title="Plan Execution", **kwargs)
        self.plan = plan

        # Pre-populate items from plan
        for step in plan.steps:
            self.add_item(step.number, step.description)

    def on_step_start(self, step: Any) -> None:
        """Callback for plan step start."""
        self.start_item(step.number, step.description)

    def on_step_complete(self, step: Any) -> None:
        """Callback for plan step completion."""
        self.complete_item(step.number, step.result[:50] if step.result else None)

    def on_error(self, step: Any, error: Exception) -> bool:
        """Callback for plan step error. Returns True to continue."""
        self.fail_item(step.number, str(error))
        return True  # Continue by default
