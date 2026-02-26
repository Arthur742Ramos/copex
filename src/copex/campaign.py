"""
Campaign orchestration — high-level multi-wave fleet execution.

Takes a goal description and a discovery command, then:
  1. Runs the discovery command to find targets
  2. Batches targets into groups
  3. Generates fleet tasks for each batch
  4. Runs waves sequentially (each wave is a parallel fleet batch)
  5. Reports results and stores campaign state for resume

State is stored in .copex/campaign.json so interrupted campaigns can resume.

Usage:
    copex campaign \\
        --goal "Add type annotations to all modules" \\
        --discover "find . -name '*.py' -not -path './.venv/*'" \\
        --batch-size 5
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from copex.fleet import FleetTask

logger = logging.getLogger(__name__)

# Default campaign state directory
CAMPAIGN_DIR = Path(".copex")
CAMPAIGN_STATE_FILE = CAMPAIGN_DIR / "campaign.json"


class WaveStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WaveResult:
    """Result of a single wave execution."""

    wave_index: int
    targets: list[str]
    status: WaveStatus = WaveStatus.PENDING
    succeeded: int = 0
    failed: int = 0
    duration_seconds: float = 0.0
    task_results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class CampaignState:
    """Persistent campaign state for resume support."""

    goal: str
    discover_command: str
    batch_size: int
    all_targets: list[str]
    waves: list[WaveResult]
    created_at: str = ""
    updated_at: str = ""
    total_duration_seconds: float = 0.0
    status: str = "in_progress"  # in_progress | completed | failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "discover_command": self.discover_command,
            "batch_size": self.batch_size,
            "all_targets": self.all_targets,
            "waves": [
                {
                    "wave_index": w.wave_index,
                    "targets": w.targets,
                    "status": w.status.value,
                    "succeeded": w.succeeded,
                    "failed": w.failed,
                    "duration_seconds": w.duration_seconds,
                    "task_results": w.task_results,
                    "error": w.error,
                }
                for w in self.waves
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_duration_seconds": self.total_duration_seconds,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignState:
        waves = []
        for w in data.get("waves", []):
            waves.append(
                WaveResult(
                    wave_index=w["wave_index"],
                    targets=w["targets"],
                    status=WaveStatus(w.get("status", "pending")),
                    succeeded=w.get("succeeded", 0),
                    failed=w.get("failed", 0),
                    duration_seconds=w.get("duration_seconds", 0.0),
                    task_results=w.get("task_results", []),
                    error=w.get("error"),
                )
            )
        return cls(
            goal=data["goal"],
            discover_command=data["discover_command"],
            batch_size=data.get("batch_size", 5),
            all_targets=data.get("all_targets", []),
            waves=waves,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            total_duration_seconds=data.get("total_duration_seconds", 0.0),
            status=data.get("status", "in_progress"),
        )


def save_campaign_state(state: CampaignState, path: Path | None = None) -> Path:
    """Persist campaign state to disk."""
    path = path or CAMPAIGN_STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    return path


def load_campaign_state(path: Path | None = None) -> CampaignState | None:
    """Load campaign state from disk, or return None if not found."""
    path = path or CAMPAIGN_STATE_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CampaignState.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to load campaign state from %s: %s", path, exc)
        return None


def run_discover_command(command: str, cwd: str | None = None) -> list[str]:
    """Run the discovery command and return discovered targets (one per line).

    Args:
        command: Shell command to run.
        cwd: Working directory (defaults to current).

    Returns:
        List of non-empty target strings (file paths, identifiers, etc.)

    Raises:
        RuntimeError: If the discovery command fails.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Discovery command timed out after 120s: {command}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"Discovery command failed (exit {result.returncode}): {stderr or command}"
        )

    targets = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    logger.info("Discovery found %d targets", len(targets))
    return targets


def batch_targets(targets: list[str], batch_size: int) -> list[list[str]]:
    """Split targets into batches of the given size."""
    if batch_size < 1:
        batch_size = 1
    return [targets[i : i + batch_size] for i in range(0, len(targets), batch_size)]


def generate_wave_tasks(
    goal: str,
    targets: list[str],
    wave_index: int,
) -> list[FleetTask]:
    """Generate fleet tasks for a single wave (batch of targets).

    Each target gets its own FleetTask with the goal contextualized
    to that specific target.
    """
    tasks: list[FleetTask] = []
    for i, target in enumerate(targets):
        task_id = f"wave-{wave_index}-task-{i + 1}"
        prompt = (
            f"Goal: {goal}\n\n"
            f"Target: {target}\n\n"
            f"Apply the goal to this specific target. "
            f"Make the necessary changes and verify they work."
        )
        tasks.append(
            FleetTask(
                id=task_id,
                prompt=prompt,
            )
        )
    return tasks


def get_pending_wave_indices(state: CampaignState) -> list[int]:
    """Return indices of waves that haven't completed yet."""
    return [
        i
        for i, w in enumerate(state.waves)
        if w.status in (WaveStatus.PENDING, WaveStatus.FAILED)
    ]


def create_campaign(
    goal: str,
    discover_command: str,
    batch_size: int,
    targets: list[str],
) -> CampaignState:
    """Create a new campaign state from discovered targets."""
    batches = batch_targets(targets, batch_size)
    waves = [
        WaveResult(wave_index=i, targets=batch)
        for i, batch in enumerate(batches)
    ]

    state = CampaignState(
        goal=goal,
        discover_command=discover_command,
        batch_size=batch_size,
        all_targets=targets,
        waves=waves,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return state
