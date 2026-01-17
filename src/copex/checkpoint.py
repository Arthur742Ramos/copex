"""
Checkpointing - Save and restore Ralph loop state for crash recovery.

Enables:
- Resuming interrupted Ralph loops
- Crash recovery
- Inspection of loop progress
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Checkpoint:
    """A checkpoint of Ralph loop state."""

    # Identity
    checkpoint_id: str
    loop_id: str

    # Loop state
    prompt: str
    iteration: int
    max_iterations: int | None
    completion_promise: str | None

    # Timestamps
    created_at: str
    updated_at: str
    started_at: str

    # History
    history: list[str] = field(default_factory=list)

    # Status
    completed: bool = False
    completion_reason: str | None = None

    # Metadata
    model: str = "gpt-5.2-codex"
    reasoning_effort: str = "xhigh"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Create from dictionary."""
        return cls(**data)


class CheckpointStore:
    """
    Persistent storage for Ralph loop checkpoints.

    Usage:
        store = CheckpointStore()
        
        # Create checkpoint
        cp = store.create("my-loop", prompt, iteration, ...)
        
        # Update on each iteration
        store.update(cp.checkpoint_id, iteration=5, history=[...])
        
        # Resume after crash
        cp = store.get_latest("my-loop")
    """

    def __init__(self, base_dir: Path | str | None = None):
        """
        Initialize checkpoint store.

        Args:
            base_dir: Directory for checkpoint files. Defaults to ~/.copex/checkpoints
        """
        if base_dir is None:
            base_dir = Path.home() / ".copex" / "checkpoints"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get path for a checkpoint file."""
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in checkpoint_id)
        return self.base_dir / f"{safe_id}.json"

    def create(
        self,
        loop_id: str,
        prompt: str,
        max_iterations: int | None = None,
        completion_promise: str | None = None,
        model: str = "gpt-5.2-codex",
        reasoning_effort: str = "xhigh",
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """
        Create a new checkpoint.

        Args:
            loop_id: Identifier for this loop (e.g., project name)
            prompt: The loop prompt
            max_iterations: Maximum iterations
            completion_promise: Completion promise text
            model: Model being used
            reasoning_effort: Reasoning effort level
            metadata: Additional metadata

        Returns:
            New Checkpoint object
        """
        now = datetime.now().isoformat()
        checkpoint_id = f"{loop_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            loop_id=loop_id,
            prompt=prompt,
            iteration=0,
            max_iterations=max_iterations,
            completion_promise=completion_promise,
            created_at=now,
            updated_at=now,
            started_at=now,
            model=model,
            reasoning_effort=reasoning_effort,
            metadata=metadata or {},
        )

        self._save(checkpoint)
        return checkpoint

    def _save(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to disk."""
        path = self._checkpoint_path(checkpoint.checkpoint_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)

    def update(
        self,
        checkpoint_id: str,
        iteration: int | None = None,
        history: list[str] | None = None,
        completed: bool | None = None,
        completion_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint | None:
        """
        Update an existing checkpoint.

        Returns:
            Updated checkpoint, or None if not found
        """
        checkpoint = self.load(checkpoint_id)
        if not checkpoint:
            return None

        checkpoint.updated_at = datetime.now().isoformat()

        if iteration is not None:
            checkpoint.iteration = iteration
        if history is not None:
            checkpoint.history = history
        if completed is not None:
            checkpoint.completed = completed
        if completion_reason is not None:
            checkpoint.completion_reason = completion_reason
        if metadata is not None:
            checkpoint.metadata.update(metadata)

        self._save(checkpoint)
        return checkpoint

    def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint by ID."""
        path = self._checkpoint_path(checkpoint_id)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return Checkpoint.from_dict(data)

    def get_latest(self, loop_id: str) -> Checkpoint | None:
        """
        Get the latest checkpoint for a loop.

        Args:
            loop_id: The loop identifier

        Returns:
            Latest checkpoint, or None if none found
        """
        checkpoints = self.list_checkpoints(loop_id=loop_id)
        if not checkpoints:
            return None

        # Already sorted by updated_at descending
        latest_id = checkpoints[0]["checkpoint_id"]
        return self.load(latest_id)

    def get_incomplete(self, loop_id: str | None = None) -> list[Checkpoint]:
        """
        Get all incomplete checkpoints.

        Args:
            loop_id: Optional filter by loop ID

        Returns:
            List of incomplete checkpoints
        """
        result = []
        for path in self.base_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data.get("completed", False):
                    continue

                if loop_id and data.get("loop_id") != loop_id:
                    continue

                result.append(Checkpoint.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by updated_at descending
        result.sort(key=lambda x: x.updated_at, reverse=True)
        return result

    def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        path = self._checkpoint_path(checkpoint_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def cleanup(self, loop_id: str | None = None, keep_latest: int = 5) -> int:
        """
        Clean up old checkpoints, keeping only the latest N.

        Args:
            loop_id: Optional filter by loop ID
            keep_latest: Number of checkpoints to keep per loop

        Returns:
            Number of checkpoints deleted
        """
        # Group by loop_id
        by_loop: dict[str, list[dict]] = {}
        for cp in self.list_checkpoints():
            lid = cp["loop_id"]
            if loop_id and lid != loop_id:
                continue
            if lid not in by_loop:
                by_loop[lid] = []
            by_loop[lid].append(cp)

        deleted = 0
        for lid, checkpoints in by_loop.items():
            # Sort by updated_at descending
            checkpoints.sort(key=lambda x: x["updated_at"], reverse=True)

            # Delete old ones
            for cp in checkpoints[keep_latest:]:
                if self.delete(cp["checkpoint_id"]):
                    deleted += 1

        return deleted

    def list_checkpoints(self, loop_id: str | None = None) -> list[dict[str, Any]]:
        """
        List all checkpoints.

        Args:
            loop_id: Optional filter by loop ID

        Returns:
            List of checkpoint summaries
        """
        checkpoints = []
        for path in self.base_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if loop_id and data.get("loop_id") != loop_id:
                    continue

                checkpoints.append({
                    "checkpoint_id": data["checkpoint_id"],
                    "loop_id": data["loop_id"],
                    "iteration": data["iteration"],
                    "max_iterations": data.get("max_iterations"),
                    "completed": data.get("completed", False),
                    "completion_reason": data.get("completion_reason"),
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                })
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by updated_at descending
        checkpoints.sort(key=lambda x: x["updated_at"], reverse=True)
        return checkpoints


class CheckpointedRalph:
    """
    Ralph Wiggum with automatic checkpointing.

    Usage:
        from copex import Copex
        from copex.checkpoint import CheckpointedRalph, CheckpointStore

        store = CheckpointStore()

        async with Copex() as copex:
            ralph = CheckpointedRalph(copex, store, loop_id="my-project")

            # Start or resume a loop
            result = await ralph.loop(
                prompt="Build a REST API",
                completion_promise="DONE",
            )
    """

    def __init__(
        self,
        client: Any,
        store: CheckpointStore,
        loop_id: str,
    ):
        """
        Initialize checkpointed Ralph.

        Args:
            client: Copex client
            store: Checkpoint store
            loop_id: Identifier for this loop
        """
        self.client = client
        self.store = store
        self.loop_id = loop_id
        self._checkpoint: Checkpoint | None = None

    async def loop(
        self,
        prompt: str,
        *,
        max_iterations: int | None = None,
        completion_promise: str | None = None,
        resume: bool = True,
    ) -> Checkpoint:
        """
        Run a checkpointed Ralph loop.

        Args:
            prompt: Task prompt
            max_iterations: Maximum iterations
            completion_promise: Text that signals completion
            resume: Whether to resume from last checkpoint if available

        Returns:
            Final checkpoint state
        """
        from copex.ralph import RalphWiggum, RalphConfig

        # Check for existing checkpoint to resume
        if resume:
            existing = self.store.get_latest(self.loop_id)
            if existing and not existing.completed:
                self._checkpoint = existing
                start_iteration = existing.iteration
                history = existing.history
            else:
                self._checkpoint = None
                start_iteration = 0
                history = []
        else:
            start_iteration = 0
            history = []

        # Create new checkpoint if needed
        if not self._checkpoint:
            model = self.client.config.model.value if hasattr(self.client.config.model, 'value') else str(self.client.config.model)
            reasoning = self.client.config.reasoning_effort.value if hasattr(self.client.config.reasoning_effort, 'value') else str(self.client.config.reasoning_effort)

            self._checkpoint = self.store.create(
                loop_id=self.loop_id,
                prompt=prompt,
                max_iterations=max_iterations,
                completion_promise=completion_promise,
                model=model,
                reasoning_effort=reasoning,
            )

        # Create Ralph with config
        config = RalphConfig(
            max_iterations=max_iterations,
            completion_promise=completion_promise,
        )
        ralph = RalphWiggum(self.client, config)

        # Set up iteration callback to save checkpoints
        def on_iteration(iteration: int, response: str) -> None:
            history.append(response)
            self.store.update(
                self._checkpoint.checkpoint_id,
                iteration=iteration,
                history=history,
            )

        def on_complete(state) -> None:
            self.store.update(
                self._checkpoint.checkpoint_id,
                iteration=state.iteration,
                completed=True,
                completion_reason=state.completion_reason,
                history=history,
            )

        # Run the loop
        await ralph.loop(
            prompt,
            max_iterations=max_iterations,
            completion_promise=completion_promise,
            on_iteration=on_iteration,
            on_complete=on_complete,
        )

        # Return updated checkpoint
        return self.store.load(self._checkpoint.checkpoint_id)
