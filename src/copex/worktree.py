"""Git worktree isolation for fleet tasks.

Provides WorktreeManager to create isolated git worktrees per fleet task,
preventing concurrent tasks from cross-contaminating each other's uncommitted
changes.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

WORKTREE_PREFIX = ".copex-worktree-"


@dataclass
class WorktreeResult:
    """Result of worktree lifecycle operations."""

    success: bool
    worktree_path: Path | None = None
    commit_hash: str | None = None
    error: str | None = None


@dataclass
class GitFinalizeResult:
    """Result of a git finalize operation."""

    success: bool
    commit_hash: str | None = None
    files_staged: int = 0
    error: str | None = None


class GitFinalizer:
    """Stages and commits changes after fleet execution."""

    def __init__(self, message: str = "fleet: auto-commit after fleet run") -> None:
        self._message = message

    @staticmethod
    def is_git_repo() -> bool:
        """Check if the current directory is inside a git repository."""
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0

    @staticmethod
    def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def _has_changes(self) -> bool:
        """Check if there are any staged or unstaged changes."""
        status = self._run_git("status", "--porcelain")
        return bool(status.stdout.strip())

    def _stage_all(self) -> int:
        """Stage all changes and return count of staged files."""
        self._run_git("add", "-A")
        result = self._run_git("diff", "--cached", "--name-only")
        files = [f for f in result.stdout.strip().splitlines() if f]
        return len(files)

    def _commit(self) -> str | None:
        """Commit staged changes and return the commit hash."""
        result = self._run_git("commit", "-m", self._message)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git commit failed")
        hash_result = self._run_git("rev-parse", "--short", "HEAD")
        return hash_result.stdout.strip() or None

    async def finalize(self) -> GitFinalizeResult:
        """Stage all changes and commit."""
        loop = asyncio.get_running_loop()
        try:
            has_changes = await loop.run_in_executor(None, self._has_changes)
            if not has_changes:
                return GitFinalizeResult(success=True, files_staged=0)

            files_staged = await loop.run_in_executor(None, self._stage_all)
            if files_staged == 0:
                return GitFinalizeResult(success=True, files_staged=0)

            commit_hash = await loop.run_in_executor(None, self._commit)
            return GitFinalizeResult(
                success=True,
                commit_hash=commit_hash,
                files_staged=files_staged,
            )
        except Exception as exc:  # Catch-all: any git failure returns error result
            return GitFinalizeResult(success=False, error=str(exc))


@dataclass
class WorktreeManager:
    """Manages git worktree lifecycle for a single fleet task.

    Usage:
        mgr = WorktreeManager(repo_root=Path("."))
        result = mgr.create_worktree()
        # ... run task in result.worktree_path ...
        merge_result = mgr.merge_back(message="task: done")
        mgr.cleanup_worktree()
    """

    repo_root: Path
    worktree_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _worktree_path: Path | None = field(default=None, init=False, repr=False)
    _commit_hash: str | None = field(default=None, init=False, repr=False)

    @property
    def worktree_name(self) -> str:
        return f"{WORKTREE_PREFIX}{self.worktree_id}"

    @property
    def worktree_path(self) -> Path | None:
        return self._worktree_path

    # ── Validation ───────────────────────────────────────────────────

    @staticmethod
    def is_git_repo(path: Path) -> bool:
        """Check if *path* is inside a git repository."""
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            cwd=str(path),
            check=False,
        )
        return result.returncode == 0

    @staticmethod
    def get_repo_root(path: Path) -> Path | None:
        """Return the repository root or None."""
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            cwd=str(path),
            check=False,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
        return None

    @staticmethod
    def current_branch(path: Path) -> str | None:
        """Return current branch name or None if detached."""
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(path),
            check=False,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            return branch if branch != "HEAD" else None
        return None

    @staticmethod
    def has_uncommitted_changes(path: Path) -> bool:
        """Return True if there are staged or unstaged changes."""
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(path),
            check=False,
        )
        return bool(result.stdout.strip())

    # ── Worktree lifecycle ───────────────────────────────────────────

    def create_worktree(self, *, ref: str = "HEAD") -> WorktreeResult:
        """Create a new git worktree for this task.

        Args:
            ref: The git ref to base the worktree on. Defaults to HEAD.

        Returns:
            WorktreeResult with the worktree path on success.
        """
        if not self.is_git_repo(self.repo_root):
            return WorktreeResult(
                success=False,
                error=f"Not a git repository: {self.repo_root}",
            )

        wt_path = self.repo_root / self.worktree_name
        if wt_path.exists():
            return WorktreeResult(
                success=False,
                error=f"Worktree path already exists: {wt_path}",
            )

        logger.info("Creating worktree %s at %s (ref=%s)", self.worktree_id, wt_path, ref)

        # Create a detached worktree from the given ref
        result = subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt_path), ref],
            capture_output=True,
            text=True,
            cwd=str(self.repo_root),
            check=False,
        )

        if result.returncode != 0:
            return WorktreeResult(
                success=False,
                error=f"git worktree add failed: {result.stderr.strip()}",
            )

        self._worktree_path = wt_path
        logger.info("Worktree created: %s", wt_path)
        return WorktreeResult(success=True, worktree_path=wt_path)

    def commit_in_worktree(self, message: str) -> WorktreeResult:
        """Stage all changes in the worktree and commit.

        Returns:
            WorktreeResult with the commit hash on success.
        """
        if self._worktree_path is None:
            return WorktreeResult(success=False, error="No worktree created yet")

        wt = str(self._worktree_path)

        # Check if there are any changes to commit
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=wt,
            check=False,
        )
        if not status.stdout.strip():
            logger.info("Worktree %s: no changes to commit", self.worktree_id)
            return WorktreeResult(
                success=True,
                worktree_path=self._worktree_path,
            )

        # Stage everything
        result = subprocess.run(
            ["git", "add", "-A"],
            capture_output=True,
            text=True,
            cwd=wt,
            check=False,
        )
        if result.returncode != 0:
            return WorktreeResult(
                success=False,
                error=f"git add failed: {result.stderr.strip()}",
            )

        # Commit
        result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True,
            cwd=wt,
            check=False,
        )
        if result.returncode != 0:
            return WorktreeResult(
                success=False,
                error=f"git commit failed: {result.stderr.strip()}",
            )

        # Get the commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=wt,
            check=False,
        )
        commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else None
        self._commit_hash = commit_hash

        logger.info("Worktree %s: committed %s", self.worktree_id, commit_hash)
        return WorktreeResult(
            success=True,
            worktree_path=self._worktree_path,
            commit_hash=commit_hash,
        )

    def merge_back(self, *, message: str | None = None) -> WorktreeResult:
        """Cherry-pick the worktree commit back to the main branch.

        If there's a commit in the worktree, cherry-pick it onto the branch
        that was checked out in the main repo root.

        Returns:
            WorktreeResult with success status.
        """
        if self._commit_hash is None:
            return WorktreeResult(
                success=True,
                error="No commit to merge back (no changes were made)",
            )

        root = str(self.repo_root)

        logger.info(
            "Cherry-picking %s back to main repo at %s",
            self._commit_hash,
            self.repo_root,
        )

        result = subprocess.run(
            ["git", "cherry-pick", self._commit_hash],
            capture_output=True,
            text=True,
            cwd=root,
            check=False,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Attempt to abort the cherry-pick on conflict
            if "conflict" in stderr.lower():
                logger.warning("Cherry-pick conflict for %s, aborting", self._commit_hash)
                subprocess.run(
                    ["git", "cherry-pick", "--abort"],
                    capture_output=True,
                    text=True,
                    cwd=root,
                    check=False,
                )
                return WorktreeResult(
                    success=False,
                    commit_hash=self._commit_hash,
                    error=f"Cherry-pick conflict: {stderr}",
                )
            return WorktreeResult(
                success=False,
                commit_hash=self._commit_hash,
                error=f"Cherry-pick failed: {stderr}",
            )

        logger.info("Cherry-pick successful: %s", self._commit_hash)
        return WorktreeResult(
            success=True,
            commit_hash=self._commit_hash,
        )

    def cleanup_worktree(self) -> WorktreeResult:
        """Remove the worktree and prune git's internal list.

        Always safe to call, even if the worktree was never created.
        """
        if self._worktree_path is None:
            return WorktreeResult(success=True)

        wt = str(self._worktree_path)
        root = str(self.repo_root)

        logger.info("Cleaning up worktree %s at %s", self.worktree_id, wt)

        # Force remove the worktree
        result = subprocess.run(
            ["git", "worktree", "remove", "--force", wt],
            capture_output=True,
            text=True,
            cwd=root,
            check=False,
        )

        if result.returncode != 0:
            logger.warning(
                "git worktree remove failed (will prune): %s", result.stderr.strip()
            )
            # Fallback: manually remove + prune
            import shutil

            if self._worktree_path.exists():
                shutil.rmtree(self._worktree_path, ignore_errors=True)
            subprocess.run(
                ["git", "worktree", "prune"],
                capture_output=True,
                text=True,
                cwd=root,
                check=False,
            )

        self._worktree_path = None
        return WorktreeResult(success=True)


# ── High-level helpers for fleet integration ─────────────────────────


async def run_task_in_worktree(
    repo_root: Path,
    task_id: str,
    run_fn: Callable[[Path], Awaitable[None]],
    *,
    commit_message: str | None = None,
) -> tuple[WorktreeManager, WorktreeResult | None]:
    """Create a worktree, run *run_fn(worktree_path)*, merge back, clean up.

    Args:
        repo_root: Repository root directory.
        task_id: Fleet task ID (used for logging).
        run_fn: Async callable receiving the worktree Path; should run the task.
        commit_message: Optional commit message. Defaults to task-based message.

    Returns:
        (manager, merge_result) — merge_result is None if the task itself failed.
    """
    import asyncio

    mgr = WorktreeManager(repo_root=repo_root)
    msg = commit_message or f"fleet({task_id}): apply changes"

    create = mgr.create_worktree()
    if not create.success:
        logger.error("Failed to create worktree for %s: %s", task_id, create.error)
        return mgr, create

    assert mgr.worktree_path is not None
    loop = asyncio.get_running_loop()

    try:
        # Run the actual task in the worktree
        await run_fn(mgr.worktree_path)

        # Commit changes made by the task
        commit_result = await loop.run_in_executor(
            None, lambda: mgr.commit_in_worktree(msg)
        )
        if not commit_result.success and commit_result.error != "No changes to commit":
            logger.error("Commit in worktree failed for %s: %s", task_id, commit_result.error)
            return mgr, commit_result

        # Cherry-pick back to main
        merge_result = await loop.run_in_executor(None, mgr.merge_back)
        return mgr, merge_result

    except Exception:
        logger.exception("Task %s failed in worktree", task_id)
        raise

    finally:
        await loop.run_in_executor(None, mgr.cleanup_worktree)
