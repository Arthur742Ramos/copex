"""Git finalization utilities for Copex."""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass


@dataclass
class GitFinalizeResult:
    """Result of a git finalize operation."""

    success: bool
    commit_hash: str | None = None
    files_staged: int = 0
    error: str | None = None


class GitFinalizer:
    """Stages and commits changes after fleet execution.

    Usage:
        finalizer = GitFinalizer(message="fleet: completed tasks")
        result = await finalizer.finalize()
    """

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
        """Stage all changes and commit.

        Runs git operations in a thread to avoid blocking the event loop.
        """
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
