"""Tests for GitFinalizer (git.py)."""

from __future__ import annotations

import asyncio
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from copex.git import GitFinalizer, GitFinalizeResult


class TestGitFinalizeResult:
    """Tests for GitFinalizeResult dataclass."""

    def test_default_values(self):
        result = GitFinalizeResult(success=True)
        assert result.success is True
        assert result.commit_hash is None
        assert result.files_staged == 0
        assert result.error is None

    def test_full_result(self):
        result = GitFinalizeResult(
            success=True,
            commit_hash="abc1234",
            files_staged=5,
        )
        assert result.commit_hash == "abc1234"
        assert result.files_staged == 5

    def test_error_result(self):
        result = GitFinalizeResult(
            success=False,
            error="Not a git repository",
        )
        assert result.success is False
        assert result.error == "Not a git repository"


class TestGitFinalizer:
    """Tests for GitFinalizer class."""

    def test_default_message(self):
        finalizer = GitFinalizer()
        assert "fleet" in finalizer._message

    def test_custom_message(self):
        finalizer = GitFinalizer(message="my commit message")
        assert finalizer._message == "my commit message"


class TestGitFinalizerHasChanges:
    """Tests for _has_changes method."""

    def test_has_changes_true(self):
        finalizer = GitFinalizer()
        mock_result = MagicMock()
        mock_result.stdout = " M src/file.py\n"

        with patch.object(GitFinalizer, "_run_git", return_value=mock_result):
            assert finalizer._has_changes() is True

    def test_has_changes_false(self):
        finalizer = GitFinalizer()
        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch.object(GitFinalizer, "_run_git", return_value=mock_result):
            assert finalizer._has_changes() is False

    def test_has_changes_whitespace_only(self):
        finalizer = GitFinalizer()
        mock_result = MagicMock()
        mock_result.stdout = "   \n\n  "

        with patch.object(GitFinalizer, "_run_git", return_value=mock_result):
            assert finalizer._has_changes() is False


class TestGitFinalizerStageAll:
    """Tests for _stage_all method."""

    def test_stage_all_returns_count(self):
        finalizer = GitFinalizer()

        def mock_git(*args):
            result = MagicMock()
            if "diff" in args:
                result.stdout = "file1.py\nfile2.py\nfile3.py\n"
            else:
                result.stdout = ""
            return result

        with patch.object(GitFinalizer, "_run_git", side_effect=mock_git):
            count = finalizer._stage_all()
            assert count == 3

    def test_stage_all_empty(self):
        finalizer = GitFinalizer()

        def mock_git(*args):
            result = MagicMock()
            result.stdout = ""
            return result

        with patch.object(GitFinalizer, "_run_git", side_effect=mock_git):
            count = finalizer._stage_all()
            assert count == 0


class TestGitFinalizerCommit:
    """Tests for _commit method."""

    def test_commit_success(self):
        finalizer = GitFinalizer(message="test commit")

        def mock_git(*args):
            result = MagicMock()
            if "commit" in args:
                result.returncode = 0
                result.stdout = "[main abc1234] test commit"
            elif "rev-parse" in args:
                result.stdout = "abc1234"
            return result

        with patch.object(GitFinalizer, "_run_git", side_effect=mock_git):
            commit_hash = finalizer._commit()
            assert commit_hash == "abc1234"

    def test_commit_failure(self):
        finalizer = GitFinalizer(message="test commit")

        def mock_git(*args):
            result = MagicMock()
            if "commit" in args:
                result.returncode = 1
                result.stderr = "nothing to commit"
            return result

        with patch.object(GitFinalizer, "_run_git", side_effect=mock_git):
            with pytest.raises(RuntimeError, match="nothing to commit"):
                finalizer._commit()


class TestGitFinalizerFinalize:
    """Tests for async finalize method."""

    @pytest.mark.asyncio
    async def test_finalize_no_changes(self):
        finalizer = GitFinalizer()

        with patch.object(finalizer, "_has_changes", return_value=False):
            result = await finalizer.finalize()

        assert result.success is True
        assert result.files_staged == 0
        assert result.commit_hash is None

    @pytest.mark.asyncio
    async def test_finalize_with_changes(self):
        finalizer = GitFinalizer(message="test")

        with (
            patch.object(finalizer, "_has_changes", return_value=True),
            patch.object(finalizer, "_stage_all", return_value=3),
            patch.object(finalizer, "_commit", return_value="def5678"),
        ):
            result = await finalizer.finalize()

        assert result.success is True
        assert result.files_staged == 3
        assert result.commit_hash == "def5678"

    @pytest.mark.asyncio
    async def test_finalize_stages_but_nothing_to_commit(self):
        """Edge case: has_changes=True but stage_all returns 0."""
        finalizer = GitFinalizer()

        with (
            patch.object(finalizer, "_has_changes", return_value=True),
            patch.object(finalizer, "_stage_all", return_value=0),
        ):
            result = await finalizer.finalize()

        assert result.success is True
        assert result.files_staged == 0

    @pytest.mark.asyncio
    async def test_finalize_error_handling(self):
        finalizer = GitFinalizer()

        with patch.object(
            finalizer, "_has_changes", side_effect=RuntimeError("not a git repo")
        ):
            result = await finalizer.finalize()

        assert result.success is False
        assert "not a git repo" in result.error


class TestGitFinalizerIntegration:
    """Integration-style tests using mocked subprocess."""

    @pytest.mark.asyncio
    async def test_full_flow_with_subprocess_mock(self):
        finalizer = GitFinalizer(message="fleet: test commit")

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""

            if cmd == ["git", "status", "--porcelain"]:
                result.stdout = " M file1.py\n"
            elif cmd == ["git", "add", "-A"]:
                result.stdout = ""
            elif cmd == ["git", "diff", "--cached", "--name-only"]:
                result.stdout = "file1.py\n"
            elif cmd[:2] == ["git", "commit"]:
                result.stdout = "[main abc1234] fleet: test commit"
            elif cmd == ["git", "rev-parse", "--short", "HEAD"]:
                result.stdout = "abc1234"
            else:
                result.stdout = ""

            return result

        with patch("subprocess.run", side_effect=mock_run):
            result = await finalizer.finalize()

        assert result.success is True
        assert result.commit_hash == "abc1234"
        assert result.files_staged == 1


class TestGitFinalizerNotAGitRepo:
    """Test behavior when not in a git repository."""

    @pytest.mark.asyncio
    async def test_not_a_git_repo(self):
        finalizer = GitFinalizer()

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 128
            result.stdout = ""
            result.stderr = "fatal: not a git repository"
            return result

        with patch("subprocess.run", side_effect=mock_run):
            # _has_changes will return False because stdout is empty
            result = await finalizer.finalize()

        # Should succeed with no changes (graceful handling)
        assert result.success is True
        assert result.files_staged == 0
