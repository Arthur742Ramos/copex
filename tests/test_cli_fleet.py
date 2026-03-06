from __future__ import annotations

import asyncio
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from copex.cli_fleet import _run_fleet
from copex.config import CopexConfig
from copex.fleet import FleetTask


class _FakeLive:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __enter__(self) -> "_FakeLive":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def update(self, *_args: object, **_kwargs: object) -> None:
        return None


class _FakeRepoMap:
    def __init__(self, _root: Path) -> None:
        pass

    def refresh(self, *, force: bool = False) -> None:
        return None

    def relevant_context(self, *_args: object, **_kwargs: object) -> str:
        return ""


@pytest.mark.asyncio
async def test_run_fleet_cleans_up_worktrees_on_failure(monkeypatch, tmp_path: Path) -> None:
    cleanup_calls: list[str] = []

    class _FakeWorktreeManager:
        def __init__(self, *, repo_root: Path) -> None:
            self.repo_root = repo_root
            self.worktree_path = repo_root / "wt-task-1"

        @staticmethod
        def get_repo_root(path: Path) -> Path:
            return path

        @staticmethod
        def has_uncommitted_changes(_root: Path) -> bool:
            return False

        def create_worktree(self):
            self.worktree_path.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(success=True, worktree_path=self.worktree_path, error=None)

        def cleanup_worktree(self):
            cleanup_calls.append(str(self.worktree_path))
            return SimpleNamespace(success=True)

    class _FailingFleet:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def __aenter__(self) -> "_FailingFleet":
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        def add(self, *_args, **_kwargs) -> None:
            return None

        async def run(self, **_kwargs):
            raise RuntimeError("fleet exploded")

    repo_map_module = ModuleType("copex.repo_map")
    repo_map_module.RepoMap = _FakeRepoMap  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "copex.repo_map", repo_map_module)
    monkeypatch.setattr("copex.worktree.WorktreeManager", _FakeWorktreeManager)
    monkeypatch.setattr("copex.fleet.Fleet", _FailingFleet)
    monkeypatch.setattr("copex.cli_fleet.Live", _FakeLive)

    config = CopexConfig(working_directory=str(tmp_path))
    tasks = [FleetTask(id="task-1", prompt="Do it")]

    with pytest.raises(RuntimeError, match="fleet exploded"):
        await _run_fleet(
            config=config,
            prompts=[],
            file=None,
            config_file=None,
            max_concurrent=1,
            fail_fast=False,
            shared_context=None,
            timeout=30.0,
            tasks_override=tasks,
            git_finalize=False,
            worktree=True,
        )

    assert cleanup_calls == [str(tmp_path / "wt-task-1")]


@pytest.mark.asyncio
async def test_run_fleet_build_check_uses_argv_without_shell(
    monkeypatch, tmp_path: Path
) -> None:
    class _SuccessfulFleet:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def __aenter__(self) -> "_SuccessfulFleet":
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        def add(self, *_args, **_kwargs) -> None:
            return None

        async def run(self, **_kwargs):
            return [
                SimpleNamespace(
                    task_id="task-1",
                    success=True,
                    response=SimpleNamespace(content="ok"),
                    error=None,
                )
            ]

    repo_map_module = ModuleType("copex.repo_map")
    repo_map_module.RepoMap = _FakeRepoMap  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "copex.repo_map", repo_map_module)
    monkeypatch.setattr("copex.fleet.Fleet", _SuccessfulFleet)
    monkeypatch.setattr("copex.cli_fleet.Live", _FakeLive)

    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    run_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _fake_run(*args: object, **kwargs: object):
        run_calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("copex.cli_fleet.subprocess.run", _fake_run)

    config = CopexConfig(working_directory=str(tmp_path))
    tasks = [FleetTask(id="task-1", prompt="Build it")]

    await _run_fleet(
        config=config,
        prompts=[],
        file=None,
        config_file=None,
        max_concurrent=1,
        fail_fast=False,
        shared_context=None,
        timeout=30.0,
        tasks_override=tasks,
        git_finalize=False,
        retry=1,
    )

    assert run_calls
    args, kwargs = run_calls[0]
    assert args[0] == ["npm", "run", "build"]
    assert kwargs.get("shell", False) is False
