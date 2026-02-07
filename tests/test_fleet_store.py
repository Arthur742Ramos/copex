from __future__ import annotations

import pytest

from unittest.mock import AsyncMock, patch

from copex.client import Response
from copex.config import CopexConfig
from copex.fleet import Fleet, FleetConfig, FleetTask
from copex.fleet_store import FleetStore


def _make_mock_copex(response=None, error=None):
    """Return a mock Copex that acts as an async context manager."""
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    if error:
        mock.send = AsyncMock(side_effect=error)
    else:
        mock.send = AsyncMock(return_value=response or Response(content="done"))
    return mock


# ---------------------------------------------------------------------------
# TestFleetStoreBasic
# ---------------------------------------------------------------------------


class TestFleetStoreBasic:
    def test_create_and_get_run(self, tmp_path):
        """Create a run and retrieve it."""
        store = FleetStore(tmp_path / "test.db")
        run_id = store.create_run(config={"max_concurrent": 3})
        run = store.get_run(run_id)
        assert run is not None
        assert run.run_id == run_id
        assert run.status == "running"
        store.close()

    def test_add_and_get_task(self, tmp_path):
        """Add a task and retrieve it."""
        store = FleetStore(tmp_path / "test.db")
        run_id = store.create_run()
        store.add_task(run_id, "task-1", "Do something", depends_on=["other"])
        task = store.get_task(run_id, "task-1")
        assert task is not None
        assert task.task_id == "task-1"
        assert task.prompt == "Do something"
        assert task.depends_on == ["other"]
        assert task.status == "pending"
        store.close()

    def test_get_tasks_returns_all(self, tmp_path):
        """Get all tasks for a run."""
        store = FleetStore(tmp_path / "test.db")
        run_id = store.create_run()
        store.add_task(run_id, "a", "Task A")
        store.add_task(run_id, "b", "Task B")
        store.add_task(run_id, "c", "Task C")
        tasks = store.get_tasks(run_id)
        assert len(tasks) == 3
        store.close()

    def test_nonexistent_run_returns_none(self, tmp_path):
        store = FleetStore(tmp_path / "test.db")
        assert store.get_run("nonexistent") is None
        store.close()

    def test_nonexistent_task_returns_none(self, tmp_path):
        store = FleetStore(tmp_path / "test.db")
        run_id = store.create_run()
        assert store.get_task(run_id, "nonexistent") is None
        store.close()

    def test_context_manager(self, tmp_path):
        with FleetStore(tmp_path / "test.db") as store:
            run_id = store.create_run()
            assert store.get_run(run_id) is not None

    def test_list_runs(self, tmp_path):
        store = FleetStore(tmp_path / "test.db")
        store.create_run()
        store.create_run()
        store.create_run()
        runs = store.list_runs()
        assert len(runs) == 3
        store.close()


# ---------------------------------------------------------------------------
# TestFleetStoreLifecycle
# ---------------------------------------------------------------------------


class TestFleetStoreLifecycle:
    def test_task_status_transitions(self, tmp_path):
        """Track task through its lifecycle."""
        store = FleetStore(tmp_path / "test.db")
        run_id = store.create_run()
        store.add_task(run_id, "t1", "Task 1")

        store.update_task_status(run_id, "t1", "running")
        task = store.get_task(run_id, "t1")
        assert task.status == "running"
        assert task.started_at is not None

        store.record_result(run_id, "t1", success=True, content="Result", duration_ms=100)
        task = store.get_task(run_id, "t1")
        assert task.status == "done"
        assert task.result_content == "Result"
        assert task.duration_ms == 100
        assert task.completed_at is not None
        store.close()

    def test_failed_task_records_error(self, tmp_path):
        store = FleetStore(tmp_path / "test.db")
        run_id = store.create_run()
        store.add_task(run_id, "t1", "Fail task")
        store.record_result(run_id, "t1", success=False, error="boom", duration_ms=50)
        task = store.get_task(run_id, "t1")
        assert task.status == "failed"
        assert task.error_text == "boom"
        store.close()

    def test_complete_run_counts(self, tmp_path):
        store = FleetStore(tmp_path / "test.db")
        run_id = store.create_run()
        store.add_task(run_id, "a", "A")
        store.add_task(run_id, "b", "B")
        store.add_task(run_id, "c", "C")
        store.record_result(run_id, "a", success=True, content="ok")
        store.record_result(run_id, "b", success=True, content="ok")
        store.record_result(run_id, "c", success=False, error="bad")
        store.complete_run(run_id)

        run = store.get_run(run_id)
        assert run.status == "failed"  # has failures
        assert run.total_tasks == 3
        assert run.succeeded == 2
        assert run.failed == 1
        store.close()

    def test_complete_run_all_success(self, tmp_path):
        store = FleetStore(tmp_path / "test.db")
        run_id = store.create_run()
        store.add_task(run_id, "a", "A")
        store.record_result(run_id, "a", success=True, content="ok")
        store.complete_run(run_id)

        run = store.get_run(run_id)
        assert run.status == "completed"
        assert run.succeeded == 1
        assert run.failed == 0
        store.close()

    def test_get_incomplete_tasks(self, tmp_path):
        store = FleetStore(tmp_path / "test.db")
        run_id = store.create_run()
        store.add_task(run_id, "a", "A")
        store.add_task(run_id, "b", "B")
        store.add_task(run_id, "c", "C")
        store.record_result(run_id, "a", success=True, content="ok")

        incomplete = store.get_incomplete_tasks(run_id)
        assert len(incomplete) == 2
        assert {t.task_id for t in incomplete} == {"b", "c"}
        store.close()

    def test_get_completed_task_ids(self, tmp_path):
        store = FleetStore(tmp_path / "test.db")
        run_id = store.create_run()
        store.add_task(run_id, "a", "A")
        store.add_task(run_id, "b", "B")
        store.record_result(run_id, "a", success=True, content="ok")

        completed = store.get_completed_task_ids(run_id)
        assert completed == {"a"}
        store.close()


# ---------------------------------------------------------------------------
# TestFleetStoreIsolation
# ---------------------------------------------------------------------------


class TestFleetStoreIsolation:
    def test_multiple_runs_isolated(self, tmp_path):
        """Tasks from different runs don't interfere."""
        store = FleetStore(tmp_path / "test.db")
        run1 = store.create_run()
        run2 = store.create_run()
        store.add_task(run1, "shared-name", "Run 1 task")
        store.add_task(run2, "shared-name", "Run 2 task")

        t1 = store.get_task(run1, "shared-name")
        t2 = store.get_task(run2, "shared-name")
        assert t1.prompt == "Run 1 task"
        assert t2.prompt == "Run 2 task"
        store.close()


# ---------------------------------------------------------------------------
# TestFleetWithPersistence
# ---------------------------------------------------------------------------


class TestFleetWithPersistence:
    @pytest.mark.asyncio
    async def test_run_persists_to_db(self, tmp_path):
        """Fleet.run() with db_path persists task state."""
        db = tmp_path / "fleet.db"
        mock_copex = _make_mock_copex(Response(content="result"))

        fleet = Fleet(db_path=db)
        fleet.add("Task A", task_id="a")
        fleet.add("Task B", task_id="b")

        with patch("copex.fleet.CopilotClient", None), \
             patch("copex.fleet.Copex", return_value=mock_copex):
            results = await fleet.run()

        assert all(r.success for r in results)

        # Verify data was persisted
        store = FleetStore(db)
        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0].status == "completed"
        assert runs[0].succeeded == 2

        tasks = store.get_tasks(runs[0].run_id)
        assert len(tasks) == 2
        assert all(t.status == "done" for t in tasks)
        store.close()

        # Clean up fleet store
        await fleet.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_run_without_db_unchanged(self):
        """Fleet.run() without db_path works as before."""
        mock_copex = _make_mock_copex()
        fleet = Fleet()
        fleet.add("Task", task_id="t1")

        with patch("copex.fleet.CopilotClient", None), \
             patch("copex.fleet.Copex", return_value=mock_copex):
            results = await fleet.run()

        assert len(results) == 1
        assert results[0].success
        assert fleet._store is None


# ---------------------------------------------------------------------------
# TestFleetResume
# ---------------------------------------------------------------------------


class TestFleetResume:
    @pytest.mark.asyncio
    async def test_resume_skips_completed_tasks(self, tmp_path):
        """Resume should only run incomplete tasks."""
        db = tmp_path / "fleet.db"

        # Simulate a partial run: create a run with 3 tasks, mark 1 as done
        store = FleetStore(db)
        run_id = store.create_run(config={"max_concurrent": 5, "timeout": 600, "fail_fast": False})
        store.add_task(run_id, "a", "Task A")
        store.add_task(run_id, "b", "Task B", depends_on=["a"])
        store.add_task(run_id, "c", "Task C")
        store.record_result(run_id, "a", success=True, content="done A")
        store.close()

        # Resume should only run b and c (a is already done)
        mock_copex = _make_mock_copex(Response(content="resumed"))

        with patch("copex.fleet.CopilotClient", None), \
             patch("copex.fleet.Copex", return_value=mock_copex):
            results = await Fleet.resume(db, run_id)

        assert len(results) == 2
        assert all(r.success for r in results)
        task_ids = {r.task_id for r in results}
        assert task_ids == {"b", "c"}

    @pytest.mark.asyncio
    async def test_resume_nonexistent_run_raises(self, tmp_path):
        db = tmp_path / "fleet.db"
        # Create the DB file
        FleetStore(db).close()

        with pytest.raises(ValueError, match="Run not found"):
            await Fleet.resume(db, "nonexistent")

    @pytest.mark.asyncio
    async def test_resume_all_done_returns_empty(self, tmp_path):
        """If all tasks are done, resume returns empty list."""
        db = tmp_path / "fleet.db"
        store = FleetStore(db)
        run_id = store.create_run()
        store.add_task(run_id, "a", "Task A")
        store.record_result(run_id, "a", success=True, content="ok")
        store.close()

        results = await Fleet.resume(db, run_id)
        assert results == []

    @pytest.mark.asyncio
    async def test_resume_completes_run_in_store(self, tmp_path):
        """After resume, the run should be marked completed in the DB."""
        db = tmp_path / "fleet.db"
        store = FleetStore(db)
        run_id = store.create_run()
        store.add_task(run_id, "a", "Task A")
        store.close()

        mock_copex = _make_mock_copex(Response(content="ok"))

        with patch("copex.fleet.CopilotClient", None), \
             patch("copex.fleet.Copex", return_value=mock_copex):
            await Fleet.resume(db, run_id)

        store = FleetStore(db)
        run = store.get_run(run_id)
        assert run.status == "completed"
        assert run.succeeded == 1
        store.close()
