"""
Fleet Store - SQLite-backed persistence for fleet task orchestration.

Enables:
- Persistent task state tracking across crashes
- Resume interrupted fleet runs
- Audit trail of fleet executions
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TaskRecord:
    """Persistent record of a fleet task."""

    task_id: str
    run_id: str
    prompt: str
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, done, failed, blocked, cancelled
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: float = 0
    result_content: str | None = None
    error_text: str | None = None


@dataclass
class RunRecord:
    """Persistent record of a fleet run."""

    run_id: str
    created_at: str
    completed_at: str | None = None
    status: str = "running"  # running, completed, failed, cancelled
    config_json: str = "{}"
    total_tasks: int = 0
    succeeded: int = 0
    failed: int = 0


class FleetStore:
    """SQLite-backed persistence for fleet orchestration.

    Usage:
        store = FleetStore("fleet.db")
        run_id = store.create_run(config={"max_concurrent": 5})
        store.add_task(run_id, "task-1", "Write tests", depends_on=[])
        store.update_task_status(run_id, "task-1", "running")
        store.record_result(run_id, "task-1", success=True, content="Done", duration_ms=1500)
        store.complete_run(run_id)

        # Resume after crash
        tasks = store.get_incomplete_tasks(run_id)
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS fleet_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                config_json TEXT DEFAULT '{}',
                total_tasks INTEGER DEFAULT 0,
                succeeded INTEGER DEFAULT 0,
                failed INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS fleet_tasks (
                task_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                depends_on_json TEXT DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                duration_ms REAL DEFAULT 0,
                result_content TEXT,
                error_text TEXT,
                PRIMARY KEY (run_id, task_id),
                FOREIGN KEY (run_id) REFERENCES fleet_runs(run_id)
            );
        """)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> FleetStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # --- Run management ---

    def create_run(self, config: dict[str, Any] | None = None) -> str:
        """Create a new fleet run. Returns the run_id."""
        run_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO fleet_runs (run_id, created_at, config_json) VALUES (?, ?, ?)",
            (run_id, now, json.dumps(config or {})),
        )
        self._conn.commit()
        return run_id

    def complete_run(self, run_id: str) -> None:
        """Mark a run as completed, updating success/failure counts."""
        now = datetime.now(timezone.utc).isoformat()
        row = self._conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) as succeeded, "
            "SUM(CASE WHEN status IN ('failed', 'blocked', 'cancelled') THEN 1 ELSE 0 END) as failed "
            "FROM fleet_tasks WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        total = row["total"] or 0
        succeeded = row["succeeded"] or 0
        failed = row["failed"] or 0
        status = "completed" if failed == 0 else "failed"

        self._conn.execute(
            "UPDATE fleet_runs SET completed_at = ?, status = ?, "
            "total_tasks = ?, succeeded = ?, failed = ? WHERE run_id = ?",
            (now, status, total, succeeded, failed, run_id),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> RunRecord | None:
        """Get a run record by ID."""
        row = self._conn.execute("SELECT * FROM fleet_runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        return RunRecord(
            run_id=row["run_id"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            status=row["status"],
            config_json=row["config_json"],
            total_tasks=row["total_tasks"],
            succeeded=row["succeeded"],
            failed=row["failed"],
        )

    def list_runs(self, limit: int = 20) -> list[RunRecord]:
        """List recent fleet runs."""
        rows = self._conn.execute(
            "SELECT * FROM fleet_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            RunRecord(
                run_id=r["run_id"],
                created_at=r["created_at"],
                completed_at=r["completed_at"],
                status=r["status"],
                config_json=r["config_json"],
                total_tasks=r["total_tasks"],
                succeeded=r["succeeded"],
                failed=r["failed"],
            )
            for r in rows
        ]

    # --- Task management ---

    def add_task(
        self,
        run_id: str,
        task_id: str,
        prompt: str,
        depends_on: list[str] | None = None,
    ) -> None:
        """Add a task to a run."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO fleet_tasks (task_id, run_id, prompt, depends_on_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, run_id, prompt, json.dumps(depends_on or []), now),
        )
        self._conn.commit()

    def update_task_status(
        self,
        run_id: str,
        task_id: str,
        status: str,
    ) -> None:
        """Update a task's status."""
        updates: dict[str, str] = {"status": status}
        if status == "running":
            updates["started_at"] = datetime.now(timezone.utc).isoformat()
        elif status in ("done", "failed", "blocked", "cancelled"):
            updates["completed_at"] = datetime.now(timezone.utc).isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [run_id, task_id]
        self._conn.execute(
            f"UPDATE fleet_tasks SET {set_clause} WHERE run_id = ? AND task_id = ?",
            values,
        )
        self._conn.commit()

    def record_result(
        self,
        run_id: str,
        task_id: str,
        *,
        success: bool,
        content: str | None = None,
        error: str | None = None,
        duration_ms: float = 0,
    ) -> None:
        """Record a task result (success or failure)."""
        now = datetime.now(timezone.utc).isoformat()
        status = "done" if success else "failed"
        self._conn.execute(
            "UPDATE fleet_tasks SET status = ?, completed_at = ?, "
            "duration_ms = ?, result_content = ?, error_text = ? "
            "WHERE run_id = ? AND task_id = ?",
            (status, now, duration_ms, content, error, run_id, task_id),
        )
        self._conn.commit()

    def get_task(self, run_id: str, task_id: str) -> TaskRecord | None:
        """Get a single task record."""
        row = self._conn.execute(
            "SELECT * FROM fleet_tasks WHERE run_id = ? AND task_id = ?",
            (run_id, task_id),
        ).fetchone()
        if not row:
            return None
        return self._row_to_task(row)

    def get_tasks(self, run_id: str) -> list[TaskRecord]:
        """Get all tasks for a run."""
        rows = self._conn.execute(
            "SELECT * FROM fleet_tasks WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def get_incomplete_tasks(self, run_id: str) -> list[TaskRecord]:
        """Get tasks that haven't completed successfully — for resume."""
        rows = self._conn.execute(
            "SELECT * FROM fleet_tasks WHERE run_id = ? AND status NOT IN ('done') "
            "ORDER BY created_at",
            (run_id,),
        ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def get_completed_task_ids(self, run_id: str) -> set[str]:
        """Get IDs of tasks that completed successfully."""
        rows = self._conn.execute(
            "SELECT task_id FROM fleet_tasks WHERE run_id = ? AND status = 'done'",
            (run_id,),
        ).fetchall()
        return {r["task_id"] for r in rows}

    def _row_to_task(self, row: sqlite3.Row) -> TaskRecord:
        """Convert a database row to a TaskRecord."""
        return TaskRecord(
            task_id=row["task_id"],
            run_id=row["run_id"],
            prompt=row["prompt"],
            depends_on=json.loads(row["depends_on_json"]),
            status=row["status"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            duration_ms=row["duration_ms"],
            result_content=row["result_content"],
            error_text=row["error_text"],
        )
