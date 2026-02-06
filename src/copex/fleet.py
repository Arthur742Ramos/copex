"""
Fleet - Parallel AI sub-agent orchestration.

Enables dispatching multiple Copex agents that work on separate tasks
concurrently, with optional dependency ordering between tasks.

Usage:
    from copex import Fleet, FleetTask, FleetConfig

    async with Fleet(copex_config) as fleet:
        fleet.add("Write auth tests")
        fleet.add("Refactor DB", depends_on=["write-auth-tests"])
        results = await fleet.run()

    # Streaming progress (new in v1.3.0)
    async with Fleet(copex_config) as fleet:
        fleet.add("Task 1")
        fleet.add("Task 2")
        async for event in fleet.run_streaming():
            print(f"{event.task_id}: {event.event_type}")
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from copex.client import Copex, Response
from copex.config import CopexConfig
from copex.models import Model, ReasoningEffort


class FleetEventType(Enum):
    """Types of fleet events for streaming progress."""

    TASK_QUEUED = "task_queued"
    TASK_WAITING = "task_waiting"  # Waiting for dependencies
    TASK_RUNNING = "task_running"
    TASK_DONE = "task_done"
    TASK_FAILED = "task_failed"
    TASK_BLOCKED = "task_blocked"  # Blocked due to dependency failure
    TASK_CANCELLED = "task_cancelled"
    FLEET_START = "fleet_start"
    FLEET_COMPLETE = "fleet_complete"
    MESSAGE_DELTA = "message_delta"  # Streaming content from task
    MAILBOX_MESSAGE = "mailbox_message"  # Inter-task communication


@dataclass
class FleetEvent:
    """Event emitted during fleet execution for streaming progress.

    Attributes:
        event_type: Type of the event
        task_id: ID of the task (None for fleet-level events)
        timestamp: When the event occurred
        data: Additional event-specific data
        delta: Content delta for MESSAGE_DELTA events
        error: Error message for failure events
    """

    event_type: FleetEventType
    task_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    delta: str = ""
    error: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp == 0:
            self.timestamp = time.time()


class FleetMailbox:
    """Inter-task communication via async message queues.

    Enables tasks to send messages to each other during fleet execution.
    Each task has its own inbox queue.

    Usage:
        mailbox = FleetMailbox()
        mailbox.create_inbox("task-1")
        mailbox.create_inbox("task-2")

        # Task 1 sends to Task 2
        await mailbox.send("task-2", {"result": "some data"}, from_task="task-1")

        # Task 2 receives
        message = await mailbox.receive("task-2", timeout=5.0)
    """

    def __init__(self) -> None:
        self._inboxes: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    def create_inbox(self, task_id: str) -> None:
        """Create an inbox for a task."""
        if task_id not in self._inboxes:
            self._inboxes[task_id] = asyncio.Queue()

    def remove_inbox(self, task_id: str) -> None:
        """Remove a task's inbox."""
        self._inboxes.pop(task_id, None)

    async def send(
        self,
        to_task: str,
        message: dict[str, Any],
        *,
        from_task: str | None = None,
    ) -> bool:
        """Send a message to a task's inbox.

        Args:
            to_task: Target task ID
            message: Message payload
            from_task: Optional sender task ID

        Returns:
            True if message was queued, False if inbox doesn't exist
        """
        if to_task not in self._inboxes:
            return False

        envelope = {
            "from": from_task,
            "to": to_task,
            "payload": message,
            "timestamp": time.time(),
        }
        await self._inboxes[to_task].put(envelope)
        return True

    async def receive(
        self,
        task_id: str,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """Receive a message from a task's inbox.

        Args:
            task_id: Task ID to receive for
            timeout: Optional timeout in seconds

        Returns:
            Message envelope or None on timeout/missing inbox
        """
        if task_id not in self._inboxes:
            return None

        try:
            if timeout is not None:
                return await asyncio.wait_for(
                    self._inboxes[task_id].get(),
                    timeout=timeout,
                )
            return await self._inboxes[task_id].get()
        except asyncio.TimeoutError:
            return None

    def try_receive(self, task_id: str) -> dict[str, Any] | None:
        """Non-blocking receive from inbox.

        Returns:
            Message envelope or None if empty/missing
        """
        if task_id not in self._inboxes:
            return None

        try:
            return self._inboxes[task_id].get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def broadcast(
        self,
        message: dict[str, Any],
        *,
        from_task: str | None = None,
        exclude: list[str] | None = None,
    ) -> int:
        """Broadcast a message to all inboxes.

        Args:
            message: Message payload
            from_task: Optional sender task ID
            exclude: Task IDs to exclude from broadcast

        Returns:
            Number of tasks that received the message
        """
        exclude_set = set(exclude or [])
        count = 0
        for task_id in self._inboxes:
            if task_id not in exclude_set:
                if await self.send(task_id, message, from_task=from_task):
                    count += 1
        return count

    def pending_count(self, task_id: str) -> int:
        """Get count of pending messages for a task."""
        if task_id not in self._inboxes:
            return 0
        return self._inboxes[task_id].qsize()


class FleetContext:
    """Shared mutable state for fleet tasks with result aggregation.

    Provides thread-safe shared state that tasks can read and write to,
    plus result aggregation for collecting task outputs.

    Usage:
        context = FleetContext()
        context.set("shared_config", {"key": "value"})

        # In tasks:
        config = context.get("shared_config")
        context.add_result("task-1", {"status": "success", "data": [...]})

        # After execution:
        all_results = context.get_results()
    """

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        self._state: dict[str, Any] = initial_state.copy() if initial_state else {}
        self._results: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._result_lock = asyncio.Lock()

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from shared state."""
        async with self._lock:
            return self._state.get(key, default)

    def get_sync(self, key: str, default: Any = None) -> Any:
        """Synchronous get (for use in callbacks)."""
        return self._state.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        """Set a value in shared state."""
        async with self._lock:
            self._state[key] = value

    def set_sync(self, key: str, value: Any) -> None:
        """Synchronous set (for use in callbacks)."""
        self._state[key] = value

    async def update(self, updates: dict[str, Any]) -> None:
        """Update multiple values atomically."""
        async with self._lock:
            self._state.update(updates)

    async def delete(self, key: str) -> bool:
        """Delete a key from shared state."""
        async with self._lock:
            if key in self._state:
                del self._state[key]
                return True
            return False

    async def add_result(self, task_id: str, result: Any) -> None:
        """Add a result from a task."""
        async with self._result_lock:
            self._results[task_id] = result

    def add_result_sync(self, task_id: str, result: Any) -> None:
        """Synchronous add_result (for use in callbacks)."""
        self._results[task_id] = result

    async def get_result(self, task_id: str) -> Any | None:
        """Get a specific task's result."""
        async with self._result_lock:
            return self._results.get(task_id)

    def get_result_sync(self, task_id: str) -> Any | None:
        """Synchronous get_result."""
        return self._results.get(task_id)

    async def get_results(self) -> dict[str, Any]:
        """Get all task results."""
        async with self._result_lock:
            return self._results.copy()

    def get_results_sync(self) -> dict[str, Any]:
        """Synchronous get_results."""
        return self._results.copy()

    async def aggregate_results(
        self,
        aggregator: Callable[[dict[str, Any]], Any],
    ) -> Any:
        """Apply an aggregation function to all results.

        Args:
            aggregator: Function that takes results dict and returns aggregated value

        Returns:
            Aggregated result
        """
        async with self._result_lock:
            return aggregator(self._results.copy())

    def clear(self) -> None:
        """Clear all state and results."""
        self._state.clear()
        self._results.clear()

    @property
    def state(self) -> dict[str, Any]:
        """Direct access to state dict (use carefully)."""
        return self._state

    @property
    def results(self) -> dict[str, Any]:
        """Direct access to results dict (use carefully)."""
        return self._results


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug for use as task IDs."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:64] or "task"


def _extract_paths(content: str) -> list[str]:
    """Extract file paths from response content that were likely modified."""
    paths: list[str] = []
    for line in content.splitlines():
        m = re.match(
            r"^\s*(?:Created|Modified|Updated|Wrote|Saved)\s*:?\s+(.+\.\w+)",
            line,
            re.IGNORECASE,
        )
        if m:
            path = m.group(1).strip().strip("`'\"")
            if path and not path.startswith("http"):
                paths.append(path)
    return paths


@dataclass
class FleetTask:
    """A single task to be executed by a fleet agent."""

    id: str
    prompt: str
    depends_on: list[str] = field(default_factory=list)
    model: Model | None = None
    reasoning_effort: ReasoningEffort | None = None


@dataclass
class FleetResult:
    """Result from a fleet agent's execution."""

    task_id: str
    success: bool
    response: Response | None = None
    error: Exception | None = None
    duration_ms: float = 0


@dataclass
class FleetConfig:
    """Configuration for fleet execution."""

    max_concurrent: int = 5
    timeout: float = 600.0
    fail_fast: bool = False
    shared_context: str | None = None
    git_finalize: bool = False
    git_auto_finalize: bool = True
    git_message: str | None = None


class GitFinalizer:
    """Tracks modified paths during a fleet run and commits changes to git.

    Used as a context manager around fleet execution. When finalize() is
    called (or the context manager exits after a successful run), it stages
    all tracked paths, commits with the configured message, and pushes.
    """

    def __init__(
        self,
        *,
        message: str | None = None,
        cwd: str | Path | None = None,
        push: bool = True,
    ) -> None:
        self._message = message or "fleet: apply changes"
        self._cwd = str(cwd) if cwd else None
        self._push = push
        self._modified: set[str] = set()
        self._finalized = False

    def track(self, path: str | Path) -> None:
        """Record a path as modified during the fleet run."""
        self._modified.add(str(path))

    def track_many(self, paths: list[str] | list[Path]) -> None:
        """Record multiple paths as modified."""
        for p in paths:
            self._modified.add(str(p))

    @property
    def modified_paths(self) -> list[str]:
        """Return sorted list of tracked modified paths."""
        return sorted(self._modified)

    @property
    def finalized(self) -> bool:
        """Whether finalize() has been called."""
        return self._finalized

    async def __aenter__(self) -> GitFinalizer:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        if exc_type is None and self._modified and not self._finalized:
            self.finalize()

    def finalize(self) -> None:
        """Stage tracked files, commit, and optionally push.

        Raises:
            subprocess.CalledProcessError: If any git command fails.
            RuntimeError: If no modified paths to commit.
        """
        if self._finalized:
            return
        if not self._modified:
            raise RuntimeError("GitFinalizer: no modified paths to commit")

        run_kwargs: dict = {"check": True, "capture_output": True, "text": True}
        if self._cwd:
            run_kwargs["cwd"] = self._cwd

        subprocess.run(
            ["git", "add", "--", *sorted(self._modified)],
            **run_kwargs,
        )
        subprocess.run(
            ["git", "commit", "-m", self._message],
            **run_kwargs,
        )
        if self._push:
            subprocess.run(["git", "push"], **run_kwargs)

        self._finalized = True


class FleetCoordinator:
    """Dispatches tasks respecting dependency order, collects results."""

    def __init__(self, base_config: CopexConfig):
        self._base_config = base_config

    async def run(
        self,
        tasks: list[FleetTask],
        config: FleetConfig | None = None,
        on_status: Callable[[str, str], None] | None = None,
    ) -> list[FleetResult]:
        """Execute fleet tasks respecting dependencies.

        Uses a task queue pattern: each task is launched immediately and
        awaits its own dependencies, rather than waiting for entire waves
        to complete. This improves throughput when tasks have varying
        durations.

        Args:
            tasks: Tasks to execute.
            config: Fleet configuration.
            on_status: Callback(task_id, status) for progress updates.

        Returns:
            List of FleetResult in the same order as input tasks.
        """
        config = config or FleetConfig()
        self._validate_dag(tasks)

        task_map = {t.id: t for t in tasks}
        results: dict[str, FleetResult] = {}
        semaphore = asyncio.Semaphore(config.max_concurrent)
        cancel_event = asyncio.Event()

        # Each task gets an event that is set when it finishes
        done_events: dict[str, asyncio.Event] = {t.id: asyncio.Event() for t in tasks}
        finished: dict[str, bool] = {}  # task_id -> success

        async def _run_task(task: FleetTask) -> FleetResult:
            if on_status:
                on_status(task.id, "queued")

            # Wait for all dependencies to finish
            for dep in task.depends_on:
                await done_events[dep].wait()
                if not finished.get(dep, False):
                    failed_deps = [
                        d for d in task.depends_on if not finished.get(d, False)
                    ]
                    result = FleetResult(
                        task_id=task.id,
                        success=False,
                        error=RuntimeError(
                            f"Dependency failed for task '{task.id}': "
                            f"upstream {failed_deps!r} did not succeed"
                        ),
                    )
                    results[task.id] = result
                    finished[task.id] = False
                    done_events[task.id].set()
                    if on_status:
                        on_status(task.id, "blocked")
                    return result

            if cancel_event.is_set():
                result = FleetResult(
                    task_id=task.id,
                    success=False,
                    error=asyncio.CancelledError("Fleet cancelled due to fail-fast"),
                )
                results[task.id] = result
                finished[task.id] = False
                done_events[task.id].set()
                return result

            if on_status:
                on_status(task.id, "running")

            start = time.monotonic()
            task_config = self._task_config(task, config)

            try:
                async with semaphore:
                    if cancel_event.is_set():
                        raise asyncio.CancelledError("Fleet cancelled")

                    async with Copex(task_config) as copex:
                        response = await asyncio.wait_for(
                            copex.send(task.prompt),
                            timeout=config.timeout,
                        )

                elapsed = (time.monotonic() - start) * 1000
                result = FleetResult(
                    task_id=task.id,
                    success=True,
                    response=response,
                    duration_ms=elapsed,
                )
            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                result = FleetResult(
                    task_id=task.id,
                    success=False,
                    error=exc,
                    duration_ms=elapsed,
                )
                if config.fail_fast:
                    cancel_event.set()

            results[task.id] = result
            finished[task.id] = result.success
            done_events[task.id].set()
            if result.success:
                if on_status:
                    on_status(task.id, "done")
            else:
                if on_status:
                    on_status(task.id, "failed")
            return result

        # Launch all tasks concurrently; each awaits its own dependencies
        aws = [asyncio.ensure_future(_run_task(task_map[t.id])) for t in tasks]
        await asyncio.gather(*aws, return_exceptions=True)

        return [results[t.id] for t in tasks]

    def _task_config(self, task: FleetTask, fleet_cfg: FleetConfig) -> CopexConfig:
        """Build a per-task config with overrides applied."""
        cfg = CopexConfig(
            model=task.model or self._base_config.model,
            reasoning_effort=task.reasoning_effort or self._base_config.reasoning_effort,
            cli_path=self._base_config.cli_path,
            cli_url=self._base_config.cli_url,
            cwd=self._base_config.cwd,
            timeout=fleet_cfg.timeout,
            auto_continue=self._base_config.auto_continue,
            continue_prompt=self._base_config.continue_prompt,
            skills=self._base_config.skills,
            skill_directories=self._base_config.skill_directories,
            available_tools=self._base_config.available_tools,
            excluded_tools=self._base_config.excluded_tools,
            mcp_servers=self._base_config.mcp_servers,
            mcp_config_file=self._base_config.mcp_config_file,
            retry=self._base_config.retry,
        )
        return cfg

    @staticmethod
    def _validate_dag(tasks: list[FleetTask]) -> None:
        """Validate that tasks form a valid DAG (no cycles, deps exist)."""
        task_ids = {t.id for t in tasks}

        # Check for missing dependencies
        missing_deps: list[tuple[str, str]] = []
        for task in tasks:
            for dep in task.depends_on:
                if dep not in task_ids:
                    missing_deps.append((task.id, dep))

        if missing_deps:
            details = "\n".join(
                f"  - Task '{tid}' depends on unknown task '{dep}'"
                for tid, dep in missing_deps
            )
            raise ValueError(
                f"Missing dependencies detected:\n{details}\n"
                f"Available task IDs: {sorted(task_ids)}"
            )

        # Topological sort to detect cycles (Kahn's algorithm)
        in_degree: dict[str, int] = {t.id: 0 for t in tasks}
        adjacency: dict[str, list[str]] = {t.id: [] for t in tasks}
        for task in tasks:
            for dep in task.depends_on:
                adjacency[dep].append(task.id)
                in_degree[task.id] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(tasks):
            # Find tasks involved in cycles
            cycle_tasks = [tid for tid, deg in in_degree.items() if deg > 0]
            cycle_details = []
            for tid in cycle_tasks:
                task = next(t for t in tasks if t.id == tid)
                cycle_details.append(f"  - '{tid}' depends on {task.depends_on}")
            raise ValueError(
                f"Cycle detected in fleet task dependencies. "
                f"Tasks involved in cycle:\n" + "\n".join(cycle_details)
            )


class Fleet:
    """High-level convenience wrapper for fleet execution.

    Usage:
        async with Fleet(config) as fleet:
            fleet.add("Write auth tests")
            fleet.add("Refactor DB", depends_on=["write-auth-tests"])
            results = await fleet.run()
    """

    def __init__(
        self,
        config: CopexConfig | None = None,
        fleet_config: FleetConfig | None = None,
        db_path: str | Path | None = None,
    ):
        self._config = config or CopexConfig()
        self._fleet_config = fleet_config or FleetConfig()
        self._tasks: list[FleetTask] = []
        self._coordinator = FleetCoordinator(self._config)
        self._store: FleetStore | None = None
        self._run_id: str | None = None
        if db_path is not None:
            from copex.fleet_store import FleetStore
            self._store = FleetStore(db_path)
        self._git_finalizer: GitFinalizer | None = None
        if self._fleet_config.git_finalize:
            self._git_finalizer = GitFinalizer(
                message=self._fleet_config.git_message,
                cwd=self._config.cwd,
            )

    async def __aenter__(self) -> Fleet:
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._store is not None:
            self._store.close()
        if self._git_finalizer is not None and not self._git_finalizer.finalized:
            try:
                self._git_finalizer.finalize()
            except (subprocess.CalledProcessError, RuntimeError):
                pass  # Best-effort on context exit

    def add(
        self,
        prompt: str,
        *,
        task_id: str | None = None,
        depends_on: list[str] | None = None,
        model: Model | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> str:
        """Add a task to the fleet.

        Returns:
            The task ID (auto-generated from prompt if not provided).
        """
        tid = task_id or _slugify(prompt)

        # Ensure unique ID
        existing = {t.id for t in self._tasks}
        if tid in existing:
            suffix = 2
            while f"{tid}-{suffix}" in existing:
                suffix += 1
            tid = f"{tid}-{suffix}"

        task = FleetTask(
            id=tid,
            prompt=prompt,
            depends_on=depends_on or [],
            model=model,
            reasoning_effort=reasoning_effort,
        )
        self._tasks.append(task)
        return tid

    @classmethod
    async def resume(
        cls,
        db_path: str | Path,
        run_id: str,
        config: CopexConfig | None = None,
        fleet_config: FleetConfig | None = None,
        on_status: Callable[[str, str], None] | None = None,
    ) -> list[FleetResult]:
        """Resume an interrupted fleet run from the database.

        Loads task state from a previous run, skips already-completed tasks,
        and re-runs only pending/failed/blocked tasks.

        Args:
            db_path: Path to the SQLite database.
            run_id: ID of the run to resume.
            config: Copex configuration (uses defaults if not provided).
            fleet_config: Fleet configuration (loaded from DB if not provided).

        Returns:
            List of FleetResult for re-run tasks only.

        Example:
            results = await Fleet.resume("fleet.db", "abc123def456")
        """
        from copex.fleet_store import FleetStore

        store = FleetStore(db_path)
        fleet = None
        try:
            run = store.get_run(run_id)
            if run is None:
                raise ValueError(f"Run not found: {run_id}")

            all_tasks = store.get_tasks(run_id)
            if not all_tasks:
                return []

            completed_ids = store.get_completed_task_ids(run_id)
            incomplete = [t for t in all_tasks if t.task_id not in completed_ids]

            if not incomplete:
                return []

            # Reconstruct FleetConfig from stored config if not provided
            if fleet_config is None:
                import json
                stored_config = json.loads(run.config_json) if run.config_json else {}
                fleet_config = FleetConfig(
                    max_concurrent=stored_config.get("max_concurrent", 5),
                    timeout=stored_config.get("timeout", 600.0),
                    fail_fast=stored_config.get("fail_fast", False),
                )

            # Build Fleet with the remaining tasks
            fleet = cls(config=config, fleet_config=fleet_config, db_path=db_path)
            fleet._run_id = run_id  # Reuse existing run_id

            for task_rec in incomplete:
                # Filter depends_on to only include deps that aren't already done
                remaining_deps = [d for d in task_rec.depends_on if d not in completed_ids]
                fleet._tasks.append(FleetTask(
                    id=task_rec.task_id,
                    prompt=task_rec.prompt,
                    depends_on=remaining_deps,
                ))

            # Reset task statuses in the store for re-run
            for task_rec in incomplete:
                store.update_task_status(run_id, task_rec.task_id, "pending")

            results = await fleet.run(on_status=on_status)

            # run() already calls complete_run via fleet._store,
            # so no need to call it again here.

            return results
        finally:
            store.close()
            # Also close the fleet's own store (opened by __init__)
            if fleet is not None and fleet._store is not None:
                fleet._store.close()
                fleet._store = None

    async def run(
        self,
        on_status: Callable[[str, str], None] | None = None,
    ) -> list[FleetResult]:
        """Execute all tasks and return results."""
        if not self._tasks:
            return []

        # Prepend shared context without mutating original tasks
        tasks = self._tasks
        if self._fleet_config.shared_context:
            ctx = self._fleet_config.shared_context
            tasks = [
                FleetTask(
                    id=t.id,
                    prompt=f"{ctx}\n\n{t.prompt}",
                    depends_on=t.depends_on,
                    model=t.model,
                    reasoning_effort=t.reasoning_effort,
                )
                for t in self._tasks
            ]

        # Persist to store if configured (skip if resuming an existing run)
        if self._store is not None and self._run_id is None:
            self._run_id = self._store.create_run(
                config={
                    "max_concurrent": self._fleet_config.max_concurrent,
                    "timeout": self._fleet_config.timeout,
                    "fail_fast": self._fleet_config.fail_fast,
                }
            )
            for task in tasks:
                self._store.add_task(
                    self._run_id, task.id, task.prompt, depends_on=task.depends_on
                )

        # Wrap on_status to also update store
        original_on_status = on_status
        store = self._store
        run_id = self._run_id

        def _tracking_on_status(task_id: str, status: str) -> None:
            if store is not None and run_id is not None:
                store.update_task_status(run_id, task_id, status)
            if original_on_status:
                original_on_status(task_id, status)

        results = await self._coordinator.run(
            tasks,
            self._fleet_config,
            on_status=_tracking_on_status,
        )

        # Record results in store
        if store is not None and run_id is not None:
            for result in results:
                store.record_result(
                    run_id,
                    result.task_id,
                    success=result.success,
                    content=result.response.content if result.response else None,
                    error=str(result.error) if result.error else None,
                    duration_ms=result.duration_ms,
                )
            store.complete_run(run_id)

        # Git finalize: track modified files from successful responses
        if self._git_finalizer is not None:
            for result in results:
                if result.success and result.response:
                    for path in _extract_paths(result.response.content):
                        self._git_finalizer.track(path)

        return results

    async def run_streaming(
        self,
        *,
        include_deltas: bool = False,
        context: FleetContext | None = None,
        mailbox: FleetMailbox | None = None,
    ) -> AsyncIterator[FleetEvent]:
        """Execute all tasks with streaming progress events.

        This is an async generator that yields FleetEvent objects as
        tasks progress through their lifecycle.

        Args:
            include_deltas: If True, yield MESSAGE_DELTA events for streaming content
            context: Optional FleetContext for shared state
            mailbox: Optional FleetMailbox for inter-task communication

        Yields:
            FleetEvent objects for each state change

        Example:
            async with Fleet(config) as fleet:
                fleet.add("Task 1")
                fleet.add("Task 2")
                async for event in fleet.run_streaming():
                    if event.event_type == FleetEventType.TASK_DONE:
                        print(f"{event.task_id} completed!")
        """
        if not self._tasks:
            yield FleetEvent(event_type=FleetEventType.FLEET_COMPLETE, data={"total_tasks": 0})
            return

        # Initialize context and mailbox if provided
        ctx = context or FleetContext()
        mbox = mailbox or FleetMailbox()

        # Create inboxes for all tasks
        for task in self._tasks:
            mbox.create_inbox(task.id)

        # Prepend shared context without mutating original tasks
        tasks = self._tasks
        if self._fleet_config.shared_context:
            shared_ctx = self._fleet_config.shared_context
            tasks = [
                FleetTask(
                    id=t.id,
                    prompt=f"{shared_ctx}\n\n{t.prompt}",
                    depends_on=t.depends_on,
                    model=t.model,
                    reasoning_effort=t.reasoning_effort,
                )
                for t in self._tasks
            ]

        # Persist to store if configured (skip if resuming an existing run)
        if self._store is not None and self._run_id is None:
            self._run_id = self._store.create_run(
                config={
                    "max_concurrent": self._fleet_config.max_concurrent,
                    "timeout": self._fleet_config.timeout,
                    "fail_fast": self._fleet_config.fail_fast,
                }
            )
            for task in tasks:
                self._store.add_task(
                    self._run_id, task.id, task.prompt, depends_on=task.depends_on
                )
        store = self._store
        run_id = self._run_id

        # Emit fleet start event
        yield FleetEvent(
            event_type=FleetEventType.FLEET_START,
            data={
                "total_tasks": len(tasks),
                "max_concurrent": self._fleet_config.max_concurrent,
                "task_ids": [t.id for t in tasks],
            },
        )

        # Bounded event queue for streaming – backpressure under load
        event_queue: asyncio.Queue[FleetEvent | None] = asyncio.Queue(maxsize=10000)
        results: dict[str, FleetResult] = {}
        semaphore = asyncio.Semaphore(self._fleet_config.max_concurrent)
        cancel_event = asyncio.Event()
        done_events: dict[str, asyncio.Event] = {t.id: asyncio.Event() for t in tasks}
        finished: dict[str, bool] = {}

        async def _run_task(task: FleetTask) -> FleetResult:
            # Emit queued event
            await event_queue.put(FleetEvent(
                event_type=FleetEventType.TASK_QUEUED,
                task_id=task.id,
                data={"prompt_preview": task.prompt[:100]},
            ))
            if store is not None and run_id is not None:
                store.update_task_status(run_id, task.id, "pending")

            # Wait for dependencies
            if task.depends_on:
                await event_queue.put(FleetEvent(
                    event_type=FleetEventType.TASK_WAITING,
                    task_id=task.id,
                    data={"waiting_for": task.depends_on},
                ))

            for dep in task.depends_on:
                await done_events[dep].wait()
                if not finished.get(dep, False):
                    failed_deps = [d for d in task.depends_on if not finished.get(d, False)]
                    result = FleetResult(
                        task_id=task.id,
                        success=False,
                        error=RuntimeError(
                            f"Dependency failed for task '{task.id}': "
                            f"upstream {failed_deps!r} did not succeed"
                        ),
                    )
                    results[task.id] = result
                    finished[task.id] = False
                    done_events[task.id].set()
                    await event_queue.put(FleetEvent(
                        event_type=FleetEventType.TASK_BLOCKED,
                        task_id=task.id,
                        error=f"Blocked by failed dependencies: {failed_deps}",
                    ))
                    if store is not None and run_id is not None:
                        store.update_task_status(run_id, task.id, "blocked")
                    return result

            if cancel_event.is_set():
                result = FleetResult(
                    task_id=task.id,
                    success=False,
                    error=asyncio.CancelledError("Fleet cancelled"),
                )
                results[task.id] = result
                finished[task.id] = False
                done_events[task.id].set()
                await event_queue.put(FleetEvent(
                    event_type=FleetEventType.TASK_CANCELLED,
                    task_id=task.id,
                ))
                return result

            # Emit running event
            await event_queue.put(FleetEvent(
                event_type=FleetEventType.TASK_RUNNING,
                task_id=task.id,
            ))
            if store is not None and run_id is not None:
                store.update_task_status(run_id, task.id, "running")

            start = time.monotonic()
            task_config = self._coordinator._task_config(task, self._fleet_config)

            try:
                async with semaphore:
                    if cancel_event.is_set():
                        raise asyncio.CancelledError("Fleet cancelled")

                    async with Copex(task_config) as copex:
                        # Build streaming callback if needed
                        on_chunk = None
                        if include_deltas:
                            async def _emit_delta(delta: str) -> None:
                                await event_queue.put(FleetEvent(
                                    event_type=FleetEventType.MESSAGE_DELTA,
                                    task_id=task.id,
                                    delta=delta,
                                ))

                            # Sync wrapper for the async callback
                            def on_chunk(chunk: Any) -> None:
                                if hasattr(chunk, "delta") and chunk.delta:
                                    evt = FleetEvent(
                                        event_type=FleetEventType.MESSAGE_DELTA,
                                        task_id=task.id,
                                        delta=chunk.delta,
                                    )
                                    try:
                                        event_queue.put_nowait(evt)
                                    except asyncio.QueueFull:
                                        pass  # Drop delta events under backpressure

                        response = await asyncio.wait_for(
                            copex.send(task.prompt, on_chunk=on_chunk if include_deltas else None),
                            timeout=self._fleet_config.timeout,
                        )

                elapsed = (time.monotonic() - start) * 1000
                result = FleetResult(
                    task_id=task.id,
                    success=True,
                    response=response,
                    duration_ms=elapsed,
                )

                # Store in context if provided
                if context:
                    ctx.add_result_sync(task.id, {
                        "success": True,
                        "content": response.content,
                        "duration_ms": elapsed,
                    })

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                result = FleetResult(
                    task_id=task.id,
                    success=False,
                    error=exc,
                    duration_ms=elapsed,
                )
                if self._fleet_config.fail_fast:
                    cancel_event.set()

                if context:
                    ctx.add_result_sync(task.id, {
                        "success": False,
                        "error": str(exc),
                        "duration_ms": elapsed,
                    })

            results[task.id] = result
            finished[task.id] = result.success
            done_events[task.id].set()

            # Emit completion event
            if result.success:
                await event_queue.put(FleetEvent(
                    event_type=FleetEventType.TASK_DONE,
                    task_id=task.id,
                    data={
                        "duration_ms": result.duration_ms,
                        "content_preview": (result.response.content[:200] if result.response else "")[:200],
                    },
                ))
                if store is not None and run_id is not None:
                    store.record_result(
                        run_id,
                        task.id,
                        success=True,
                        content=result.response.content if result.response else None,
                        duration_ms=result.duration_ms,
                    )
            else:
                await event_queue.put(FleetEvent(
                    event_type=FleetEventType.TASK_FAILED,
                    task_id=task.id,
                    error=str(result.error) if result.error else "Unknown error",
                    data={"duration_ms": result.duration_ms},
                ))
                if store is not None and run_id is not None:
                    store.record_result(
                        run_id,
                        task.id,
                        success=False,
                        error=str(result.error) if result.error else None,
                        duration_ms=result.duration_ms,
                    )

            return result

        # Start all tasks
        task_futures = [asyncio.ensure_future(_run_task(task)) for task in tasks]

        # Collector task that signals completion
        async def _wait_all() -> None:
            await asyncio.gather(*task_futures, return_exceptions=True)
            await event_queue.put(None)  # Signal end

        collector = asyncio.ensure_future(_wait_all())

        # Yield events as they arrive
        try:
            while True:
                event = await event_queue.get()
                if event is None:
                    break
                yield event
        finally:
            collector.cancel()
            try:
                await collector
            except asyncio.CancelledError:
                pass

        # Emit fleet complete event
        success_count = sum(1 for r in results.values() if r.success)
        total_duration = sum(r.duration_ms for r in results.values())
        if store is not None and run_id is not None:
            store.complete_run(run_id)
        yield FleetEvent(
            event_type=FleetEventType.FLEET_COMPLETE,
            data={
                "total_tasks": len(tasks),
                "succeeded": success_count,
                "failed": len(tasks) - success_count,
                "total_duration_ms": total_duration,
            },
        )

    @property
    def tasks(self) -> list[FleetTask]:
        """Return a copy of the current task list."""
        return list(self._tasks)
