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
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from copex.client import Copex, Response
from copex.config import CopexConfig
from copex.models import Model, ReasoningEffort


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug for use as task IDs."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:64] or "task"


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
    ):
        self._config = config or CopexConfig()
        self._fleet_config = fleet_config or FleetConfig()
        self._tasks: list[FleetTask] = []
        self._coordinator = FleetCoordinator(self._config)

    async def __aenter__(self) -> Fleet:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

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

    async def run(
        self,
        on_status: Callable[[str, str], None] | None = None,
    ) -> list[FleetResult]:
        """Execute all tasks and return results."""
        if not self._tasks:
            return []

        # Prepend shared context to each task's prompt
        if self._fleet_config.shared_context:
            ctx = self._fleet_config.shared_context
            for task in self._tasks:
                task.prompt = f"{ctx}\n\n{task.prompt}"

        return await self._coordinator.run(
            self._tasks,
            self._fleet_config,
            on_status=on_status,
        )

    @property
    def tasks(self) -> list[FleetTask]:
        """Return a copy of the current task list."""
        return list(self._tasks)
