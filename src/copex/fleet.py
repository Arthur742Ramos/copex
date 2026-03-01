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
import logging
import re
import subprocess
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from copex.client import Copex, Response, SessionPool
from copex.config import CopexConfig, make_client
from copex.models import Model, ReasoningEffort

logger = logging.getLogger(__name__)

try:
    from copilot import CopilotClient
except ImportError:
    CopilotClient = None  # type: ignore[misc,assignment]


# Rate limit detection regex (matches 429, "rate limit", "too many requests")
_RATE_LIMIT_RE = re.compile(
    r"\brate\s*limit|(?:^|http\s*|status\s*|code\s*|error\s*)429\b|too many requests",
    re.IGNORECASE,
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a rate limit error."""
    # Check exception type name
    exc_type = type(exc).__name__.lower()
    if "ratelimit" in exc_type or "429" in exc_type:
        return True
    # Check exception message
    msg = str(exc)
    if _RATE_LIMIT_RE.search(msg):
        return True
    # Check for status_code attribute
    if hasattr(exc, "status_code") and exc.status_code == 429:
        return True
    return False


class DynamicSemaphore:
    """A semaphore that supports dynamic resizing of the concurrency limit.

    Unlike ``asyncio.Semaphore``, the maximum number of concurrent holders
    can be changed at runtime via :meth:`resize`.
    """

    def __init__(self, value: int) -> None:
        self._limit = max(1, value)
        self._active = 0
        self._condition = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    async def resize(self, new_limit: int) -> None:
        """Change the concurrency limit and wake waiting tasks if expanded."""
        async with self._condition:
            self._limit = max(1, new_limit)
            self._condition.notify_all()

    async def acquire(self) -> None:
        async with self._condition:
            while self._active >= self._limit:
                await self._condition.wait()
            self._active += 1

    async def release(self) -> None:
        async with self._condition:
            self._active -= 1
            self._condition.notify_all()

    async def __aenter__(self) -> DynamicSemaphore:
        await self.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.release()


class AdaptiveConcurrency:
    """Manages adaptive concurrency based on rate limit errors.

    Reduces concurrency when rate limits are hit, gradually restores
    after a streak of successes.
    """

    def __init__(
        self,
        initial: int,
        minimum: int = 1,
        restore_after: int = 3,
    ) -> None:
        self._initial = initial
        self._current = initial
        self._minimum = max(1, minimum)
        self._restore_after = restore_after
        self._success_streak = 0
        self._lock = asyncio.Lock()
        self._semaphore = DynamicSemaphore(initial)

    @property
    def current(self) -> int:
        """Current concurrency level."""
        return self._current

    @property
    def semaphore(self) -> DynamicSemaphore:
        """The dynamic semaphore gating task execution."""
        return self._semaphore

    async def on_rate_limit(self) -> int:
        """Called when a rate limit error occurs. Returns new concurrency level."""
        async with self._lock:
            old = self._current
            # Halve the concurrency, respecting minimum
            self._current = max(self._minimum, self._current // 2)
            self._success_streak = 0
            if self._current < old:
                logger.warning(
                    "Rate limit hit: reducing fleet concurrency %d -> %d",
                    old,
                    self._current,
                )
        await self._semaphore.resize(self._current)
        return self._current

    async def on_success(self) -> int:
        """Called on successful task completion. Returns new concurrency level."""
        async with self._lock:
            self._success_streak += 1
            if self._current < self._initial and self._success_streak >= self._restore_after:
                old = self._current
                # Increase by 1, not exceeding initial
                self._current = min(self._initial, self._current + 1)
                self._success_streak = 0
                if self._current > old:
                    logger.info(
                        "Restoring fleet concurrency %d -> %d after %d successes",
                        old,
                        self._current,
                        self._restore_after,
                    )
        await self._semaphore.resize(self._current)
        return self._current

    async def on_failure(self) -> None:
        """Called on non-rate-limit failure. Resets success streak."""
        async with self._lock:
            self._success_streak = 0


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


class DependencyFailurePolicy(str, Enum):
    """How a task reacts when one of its dependencies fails."""

    BLOCK = "block"
    CONTINUE = "continue"


_TASK_OUTPUT_REF_RE = re.compile(
    r"\{\{\s*task:(?P<task_id>[a-zA-Z0-9_.-]+)\.(?P<field>"
    r"content|reasoning|success|error|duration_ms|prompt_tokens|completion_tokens"
    r")\s*\}\}"
)


def _normalize_dep_failure_policy(
    value: DependencyFailurePolicy | str | None,
) -> DependencyFailurePolicy:
    if value is None:
        return DependencyFailurePolicy.BLOCK
    if isinstance(value, DependencyFailurePolicy):
        return value
    try:
        return DependencyFailurePolicy(str(value).strip().lower())
    except ValueError:
        return DependencyFailurePolicy.BLOCK


def _render_prompt_with_task_outputs(
    prompt: str,
    *,
    current_task_id: str,
    results: dict[str, FleetResult],
) -> str:
    """Expand task output placeholders in a prompt.

    Supported placeholders:
      - {{task:<task_id>.content}}
      - {{task:<task_id>.reasoning}}
      - {{task:<task_id>.success}}
      - {{task:<task_id>.error}}
      - {{task:<task_id>.duration_ms}}
      - {{task:<task_id>.prompt_tokens}}
      - {{task:<task_id>.completion_tokens}}
    """

    if "{{task:" not in prompt:
        return prompt

    def _replace(match: re.Match[str]) -> str:
        ref_task = match.group("task_id")
        field = match.group("field")

        if ref_task == current_task_id:
            raise ValueError("cannot reference current task output")

        ref_result = results.get(ref_task)
        if ref_result is None:
            raise ValueError(
                f"task '{ref_task}' has no available result; add depends_on for deterministic ordering"
            )

        if field == "success":
            return "true" if ref_result.success else "false"
        if field == "duration_ms":
            return str(round(ref_result.duration_ms, 3))
        if field == "error":
            return str(ref_result.error) if ref_result.error else ""

        response = ref_result.response
        if response is None:
            return ""
        if field == "content":
            return response.content
        if field == "reasoning":
            return response.reasoning or ""
        if field == "prompt_tokens":
            return str(int(response.prompt_tokens or 0))
        if field == "completion_tokens":
            return str(int(response.completion_tokens or 0))

        return ""

    try:
        return _TASK_OUTPUT_REF_RE.sub(_replace, prompt)
    except ValueError as exc:
        raise RuntimeError(f"Task '{current_task_id}' prompt template error: {exc}") from exc


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
        if self.timestamp <= 0:
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
        """Clear all state and results.

        Note: This is a synchronous method. If called concurrently with
        async get/set/add_result, prefer ``aclear()`` instead.
        """
        self._state.clear()
        self._results.clear()

    async def aclear(self) -> None:
        """Clear all state and results (async, lock-safe)."""
        async with self._lock:
            self._state.clear()
        async with self._result_lock:
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


def _load_skills_content(skills_dirs: list[str]) -> str:
    """Load markdown content from a list of skills directories."""
    if not skills_dirs:
        return ""

    contents: list[str] = []
    for dir_path in skills_dirs:
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Skills directory not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Skills path is not a directory: {path}")

        md_files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix == ".md")
        for md_file in md_files:
            text = md_file.read_text(encoding="utf-8").strip()
            if text:
                contents.append(text)

    return "\n\n".join(contents)


def _normalize_mcp_servers(
    mcp_servers: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Normalize MCP server configs to a list of server dicts."""
    if mcp_servers is None:
        return None
    if isinstance(mcp_servers, list):
        return mcp_servers
    if not isinstance(mcp_servers, dict):
        raise ValueError("MCP server config must be a list or mapping")
    if "servers" in mcp_servers and isinstance(mcp_servers["servers"], dict):
        mcp_servers = mcp_servers["servers"]
    if not mcp_servers:
        return []
    if any(
        key in mcp_servers for key in ("command", "url", "transport", "args", "env", "cwd", "name")
    ):
        return [mcp_servers]
    normalized: list[dict[str, Any]] = []
    for name, config in mcp_servers.items():
        if not isinstance(config, dict):
            raise ValueError("MCP server entries must be mappings")
        if "name" in config:
            normalized.append(config)
        else:
            normalized.append({"name": name, **config})
    return normalized


# ---------------------------------------------------------------------------
# Shared helpers extracted from Fleet.run() / Fleet.run_streaming()
# ---------------------------------------------------------------------------


def _prepend_shared_context(
    tasks: list[FleetTask], shared_context: str | None
) -> list[FleetTask]:
    """Return tasks with *shared_context* prepended to each prompt.

    If *shared_context* is falsy the original list is returned unchanged.
    Original tasks are never mutated.
    """
    if not shared_context:
        return tasks
    return [
        FleetTask(
            id=t.id,
            prompt=f"{shared_context}\n\n{t.prompt}",
            depends_on=t.depends_on,
            model=t.model,
            reasoning_effort=t.reasoning_effort,
            cwd=t.cwd,
            skills=t.skills,
            exclude_tools=t.exclude_tools,
            mcp_servers=t.mcp_servers,
            timeout_sec=t.timeout_sec,
            skills_dirs=t.skills_dirs,
            on_dependency_failure=t.on_dependency_failure,
        )
        for t in tasks
    ]


async def _run_task_with_retry(
    task: FleetTask,
    config: FleetConfig,
    run_once: Callable[[FleetTask, int], Awaitable[FleetResult]],
    on_status: Callable[[str, str], None] | None = None,
    adaptive: AdaptiveConcurrency | None = None,
) -> FleetResult:
    """Execute *task* with per-task retry logic and adaptive concurrency."""
    max_retries = task.retries if task.retries is not None else config.default_retries
    retry_delay = (
        task.retry_delay if task.retry_delay is not None else config.default_retry_delay
    )

    result: FleetResult | None = None

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        if attempt > 0:
            # Calculate exponential backoff with jitter
            delay = retry_delay * (2 ** (attempt - 1))
            jitter = delay * 0.2 * (asyncio.get_event_loop().time() % 1)
            await asyncio.sleep(delay + jitter)
            if on_status:
                on_status(task.id, f"retry-{attempt}")

        result = await run_once(task, attempt)

        if result.success:
            if adaptive:
                await adaptive.on_success()
            return result

        last_error = result.error

        # Check if it's a rate limit error
        if last_error and _is_rate_limit_error(last_error):
            if adaptive:
                await adaptive.on_rate_limit()
            # Always retry rate limits if we have retries left
            continue
        elif adaptive:
            await adaptive.on_failure()

        # For non-rate-limit errors, check if we should retry
        if attempt >= max_retries:
            break

    # All retries exhausted – result is guaranteed non-None because the loop
    # runs at least once (max_retries >= 0).
    assert result is not None  # noqa: S101
    return result


def _make_tracking_on_status(
    on_status: Callable[[str, str], None] | None,
    store: Any,
    run_id: str | None,
) -> Callable[[str, str], None]:
    """Wrap *on_status* to also update the fleet store."""

    def _tracking(task_id: str, status: str) -> None:
        if store is not None and run_id is not None:
            store.update_task_status(run_id, task_id, status)
        if on_status:
            on_status(task_id, status)

    return _tracking


def _track_git_files(
    git_finalizer: GitFinalizer | None, results: Iterable[FleetResult]
) -> None:
    """Track modified files from successful responses for git finalization."""
    if git_finalizer is not None:
        for result in results:
            if result.success and result.response:
                for path in _extract_paths(result.response.content):
                    git_finalizer.track(path)


@dataclass
class FleetTask:
    """A single task to be executed by a fleet agent."""

    id: str
    prompt: str
    depends_on: list[str] = field(default_factory=list)
    model: Model | None = None
    reasoning_effort: ReasoningEffort | None = None
    cwd: str | None = None
    skills: list[str] | None = None
    exclude_tools: list[str] | None = None
    mcp_servers: dict[str, Any] | list[dict[str, Any]] | None = None
    timeout_sec: float | None = None
    skills_dirs: list[str] = field(default_factory=list)
    on_dependency_failure: DependencyFailurePolicy = DependencyFailurePolicy.BLOCK
    priority: int = 0  # Lower = higher priority; set by _prioritize_tasks
    # Per-task retry configuration (falls back to global config if None)
    retries: int | None = None
    retry_delay: float | None = None  # Base delay in seconds

    @property
    def excluded_tools(self) -> list[str] | None:
        return self.exclude_tools


@dataclass
class FleetResult:
    """Result from a fleet agent's execution."""

    task_id: str
    success: bool
    response: Response | None = None
    error: Exception | None = None
    duration_ms: float = 0
    # Cost tracking fields (aggregated from response)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    retries_used: int = 0  # Number of retries needed for this task


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
    # Global retry configuration (can be overridden per-task)
    default_retries: int = 3
    default_retry_delay: float = 1.0  # Base delay in seconds
    # Dependency wait timeout (0 = use task timeout / fleet timeout)
    dep_timeout: float = 0.0
    # Adaptive concurrency settings
    adaptive_concurrency: bool = True  # Auto-reduce on rate limits
    min_concurrent: int = 1  # Minimum concurrency when adapting
    concurrency_restore_after: int = 3  # Successes needed to restore


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
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.finalize)

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

    @staticmethod
    def _compute_depth(tasks: list[FleetTask]) -> dict[str, int]:
        """Compute dependency depth for each task (0 = no deps)."""
        depth: dict[str, int] = {}
        task_map = {t.id: t for t in tasks}

        def _get_depth(tid: str) -> int:
            if tid in depth:
                return depth[tid]
            task = task_map.get(tid)
            if not task or not task.depends_on:
                depth[tid] = 0
                return 0
            d = 1 + max(_get_depth(dep) for dep in task.depends_on)
            depth[tid] = d
            return d

        for t in tasks:
            _get_depth(t.id)
        return depth

    @staticmethod
    def _prioritize_tasks(tasks: list[FleetTask]) -> list[FleetTask]:
        """Sort tasks for optimal execution: shallowest deps first, then shortest prompt."""
        depth = FleetCoordinator._compute_depth(tasks)
        return sorted(tasks, key=lambda t: (depth.get(t.id, 0), len(t.prompt)))

    async def run(
        self,
        tasks: list[FleetTask],
        config: FleetConfig | None = None,
        on_status: Callable[[str, str], None] | None = None,
    ) -> list[FleetResult]:
        """Execute fleet tasks respecting dependencies.

        Uses a worker-pool pattern with a shared work-stealing deque.
        Tasks are prioritized by dependency depth (shallowest first)
        and prompt length (shortest first) to maximize throughput.

        Sessions are reused via SessionPool for efficiency (new in v1.9.0).

        New in v2.0.0:
        - Per-task retry policies with configurable retries and delay
        - Adaptive concurrency: auto-reduces on rate limit errors
        - Cost tracking: aggregates prompt_tokens, completion_tokens, total_cost

        Args:
            tasks: Tasks to execute.
            config: Fleet configuration.
            on_status: Callback(task_id, status) for progress updates.

        Returns:
            List of FleetResult in the same order as input tasks.
        """
        config = config or FleetConfig()
        self._validate_dag(tasks)

        results: dict[str, FleetResult] = {}
        cancel_event = asyncio.Event()

        # Adaptive concurrency management
        adaptive = (
            AdaptiveConcurrency(
                initial=config.max_concurrent,
                minimum=config.min_concurrent,
                restore_after=config.concurrency_restore_after,
            )
            if config.adaptive_concurrency
            else None
        )

        # Use adaptive semaphore when available, otherwise a fixed one
        semaphore: DynamicSemaphore | asyncio.Semaphore = (
            adaptive.semaphore if adaptive else asyncio.Semaphore(config.max_concurrent)
        )

        # Each task gets an event that is set when it finishes
        done_events: dict[str, asyncio.Event] = {t.id: asyncio.Event() for t in tasks}
        finished: dict[str, bool] = {}  # task_id -> success

        # Work-stealing deque: workers pop from left, steal from right
        prioritized = self._prioritize_tasks(tasks)
        work_deque: deque[FleetTask] = deque(prioritized)
        work_available = asyncio.Event()
        work_available.set()  # Initial work available

        # Create shared client and session pool for efficiency (v1.9.0)
        # Skip SDK client when use_cli is set — tasks use CopilotCLI instead
        client = None
        pool = None
        if CopilotClient is not None and not self._base_config.use_cli:
            client = CopilotClient(self._base_config.to_client_options())
            await client.start()
            pool = SessionPool(
                max_sessions=config.max_concurrent,
                max_idle_time=300.0,
            )
            await pool.start()

        try:

            async def _run_task_once(task: FleetTask, retries_used: int = 0) -> FleetResult:
                if on_status and retries_used == 0:
                    on_status(task.id, "queued")

                # Wait for all dependencies to finish
                dep_policy = _normalize_dep_failure_policy(task.on_dependency_failure)
                dep_wait_timeout = (
                    config.dep_timeout
                    or (task.timeout_sec if task.timeout_sec is not None else config.timeout)
                )
                for dep in task.depends_on:
                    try:
                        await asyncio.wait_for(
                            done_events[dep].wait(), timeout=dep_wait_timeout
                        )
                    except asyncio.TimeoutError:
                        error = RuntimeError(
                            f"Task '{task.id}' timed out waiting for dependency '{dep}' "
                            f"after {dep_wait_timeout}s"
                        )
                        result = FleetResult(
                            task_id=task.id,
                            success=False,
                            error=error,
                        )
                        results[task.id] = result
                        finished[task.id] = False
                        done_events[task.id].set()
                        if on_status:
                            on_status(task.id, "blocked")
                        return result
                failed_deps = [d for d in task.depends_on if not finished.get(d, False)]
                if failed_deps and dep_policy == DependencyFailurePolicy.BLOCK:
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

                if on_status and retries_used == 0:
                    on_status(task.id, "running")

                start = time.monotonic()

                try:
                    rendered_prompt = _render_prompt_with_task_outputs(
                        task.prompt,
                        current_task_id=task.id,
                        results=results,
                    )
                    task_config = self._task_config(task, config)
                    task_timeout = (
                        task.timeout_sec if task.timeout_sec is not None else config.timeout
                    )
                    async with semaphore:
                        if cancel_event.is_set():
                            raise asyncio.CancelledError("Fleet cancelled")

                        # Use CLI client when use_cli is set
                        if task_config.use_cli:
                            async with make_client(task_config) as cli:
                                response = await asyncio.wait_for(
                                    cli.send(rendered_prompt),
                                    timeout=task_timeout,
                                )
                        # Use session pool if available (v1.9.0)
                        elif pool is not None and client is not None:
                            async with pool.acquire(client, task_config) as session:
                                copex = Copex(task_config)
                                copex._started = True
                                copex._client = client
                                copex._session = session
                                try:
                                    response = await asyncio.wait_for(
                                        copex.send(rendered_prompt),
                                        timeout=task_timeout,
                                    )
                                finally:
                                    copex._session = None
                                    copex._client = None
                        else:
                            async with Copex(task_config) as copex:
                                response = await asyncio.wait_for(
                                    copex.send(rendered_prompt),
                                    timeout=task_timeout,
                                )

                    elapsed = (time.monotonic() - start) * 1000

                    # Extract cost tracking from response
                    prompt_tokens = response.prompt_tokens or 0
                    completion_tokens = response.completion_tokens or 0
                    total_cost = response.cost or 0.0

                    result = FleetResult(
                        task_id=task.id,
                        success=True,
                        response=response,
                        duration_ms=elapsed,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_cost=total_cost,
                        retries_used=retries_used,
                    )
                except Exception as exc:  # Catch-all: task failures are captured as FleetResult
                    elapsed = (time.monotonic() - start) * 1000
                    result = FleetResult(
                        task_id=task.id,
                        success=False,
                        error=exc,
                        duration_ms=elapsed,
                        retries_used=retries_used,
                    )
                    if config.fail_fast:
                        cancel_event.set()

                return result

            async def _run_task(task: FleetTask) -> FleetResult:
                """Run a task with retries, then update shared state."""
                result = await _run_task_with_retry(
                    task, config, _run_task_once, on_status=on_status, adaptive=adaptive
                )

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

            async def _worker() -> None:
                """Worker that pulls tasks from the shared deque."""
                while True:
                    # Try to steal from left (highest priority) first
                    task: FleetTask | None = None
                    try:
                        task = work_deque.popleft()
                    except IndexError:
                        break
                    await _run_task(task)

            # Launch worker pool sized to concurrency limit
            n_workers = min(config.max_concurrent, len(tasks))
            workers = [asyncio.create_task(_worker()) for _ in range(n_workers)]
            await asyncio.gather(*workers, return_exceptions=True)

        finally:
            # Clean up shared resources
            if pool is not None:
                await pool.stop()
            if client is not None:
                await client.stop()

        return [results[t.id] for t in tasks]

    def _task_config(self, task: FleetTask, fleet_cfg: FleetConfig) -> CopexConfig:
        """Build a per-task config with overrides applied."""
        instructions = self._base_config.instructions
        instructions_file = self._base_config.instructions_file
        if task.skills_dirs:
            skills_content = _load_skills_content(task.skills_dirs)
            if instructions is None and instructions_file:
                instructions_path = Path(instructions_file)
                if not instructions_path.exists():
                    raise FileNotFoundError(f"Instructions file not found: {instructions_path}")
                instructions = instructions_path.read_text(encoding="utf-8")

            combined_parts = []
            if skills_content:
                combined_parts.append(skills_content)
            if instructions:
                combined_parts.append(instructions)
            instructions = "\n\n".join(combined_parts) if combined_parts else None
            instructions_file = None

        task_timeout = task.timeout_sec if task.timeout_sec is not None else fleet_cfg.timeout
        task_cwd = task.cwd if task.cwd is not None else self._base_config.cwd
        task_working_dir = task.cwd if task.cwd is not None else self._base_config.working_directory
        task_skills = task.skills if task.skills is not None else self._base_config.skills
        if task.exclude_tools is None:
            task_excluded = self._base_config.excluded_tools
        else:
            task_excluded = list(
                dict.fromkeys([*self._base_config.excluded_tools, *task.exclude_tools])
            )
        task_mcp_servers = self._base_config.mcp_servers
        task_mcp_config_file = self._base_config.mcp_config_file
        if task.mcp_servers is not None:
            task_mcp_servers = _normalize_mcp_servers(task.mcp_servers) or []
            task_mcp_config_file = None

        cfg = CopexConfig(
            model=task.model or self._base_config.model,
            reasoning_effort=task.reasoning_effort or self._base_config.reasoning_effort,
            cli_path=self._base_config.cli_path,
            cli_url=self._base_config.cli_url,
            cwd=task_cwd,
            timeout=task_timeout,
            auto_continue=self._base_config.auto_continue,
            continue_prompt=self._base_config.continue_prompt,
            skills=task_skills,
            skill_directories=self._base_config.skill_directories,
            instructions=instructions,
            instructions_file=instructions_file,
            available_tools=self._base_config.available_tools,
            excluded_tools=task_excluded,
            mcp_servers=task_mcp_servers,
            mcp_config_file=task_mcp_config_file,
            working_directory=task_working_dir,
            retry=self._base_config.retry,
            use_cli=self._base_config.use_cli,
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
                f"  - Task '{tid}' depends on unknown task '{dep}'" for tid, dep in missing_deps
            )
            raise ValueError(
                f"Missing dependencies detected:\n{details}\nAvailable task IDs: {sorted(task_ids)}"
            )

        # Topological sort to detect cycles (Kahn's algorithm)
        in_degree: dict[str, int] = {t.id: 0 for t in tasks}
        adjacency: dict[str, list[str]] = {t.id: [] for t in tasks}
        for task in tasks:
            for dep in task.depends_on:
                adjacency[dep].append(task.id)
                in_degree[task.id] += 1

        queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
        visited = 0
        while queue:
            node = queue.popleft()
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
                "Cycle detected in fleet task dependencies. "
                "Tasks involved in cycle:\n" + "\n".join(cycle_details)
            )


@dataclass
class FleetSummary:
    """Aggregated summary of fleet execution results.

    Provides totals for tokens, cost, and timing across all tasks.
    """

    total_tasks: int = 0
    succeeded: int = 0
    failed: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_duration_ms: float = 0.0
    total_retries: int = 0

    @classmethod
    def from_results(cls, results: list[FleetResult]) -> FleetSummary:
        """Create a summary from a list of fleet results."""
        summary = cls(total_tasks=len(results))
        for r in results:
            if r.success:
                summary.succeeded += 1
            else:
                summary.failed += 1
            summary.total_prompt_tokens += r.prompt_tokens
            summary.total_completion_tokens += r.completion_tokens
            summary.total_cost += r.total_cost
            summary.total_duration_ms += r.duration_ms
            summary.total_retries += r.retries_used
        summary.total_tokens = summary.total_prompt_tokens + summary.total_completion_tokens
        return summary

    def __str__(self) -> str:
        """Human-readable summary string."""
        lines = [
            "═══ Fleet Summary ═══",
            f"Tasks: {self.succeeded}/{self.total_tasks} succeeded",
        ]
        if self.failed:
            lines.append(f"Failed: {self.failed}")
        if self.total_retries:
            lines.append(f"Retries used: {self.total_retries}")
        lines.extend(
            [
                f"Tokens: {self.total_prompt_tokens:,} prompt + {self.total_completion_tokens:,} completion = {self.total_tokens:,} total",
                f"Cost: ${self.total_cost:.4f}",
                f"Duration: {self.total_duration_ms / 1000:.1f}s",
            ]
        )
        return "\n".join(lines)


def summarize_fleet_results(results: list[FleetResult]) -> FleetSummary:
    """Create an aggregated summary from fleet results.

    Args:
        results: List of FleetResult from a fleet run.

    Returns:
        FleetSummary with aggregated totals.

    Example:
        results = await fleet.run()
        summary = summarize_fleet_results(results)
        print(summary)
    """
    return FleetSummary.from_results(results)


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
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._git_finalizer.finalize)
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
        on_dependency_failure: DependencyFailurePolicy | str = DependencyFailurePolicy.BLOCK,
        cwd: str | None = None,
        skills: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        mcp_servers: dict[str, Any] | list[dict[str, Any]] | None = None,
        timeout_sec: float | None = None,
        skills_dirs: list[str] | None = None,
        retries: int | None = None,
        retry_delay: float | None = None,
    ) -> str:
        """Add a task to the fleet.

        Args:
            prompt: The task prompt.
            task_id: Optional task ID (auto-generated from prompt if not provided).
            depends_on: List of task IDs this task depends on.
            model: Optional model override for this task.
            reasoning_effort: Optional reasoning effort override.
            on_dependency_failure: How to handle failed dependencies.
            cwd: Working directory for this task.
            skills: Skills to enable for this task.
            exclude_tools: Tools to exclude for this task.
            mcp_servers: MCP servers for this task.
            timeout_sec: Timeout in seconds for this task.
            skills_dirs: Skill directories for this task.
            retries: Number of retries for this task (falls back to global config).
            retry_delay: Base retry delay in seconds (falls back to global config).

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
            on_dependency_failure=_normalize_dep_failure_policy(on_dependency_failure),
            cwd=cwd,
            skills=skills,
            exclude_tools=exclude_tools,
            mcp_servers=mcp_servers,
            timeout_sec=timeout_sec,
            skills_dirs=skills_dirs or [],
            retries=retries,
            retry_delay=retry_delay,
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
                fleet._tasks.append(
                    FleetTask(
                        id=task_rec.task_id,
                        prompt=task_rec.prompt,
                        depends_on=remaining_deps,
                    )
                )

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

    def _init_store_run(self, tasks: list[FleetTask]) -> None:
        """Persist fleet run and tasks to store if configured.

        Skips if already resuming an existing run (``_run_id`` is set).
        """
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

    def _validate_tasks(self) -> None:
        """Validate all tasks upfront before execution.

        Checks:
        - skills_dirs paths exist and are directories
        - MCP server configs have required fields
        - Prompt template {{task:id.field}} references valid task IDs
        """
        task_ids = {t.id for t in self._tasks}
        errors: list[str] = []

        for task in self._tasks:
            # Validate skills_dirs
            for dir_path in task.skills_dirs:
                p = Path(dir_path)
                if not p.exists():
                    errors.append(
                        f"Task '{task.id}': skills directory not found: {p}"
                    )
                elif not p.is_dir():
                    errors.append(
                        f"Task '{task.id}': skills path is not a directory: {p}"
                    )

            # Validate MCP server configs
            if task.mcp_servers is not None:
                try:
                    normalized = _normalize_mcp_servers(task.mcp_servers)
                except ValueError as exc:
                    errors.append(f"Task '{task.id}': {exc}")
                    normalized = None
                if normalized:
                    for i, server in enumerate(normalized):
                        if not server.get("command") and not server.get("url"):
                            name = server.get("name", f"server[{i}]")
                            errors.append(
                                f"Task '{task.id}': MCP server '{name}' "
                                f"requires 'command' or 'url'"
                            )

            # Validate prompt template references
            for match in _TASK_OUTPUT_REF_RE.finditer(task.prompt):
                ref_id = match.group("task_id")
                if ref_id not in task_ids:
                    errors.append(
                        f"Task '{task.id}': prompt references unknown task "
                        f"'{{{{task:{ref_id}.{match.group('field')}}}}}'"
                    )

        if errors:
            detail = "\n".join(f"  - {e}" for e in errors)
            raise ValueError(f"Fleet validation failed:\n{detail}")

    async def run(
        self,
        on_status: Callable[[str, str], None] | None = None,
    ) -> list[FleetResult]:
        """Execute all tasks and return results."""
        if not self._tasks:
            return []

        self._validate_tasks()

        # Prepend shared context without mutating original tasks
        tasks = _prepend_shared_context(self._tasks, self._fleet_config.shared_context)

        # Persist to store if configured (skip if resuming an existing run)
        self._init_store_run(tasks)

        # Wrap on_status to also update store
        tracking_on_status = _make_tracking_on_status(on_status, self._store, self._run_id)

        results = await self._coordinator.run(
            tasks,
            self._fleet_config,
            on_status=tracking_on_status,
        )

        # Record results in store
        if self._store is not None and self._run_id is not None:
            for result in results:
                self._store.record_result(
                    self._run_id,
                    result.task_id,
                    success=result.success,
                    content=result.response.content if result.response else None,
                    error=str(result.error) if result.error else None,
                    duration_ms=result.duration_ms,
                )
            self._store.complete_run(self._run_id)

        # Git finalize: track modified files from successful responses
        _track_git_files(self._git_finalizer, results)

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

        self._validate_tasks()

        # Initialize context and mailbox if provided
        ctx = context or FleetContext()
        mbox = mailbox or FleetMailbox()

        # Create inboxes for all tasks
        for task in self._tasks:
            mbox.create_inbox(task.id)

        # Prepend shared context without mutating original tasks
        tasks = _prepend_shared_context(self._tasks, self._fleet_config.shared_context)

        # Persist to store if configured (skip if resuming an existing run)
        self._init_store_run(tasks)
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

        # Adaptive concurrency management
        adaptive = (
            AdaptiveConcurrency(
                initial=self._fleet_config.max_concurrent,
                minimum=self._fleet_config.min_concurrent,
                restore_after=self._fleet_config.concurrency_restore_after,
            )
            if self._fleet_config.adaptive_concurrency
            else None
        )

        semaphore: DynamicSemaphore | asyncio.Semaphore = (
            adaptive.semaphore if adaptive else asyncio.Semaphore(self._fleet_config.max_concurrent)
        )
        cancel_event = asyncio.Event()
        done_events: dict[str, asyncio.Event] = {t.id: asyncio.Event() for t in tasks}
        finished: dict[str, bool] = {}

        # Prioritize: shallowest deps first, shortest prompt first
        prioritized = FleetCoordinator._prioritize_tasks(tasks)

        async def _run_task_once(task: FleetTask, retries_used: int = 0) -> FleetResult:
            """Execute a single prompt attempt (no dependency handling)."""
            if cancel_event.is_set():
                return FleetResult(
                    task_id=task.id,
                    success=False,
                    error=asyncio.CancelledError("Fleet cancelled"),
                )

            start = time.monotonic()
            try:
                rendered_prompt = _render_prompt_with_task_outputs(
                    task.prompt,
                    current_task_id=task.id,
                    results=results,
                )
                task_config = self._coordinator._task_config(task, self._fleet_config)
                task_timeout = (
                    task.timeout_sec if task.timeout_sec is not None else self._fleet_config.timeout
                )
                async with semaphore:
                    if cancel_event.is_set():
                        raise asyncio.CancelledError("Fleet cancelled")

                    # Use CLI client when use_cli is set
                    client_factory = make_client if task_config.use_cli else Copex
                    async with client_factory(task_config) as copex:
                        # Build streaming callback if needed
                        on_chunk = None
                        if include_deltas and not task_config.use_cli:
                            # Sync wrapper for the streaming callback
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
                            copex.send(
                                rendered_prompt, on_chunk=on_chunk if include_deltas else None
                            ),
                            timeout=task_timeout,
                        )

                elapsed = (time.monotonic() - start) * 1000

                # Extract cost tracking from response
                prompt_tokens = response.prompt_tokens or 0
                completion_tokens = response.completion_tokens or 0
                total_cost = response.cost or 0.0

                return FleetResult(
                    task_id=task.id,
                    success=True,
                    response=response,
                    duration_ms=elapsed,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_cost=total_cost,
                    retries_used=retries_used,
                )

            except Exception as exc:  # Catch-all: task failures are captured as FleetResult
                elapsed = (time.monotonic() - start) * 1000
                result = FleetResult(
                    task_id=task.id,
                    success=False,
                    error=exc,
                    duration_ms=elapsed,
                    retries_used=retries_used,
                )
                if self._fleet_config.fail_fast:
                    cancel_event.set()
                return result

        async def _run_task(task: FleetTask) -> FleetResult:
            """Run a task: check deps, retry execution, record results."""
            # Emit queued event
            await event_queue.put(
                FleetEvent(
                    event_type=FleetEventType.TASK_QUEUED,
                    task_id=task.id,
                    data={"prompt_preview": task.prompt[:100]},
                )
            )
            if store is not None and run_id is not None:
                store.update_task_status(run_id, task.id, "pending")

            # Wait for dependencies
            if task.depends_on:
                await event_queue.put(
                    FleetEvent(
                        event_type=FleetEventType.TASK_WAITING,
                        task_id=task.id,
                        data={"waiting_for": task.depends_on},
                    )
                )

            dep_policy = _normalize_dep_failure_policy(task.on_dependency_failure)
            dep_wait_timeout = (
                self._fleet_config.dep_timeout
                or (
                    task.timeout_sec
                    if task.timeout_sec is not None
                    else self._fleet_config.timeout
                )
            )
            for dep in task.depends_on:
                try:
                    await asyncio.wait_for(
                        done_events[dep].wait(), timeout=dep_wait_timeout
                    )
                except asyncio.TimeoutError:
                    error_msg = (
                        f"Task '{task.id}' timed out waiting for dependency '{dep}' "
                        f"after {dep_wait_timeout}s"
                    )
                    result = FleetResult(
                        task_id=task.id,
                        success=False,
                        error=RuntimeError(error_msg),
                    )
                    results[task.id] = result
                    finished[task.id] = False
                    done_events[task.id].set()
                    await event_queue.put(
                        FleetEvent(
                            event_type=FleetEventType.TASK_BLOCKED,
                            task_id=task.id,
                            error=error_msg,
                        )
                    )
                    if store is not None and run_id is not None:
                        store.update_task_status(run_id, task.id, "blocked")
                    return result
            failed_deps = [d for d in task.depends_on if not finished.get(d, False)]
            if failed_deps and dep_policy == DependencyFailurePolicy.BLOCK:
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
                await event_queue.put(
                    FleetEvent(
                        event_type=FleetEventType.TASK_BLOCKED,
                        task_id=task.id,
                        error=f"Blocked by failed dependencies: {failed_deps}",
                    )
                )
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
                await event_queue.put(
                    FleetEvent(
                        event_type=FleetEventType.TASK_CANCELLED,
                        task_id=task.id,
                    )
                )
                return result

            # Emit running event
            await event_queue.put(
                FleetEvent(
                    event_type=FleetEventType.TASK_RUNNING,
                    task_id=task.id,
                )
            )
            if store is not None and run_id is not None:
                store.update_task_status(run_id, task.id, "running")

            # Execute with retries
            def _on_status(task_id: str, status: str) -> None:
                if store is not None and run_id is not None:
                    store.update_task_status(run_id, task_id, status)

            result = await _run_task_with_retry(
                task, self._fleet_config, _run_task_once,
                on_status=_on_status, adaptive=adaptive,
            )

            results[task.id] = result
            finished[task.id] = result.success
            done_events[task.id].set()

            # Emit completion event
            if result.success:
                await event_queue.put(
                    FleetEvent(
                        event_type=FleetEventType.TASK_DONE,
                        task_id=task.id,
                        data={
                            "duration_ms": result.duration_ms,
                            "content_preview": (
                                result.response.content[:200] if result.response else ""
                            )[:200],
                        },
                    )
                )
                if store is not None and run_id is not None:
                    store.record_result(
                        run_id,
                        task.id,
                        success=True,
                        content=result.response.content if result.response else None,
                        duration_ms=result.duration_ms,
                    )
            else:
                await event_queue.put(
                    FleetEvent(
                        event_type=FleetEventType.TASK_FAILED,
                        task_id=task.id,
                        error=str(result.error) if result.error else "Unknown error",
                        data={"duration_ms": result.duration_ms},
                    )
                )
                if store is not None and run_id is not None:
                    store.record_result(
                        run_id,
                        task.id,
                        success=False,
                        error=str(result.error) if result.error else None,
                        duration_ms=result.duration_ms,
                    )

            # Store in context if provided
            if context:
                ctx.add_result_sync(
                    task.id,
                    {
                        "success": result.success,
                        "content": result.response.content if result.response else None,
                        "error": str(result.error) if result.error else None,
                        "duration_ms": result.duration_ms,
                    },
                )

            return result

        # Start all tasks in priority order
        task_futures = [asyncio.create_task(_run_task(task)) for task in prioritized]

        # Collector task that signals completion
        async def _wait_all() -> None:
            await asyncio.gather(*task_futures, return_exceptions=True)
            await event_queue.put(None)  # Signal end

        collector = asyncio.create_task(_wait_all())

        # Yield events as they arrive
        try:
            while True:
                event = await event_queue.get()
                if event is None:
                    break
                yield event
        finally:
            # Cancel all individual task futures on early exit
            for fut in task_futures:
                fut.cancel()
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

        # Git finalize: track modified files from successful responses
        _track_git_files(self._git_finalizer, results.values())

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
