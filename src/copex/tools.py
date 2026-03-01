"""
Parallel Tools - Execute multiple tool calls concurrently.

Enables:
- Concurrent execution of independent tools
- Batching of tool results
- Timeout handling for slow tools
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ToolResult:
    """Result from a tool execution."""

    name: str
    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: float = 0


@dataclass
class ParallelToolConfig:
    """Configuration for parallel tool execution."""

    max_concurrent: int = 5  # Maximum concurrent tool calls
    timeout: float = 30.0  # Timeout per tool in seconds
    fail_fast: bool = False  # Stop on first error
    retry_on_error: bool = True  # Retry failed tools
    max_retries: int = 2  # Max retries per tool
    approval_workflow: Any | None = None  # Optional ApprovalWorkflow for write operations


_PATH_KEYS = (
    "path",
    "file_path",
    "file",
    "filepath",
    "target_file",
    "target",
    "filename",
    "relative_path",
    "absolute_path",
)


def _extract_candidate_path(payload: dict[str, Any]) -> str | None:
    for key in _PATH_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _normalize_path(raw_path: str, cwd: Path) -> str:
    value = raw_path.strip()
    if not value:
        return ""
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
        cwd_resolved = cwd.resolve(strict=False)
        try:
            return resolved.relative_to(cwd_resolved).as_posix()
        except ValueError:
            return resolved.as_posix()
    return candidate.as_posix()


def _review_path(review: Any) -> str | None:
    proposal = getattr(review, "proposal", None)
    display_path = getattr(proposal, "display_path", None)
    if isinstance(display_path, str) and display_path.strip():
        return display_path.replace("\\", "/")
    preview = getattr(review, "preview", None)
    file_path = getattr(preview, "file_path", None)
    if isinstance(file_path, str) and file_path.strip():
        return file_path.replace("\\", "/")
    return None


def _review_target(review: Any) -> str:
    return _review_path(review) or "change"


def _prepare_approved_execution(
    *,
    params: dict[str, Any],
    reviews: list[Any],
    cwd: Path,
) -> tuple[dict[str, Any], list[Any], list[Any], bool]:
    if not reviews:
        return params, [], [], False

    approved_reviews = [review for review in reviews if bool(getattr(review, "apply_change", True))]
    blocked_reviews = [review for review in reviews if not bool(getattr(review, "apply_change", True))]

    if not approved_reviews:
        return params, [], blocked_reviews, True
    if not blocked_reviews:
        return params, approved_reviews, [], False

    # Structured patch strings are hard to partially filter safely. Block the call
    # when any reviewed file was rejected/deferred.
    if any(key in params for key in ("patch", "diff")):
        return params, [], reviews, True

    approved_paths = {
        path for review in approved_reviews if (path := _review_path(review)) is not None and path != ""
    }
    if not approved_paths:
        return params, [], reviews, True

    execution_params = dict(params)
    kept_paths: set[str] = set()
    had_structured_list = False

    for key in ("changes", "files"):
        value = params.get(key)
        if not isinstance(value, list):
            continue
        had_structured_list = True
        kept_items: list[Any] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            raw_path = _extract_candidate_path(item)
            if raw_path is None:
                continue
            normalized = _normalize_path(raw_path, cwd)
            if normalized in approved_paths:
                kept_items.append(item)
                kept_paths.add(normalized)
        execution_params[key] = kept_items

    top_level_path = _extract_candidate_path(params)
    if top_level_path is not None:
        normalized_top_path = _normalize_path(top_level_path, cwd)
        if normalized_top_path not in approved_paths:
            return params, [], reviews, True
        kept_paths.add(normalized_top_path)
    elif not had_structured_list:
        # We have mixed decisions, but no way to safely isolate the approved subset.
        return params, [], reviews, True

    if had_structured_list and not any(execution_params.get(key) for key in ("changes", "files")):
        return execution_params, [], reviews, True

    executable_reviews: list[Any] = []
    skipped_reviews: list[Any] = []
    for review in reviews:
        review_path = _review_path(review)
        if bool(getattr(review, "apply_change", True)):
            if review_path is None or review_path in kept_paths or not had_structured_list:
                executable_reviews.append(review)
            else:
                skipped_reviews.append(review)
        else:
            skipped_reviews.append(review)

    if not executable_reviews:
        return execution_params, [], reviews, True

    return execution_params, executable_reviews, skipped_reviews, False


class ToolRegistry:
    """
    Registry for tools that can be executed in parallel.

    Usage:
        registry = ToolRegistry()

        @registry.register("get_weather")
        async def get_weather(city: str) -> str:
            return f"Weather in {city}: Sunny"

        @registry.register("get_time")
        async def get_time(timezone: str) -> str:
            return f"Time in {timezone}: 12:00"

        # Execute multiple tools in parallel
        results = await registry.execute_parallel([
            ("get_weather", {"city": "Seattle"}),
            ("get_time", {"timezone": "PST"}),
        ])
    """

    def __init__(self, config: ParallelToolConfig | None = None):
        self.config = config or ParallelToolConfig()
        self._tools: dict[str, Callable[..., Awaitable[Any]]] = {}
        self._descriptions: dict[str, str] = {}

    def register(
        self,
        name: str,
        description: str = "",
    ) -> Callable[[Callable], Callable]:
        """
        Decorator to register a tool.

        Args:
            name: Tool name
            description: Tool description

        Example:
            @registry.register("search", "Search the web")
            async def search(query: str) -> str:
                ...
        """

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable:
            self._tools[name] = func
            self._descriptions[name] = description or func.__doc__ or ""
            return func

        return decorator

    def add_tool(
        self,
        name: str,
        func: Callable[..., Awaitable[Any]],
        description: str = "",
    ) -> None:
        """Add a tool directly (not as decorator)."""
        self._tools[name] = func
        self._descriptions[name] = description or func.__doc__ or ""

    def get_tool(self, name: str) -> Callable[..., Awaitable[Any]] | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        """List all registered tools."""
        return [
            {"name": name, "description": self._descriptions.get(name, "")} for name in self._tools
        ]

    async def execute(
        self,
        name: str,
        params: dict[str, Any],
        timeout: float | None = None,
    ) -> ToolResult:
        """
        Execute a single tool.

        Args:
            name: Tool name
            params: Tool parameters
            timeout: Optional timeout override

        Returns:
            ToolResult with success/failure info
        """
        import time

        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                name=name,
                success=False,
                error=f"Tool not found: {name}",
            )

        start = time.time()
        timeout = timeout or self.config.timeout
        approval_reviews: list[Any] = []

        review_tool_call = getattr(self.config.approval_workflow, "review_tool_call", None)
        if callable(review_tool_call):
            approval_reviews = list(review_tool_call(name, params))
            if approval_reviews and all(
                not bool(getattr(review, "apply_change", True)) for review in approval_reviews
            ):
                skipped = [
                    str(getattr(review.preview, "file_path", "change"))
                    for review in approval_reviews
                    if getattr(review, "preview", None) is not None
                ]
                target = ", ".join(skipped) if skipped else "tool call"
                return ToolResult(
                    name=name,
                    success=True,
                    result=f"Skipped by approval workflow: {target}",
                )

        try:
            result = await asyncio.wait_for(
                tool(**params),
                timeout=timeout,
            )
            apply_post = getattr(self.config.approval_workflow, "apply_post_tool_decisions", None)
            if callable(apply_post) and approval_reviews:
                post_messages = list(apply_post(approval_reviews))
                if post_messages:
                    suffix = "; ".join(post_messages)
                    if isinstance(result, str):
                        result = f"{result}\n{suffix}"
                    else:
                        result = {"result": result, "approval": post_messages}
            duration = (time.time() - start) * 1000

            return ToolResult(
                name=name,
                success=True,
                result=result,
                duration_ms=duration,
            )

        except asyncio.TimeoutError:
            duration = (time.time() - start) * 1000
            return ToolResult(
                name=name,
                success=False,
                error=f"Timeout after {timeout}s",
                duration_ms=duration,
            )

        except Exception as e:  # Catch-all: tool execution failures become ToolResult
            duration = (time.time() - start) * 1000
            return ToolResult(
                name=name,
                success=False,
                error=str(e),
                duration_ms=duration,
            )

    async def execute_parallel(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        max_concurrent: int | None = None,
    ) -> list[ToolResult]:
        """
        Execute multiple tools in parallel.

        Args:
            calls: List of (tool_name, params) tuples
            max_concurrent: Override max concurrent limit

        Returns:
            List of ToolResult in same order as calls
        """
        max_concurrent = max_concurrent or self.config.max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_execute(idx: int, name: str, params: dict) -> tuple[int, ToolResult]:
            async with semaphore:
                if self.config.retry_on_error:
                    return idx, await self.execute_with_retry(name, params)
                return idx, await self.execute(name, params)

        tasks: list[asyncio.Task[tuple[int, ToolResult]]] = []
        for idx, (name, params) in enumerate(calls):
            task = asyncio.create_task(limited_execute(idx, name, params))
            tasks.append(task)

        results: list[ToolResult | None] = [None] * len(calls)
        pending = set(tasks)

        try:
            done_tasks, _ = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED) if not self.config.fail_fast else (set(), set())
            if self.config.fail_fast:
                # Process tasks as they complete to support fail_fast
                while pending:
                    done_set, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done_set:
                        try:
                            idx, result = task.result()
                        except Exception as exc:
                            # Find the index from any remaining approach
                            idx = tasks.index(task)
                            result = ToolResult(
                                name=calls[idx][0],
                                success=False,
                                error=str(exc),
                            )
                        results[idx] = result
                        if self.config.fail_fast and not result.success:
                            for pending_task in pending:
                                pending_task.cancel()
                            pending = set()
                            break
            else:
                for task in done_tasks:
                    try:
                        idx, result = task.result()
                    except Exception as exc:
                        idx = tasks.index(task)
                        result = ToolResult(
                            name=calls[idx][0],
                            success=False,
                            error=str(exc),
                        )
                    results[idx] = result
        finally:
            if pending:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        if self.config.fail_fast and any(r is None for r in results):
            for idx, result in enumerate(results):
                if result is None:
                    results[idx] = ToolResult(
                        name=calls[idx][0],
                        success=False,
                        error="Cancelled due to fail_fast",
                    )

        return [result for result in results if result is not None]

    async def execute_with_retry(
        self,
        name: str,
        params: dict[str, Any],
        max_retries: int | None = None,
    ) -> ToolResult:
        """
        Execute a tool with retries on failure.

        Args:
            name: Tool name
            params: Tool parameters
            max_retries: Override max retries

        Returns:
            ToolResult from last attempt
        """
        max_retries = max_retries or self.config.max_retries

        for attempt in range(max_retries + 1):
            result = await self.execute(name, params)

            if result.success:
                return result

            if attempt < max_retries:
                # Exponential backoff
                await asyncio.sleep(2**attempt * 0.5)

        return result


class ParallelToolExecutor:
    """
    High-level executor for parallel tool calls from Copex responses.

    Integrates with Copex to automatically handle tool calls in parallel.

    Usage:
        executor = ParallelToolExecutor()

        @executor.tool("fetch_data")
        async def fetch_data(url: str) -> str:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()

        # In Copex callback
        async def handle_tools(tool_calls: list[dict]) -> list[dict]:
            return await executor.handle_batch(tool_calls)
    """

    def __init__(self, config: ParallelToolConfig | None = None):
        self.registry = ToolRegistry(config)
        self.config = config or ParallelToolConfig()

    def tool(
        self,
        name: str,
        description: str = "",
    ) -> Callable[[Callable], Callable]:
        """Decorator to register a tool."""
        return self.registry.register(name, description)

    async def handle_batch(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Handle a batch of tool calls from Copex.

        Args:
            tool_calls: List of tool call dicts with "name" and "arguments"

        Returns:
            List of result dicts for Copex
        """
        calls = [(call["name"], call.get("arguments", {})) for call in tool_calls]

        results = await self.registry.execute_parallel(calls)

        return [
            {
                "tool_call_id": tool_calls[i].get("id", f"call_{i}"),
                "name": result.name,
                "result": result.result if result.success else None,
                "error": result.error,
                "success": result.success,
            }
            for i, result in enumerate(results)
        ]

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Get tool definitions for Copex session.

        Returns:
            List of tool definitions for create_session
        """
        definitions = []
        for tool_info in self.registry.list_tools():
            # Get the actual function to introspect
            func = self.registry.get_tool(tool_info["name"])
            if not func:
                continue

            # Try to get type hints for parameters
            import inspect

            sig = inspect.signature(func)
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue

                param_type = "string"
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation is int:
                        param_type = "integer"
                    elif param.annotation is float:
                        param_type = "number"
                    elif param.annotation is bool:
                        param_type = "boolean"

                properties[param_name] = {"type": param_type}

                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            definitions.append(
                {
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            )

        return definitions


def parallel_tools(*tools: Callable) -> list[Callable]:
    """
    Convenience wrapper to mark tools for parallel execution.

    Usage:
        from copex.tools import parallel_tools

        tools = parallel_tools(get_weather, get_time, fetch_data)

        async with Copex() as copex:
            response = await copex.send("...", tools=tools)
    """
    return list(tools)


def read_text_file(path: Path, *, encoding: str = "utf-8") -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding=encoding)


def write_text_file(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write a UTF-8 text file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)


def write_text_file_atomic(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write a text file atomically via a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=path.parent,
            delete=False,
        ) as tmp:
            tmp.write(content)
            temp_path = Path(tmp.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
