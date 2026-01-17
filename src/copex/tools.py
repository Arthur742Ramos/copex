"""
Parallel Tools - Execute multiple tool calls concurrently.

Enables:
- Concurrent execution of independent tools
- Batching of tool results
- Timeout handling for slow tools
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from functools import wraps


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
            {"name": name, "description": self._descriptions.get(name, "")}
            for name in self._tools
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

        try:
            result = await asyncio.wait_for(
                tool(**params),
                timeout=timeout,
            )
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

        except Exception as e:
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

        async def limited_execute(name: str, params: dict) -> ToolResult:
            async with semaphore:
                return await self.execute(name, params)

        tasks = [limited_execute(name, params) for name, params in calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to ToolResults
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                name = calls[i][0]
                final_results.append(ToolResult(
                    name=name,
                    success=False,
                    error=str(result),
                ))
            else:
                final_results.append(result)

        return final_results

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
                await asyncio.sleep(2 ** attempt * 0.5)

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
        calls = [
            (call["name"], call.get("arguments", {}))
            for call in tool_calls
        ]

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
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"

                properties[param_name] = {"type": param_type}

                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            definitions.append({
                "name": tool_info["name"],
                "description": tool_info["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })

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
