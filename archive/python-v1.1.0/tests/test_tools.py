from __future__ import annotations

import asyncio

from copex.tools import ParallelToolConfig, ToolRegistry


async def _echo(text: str) -> str:
    return text


async def _slow(text: str, delay: float = 0.05) -> str:
    await asyncio.sleep(delay)
    return text


def test_tool_registry_register_and_list() -> None:
    registry = ToolRegistry()

    @registry.register("echo", "Echo tool")
    async def echo(text: str) -> str:
        return text

    tools = registry.list_tools()
    assert tools[0]["name"] == "echo"
    assert "Echo tool" in tools[0]["description"]
    assert registry.get_tool("echo") is echo


def test_tool_registry_add_tool_and_execute() -> None:
    registry = ToolRegistry()
    registry.add_tool("echo", _echo, description="Echo tool")

    result = asyncio.run(registry.execute("echo", {"text": "hi"}))
    assert result.success is True
    assert result.result == "hi"


def test_tool_registry_execute_timeout() -> None:
    registry = ToolRegistry(ParallelToolConfig(timeout=0.01))
    registry.add_tool("slow", _slow)

    result = asyncio.run(registry.execute("slow", {"text": "hi", "delay": 0.05}))
    assert result.success is False
    assert "Timeout" in (result.error or "")


def test_tool_registry_execute_parallel() -> None:
    registry = ToolRegistry()
    registry.add_tool("echo", _echo)

    results = asyncio.run(registry.execute_parallel([
        ("echo", {"text": "a"}),
        ("echo", {"text": "b"}),
    ]))

    assert [r.result for r in results] == ["a", "b"]


def test_tool_registry_tool_not_found() -> None:
    registry = ToolRegistry()
    result = asyncio.run(registry.execute("missing", {}))
    assert result.success is False
    assert "Tool not found" in (result.error or "")
