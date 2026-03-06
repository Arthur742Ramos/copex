from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from copex import sdk_tools
from copex.memory import ProjectMemory
from copex.sdk_tools import (
    build_domain_tools,
    list_domain_tools,
    register_proof_checker,
    register_test_runner,
)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_hooks() -> None:
    register_proof_checker(None)
    register_test_runner(None)


def _invocation(args: dict) -> dict:
    return {
        "session_id": "session-1",
        "tool_call_id": "call-1",
        "tool_name": "tool",
        "arguments": args,
    }


def test_builtin_domain_tools_registered() -> None:
    names = list_domain_tools()
    assert "memory_search" in names
    assert "test_runner" in names
    assert "proof_checker" in names


def test_memory_search_tool_reads_project_memory(tmp_path: Path) -> None:
    memory = ProjectMemory(root=tmp_path)
    memory.add_entry("Always run lint before push", kind="preference")

    tool = build_domain_tools(["memory_search"], working_dir=tmp_path)[0]
    result = run(tool.handler(_invocation({"query": "lint", "limit": 5})))

    assert result["resultType"] == "success"
    assert "lint" in result["textResultForLlm"].lower()


def test_proof_checker_requires_registered_hook(tmp_path: Path) -> None:
    tool = build_domain_tools(["proof_checker"], working_dir=tmp_path)[0]
    result = run(tool.handler(_invocation({"claim": "forall n, n = n"})))

    assert result["resultType"] == "failure"
    assert "No proof checker is configured" in result["textResultForLlm"]


def test_proof_checker_uses_registered_hook(tmp_path: Path) -> None:
    register_proof_checker(lambda args, _cwd: f"checked: {args['claim']}")

    tool = build_domain_tools(["proof_checker"], working_dir=tmp_path)[0]
    result = run(tool.handler(_invocation({"claim": "A -> A"})))

    assert result["resultType"] == "success"
    assert "checked: A -> A" in result["textResultForLlm"]


def test_test_runner_uses_registered_hook(tmp_path: Path) -> None:
    register_test_runner(lambda _args, _cwd: {"resultType": "success", "textResultForLlm": "ok"})

    tool = build_domain_tools(["test_runner"], working_dir=tmp_path)[0]
    result = run(tool.handler(_invocation({"command": "pytest -q"})))

    assert result["resultType"] == "success"
    assert result["textResultForLlm"] == "ok"


def test_test_runner_rejects_arbitrary_commands(tmp_path: Path) -> None:
    tool = build_domain_tools(["test_runner"], working_dir=tmp_path)[0]

    result = run(tool.handler(_invocation({"command": "bash -lc 'echo nope'"})))

    assert result["resultType"] == "failure"
    assert "Only supported project test commands are allowed" in result["textResultForLlm"]


def test_test_runner_allows_uv_pytest_commands(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_run(argv: list[str], *, cwd: Path, timeout_seconds: float):
        captured["argv"] = argv
        captured["cwd"] = cwd
        captured["timeout_seconds"] = timeout_seconds
        return 0, "ok", ""

    monkeypatch.setattr(sdk_tools, "_run_test_command", fake_run)
    tool = build_domain_tools(["test_runner"], working_dir=tmp_path)[0]

    result = run(tool.handler(_invocation({"command": "uv run pytest -q", "cwd": "."})))

    assert result["resultType"] == "success"
    assert captured["argv"] == ["uv", "run", "pytest", "-q"]
    assert captured["cwd"] == tmp_path.resolve()
