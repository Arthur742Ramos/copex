"""SDK-native domain tool registration for Copex."""

from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from copilot import Tool

from copex.memory import ProjectMemory

DomainToolFactory = Callable[[Path], Tool]
ProofCheckerHook = Callable[[dict[str, Any], Path], dict[str, Any] | str]
TestRunnerHook = Callable[[dict[str, Any], Path], dict[str, Any] | str]

_MAX_TEXT_RESULT_CHARS = 8_000
_MAX_MEMORY_ENTRY_CHARS = 400
_DEFAULT_TEST_TIMEOUT_SECONDS = 300.0

_proof_checker_hook: ProofCheckerHook | None = None
_test_runner_hook: TestRunnerHook | None = None

_tool_factories: dict[str, DomainToolFactory] = {}


def _clip_text(value: Any, *, limit: int = _MAX_TEXT_RESULT_CHARS) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _success_result(
    text: Any,
    *,
    session_log: str | None = None,
    telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "resultType": "success",
        "textResultForLlm": _clip_text(text),
    }
    if session_log:
        result["sessionLog"] = _clip_text(session_log, limit=1_000)
    if telemetry:
        result["toolTelemetry"] = telemetry
    return result


def _failure_result(
    text: Any,
    *,
    error: str | None = None,
    session_log: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "resultType": "failure",
        "textResultForLlm": _clip_text(text),
    }
    if error:
        result["error"] = _clip_text(error, limit=1_000)
    if session_log:
        result["sessionLog"] = _clip_text(session_log, limit=1_000)
    return result


def register_proof_checker(handler: ProofCheckerHook | None) -> None:
    """Register a pluggable proof-checker callback used by the proof_checker tool."""

    global _proof_checker_hook
    _proof_checker_hook = handler


def register_test_runner(handler: TestRunnerHook | None) -> None:
    """Register a pluggable test-runner callback used by the test_runner tool."""

    global _test_runner_hook
    _test_runner_hook = handler


def register_domain_tool(
    name: str,
    factory: DomainToolFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a domain tool factory by name."""

    normalized = name.strip()
    if not normalized:
        raise ValueError("Domain tool name cannot be empty")
    if normalized in _tool_factories and not replace:
        raise ValueError(f"Domain tool already registered: {normalized}")
    _tool_factories[normalized] = factory


def list_domain_tools() -> list[str]:
    """Return sorted names of all registered domain tools."""

    return sorted(_tool_factories)


def build_domain_tools(
    tool_names: Sequence[str] | None,
    *,
    working_dir: Path,
) -> list[Tool]:
    """Build SDK Tool instances for the requested domain tools."""

    names = [name.strip() for name in (tool_names or list_domain_tools()) if name.strip()]
    unknown = [name for name in names if name not in _tool_factories]
    if unknown:
        known = ", ".join(list_domain_tools())
        raise ValueError(f"Unknown domain tools: {', '.join(unknown)}. Known tools: {known}")
    return [_tool_factories[name](working_dir) for name in names]


async def _run_test_command(
    argv: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    return int(process.returncode or 0), stdout, stderr


def _build_memory_search_tool(working_dir: Path) -> Tool:
    async def _handler(invocation: dict[str, Any]) -> dict[str, Any]:
        args = invocation.get("arguments") or {}
        query = str(args.get("query", "")).strip()
        if not query:
            return _failure_result(
                "Parameter 'query' is required for memory_search.",
                error="missing query",
            )
        try:
            limit = int(args.get("limit", 5))
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 20))

        memory = ProjectMemory(root=working_dir)
        matches = [
            entry
            for entry in memory.parse_entries()
            if query.lower() in entry.text.lower() or query.lower() in entry.kind.lower()
        ]
        matches = matches[-limit:]

        if not matches:
            return _success_result(
                f"No memory entries matched '{query}'.",
                session_log=f"memory_search query={query!r} matches=0",
            )

        lines = [
            f"- [{entry.timestamp}] [{entry.kind}] {_clip_text(entry.text, limit=_MAX_MEMORY_ENTRY_CHARS)}"
            for entry in matches
        ]
        return _success_result(
            "\n".join(lines),
            session_log=f"memory_search query={query!r} matches={len(matches)}",
            telemetry={"match_count": len(matches)},
        )

    return Tool(
        name="memory_search",
        description="Search structured entries from Copex project memory.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to search for in memory entries"},
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matching entries to return",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["query"],
        },
        handler=_handler,
    )


def _build_test_runner_tool(working_dir: Path) -> Tool:
    async def _handler(invocation: dict[str, Any]) -> dict[str, Any]:
        args = invocation.get("arguments") or {}

        if _test_runner_hook is not None:
            try:
                hook_result = _test_runner_hook(args, working_dir)
            except Exception as exc:
                return _failure_result(
                    "test_runner hook failed.",
                    error=str(exc),
                    session_log="test_runner hook exception",
                )
            if isinstance(hook_result, dict):
                return hook_result
            return _success_result(
                hook_result,
                session_log="test_runner hook result",
            )

        command = str(args.get("command") or "pytest -q").strip()
        if not command:
            return _failure_result("Parameter 'command' cannot be empty.", error="empty command")

        try:
            timeout_seconds = float(args.get("timeout_seconds", _DEFAULT_TEST_TIMEOUT_SECONDS))
        except (TypeError, ValueError):
            timeout_seconds = _DEFAULT_TEST_TIMEOUT_SECONDS
        timeout_seconds = max(1.0, min(timeout_seconds, 1800.0))

        command_cwd = working_dir
        if "cwd" in args and args["cwd"]:
            candidate = Path(str(args["cwd"]))
            if not candidate.is_absolute():
                candidate = (working_dir / candidate).resolve()
            else:
                candidate = candidate.resolve()
            if not str(candidate).startswith(str(working_dir.resolve())):
                return _failure_result(
                    "test_runner cwd must stay within the configured working directory.",
                    error="cwd outside working directory",
                )
            command_cwd = candidate

        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return _failure_result("Invalid test command syntax.", error=str(exc))
        if not argv:
            return _failure_result("Parameter 'command' cannot be empty.", error="empty argv")

        try:
            return_code, stdout, stderr = await _run_test_command(
                argv,
                cwd=command_cwd,
                timeout_seconds=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return _failure_result(
                f"Test command timed out after {timeout_seconds:.0f}s.",
                error="timeout",
                session_log=f"test_runner timeout command={command!r}",
            )
        except OSError as exc:
            return _failure_result(
                "Failed to execute test command.",
                error=str(exc),
                session_log=f"test_runner execution failure command={command!r}",
            )

        output = {
            "command": command,
            "cwd": str(command_cwd),
            "return_code": return_code,
            "stdout": _clip_text(stdout),
            "stderr": _clip_text(stderr),
        }
        text = json.dumps(output, ensure_ascii=True)
        if return_code == 0:
            return _success_result(
                text,
                session_log=f"test_runner success command={command!r}",
                telemetry={"return_code": return_code},
            )
        return _failure_result(
            text,
            error=f"command exited with status {return_code}",
            session_log=f"test_runner failure command={command!r}",
        )

    return Tool(
        name="test_runner",
        description="Run project tests and return structured pass/fail output.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute (default: 'pytest -q')",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory, relative to project root",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Execution timeout in seconds (1-1800)",
                    "minimum": 1,
                    "maximum": 1800,
                },
            },
        },
        handler=_handler,
    )


def _build_proof_checker_tool(working_dir: Path) -> Tool:
    async def _handler(invocation: dict[str, Any]) -> dict[str, Any]:
        args = invocation.get("arguments") or {}
        claim = str(args.get("claim", "")).strip()
        if not claim:
            return _failure_result("Parameter 'claim' is required for proof_checker.", error="missing claim")

        if _proof_checker_hook is None:
            return _failure_result(
                "No proof checker is configured. Register one via copex.sdk_tools.register_proof_checker().",
                error="proof checker not configured",
                session_log="proof_checker unavailable",
            )

        try:
            result = _proof_checker_hook(args, working_dir)
        except Exception as exc:
            return _failure_result(
                "Proof checker invocation failed.",
                error=str(exc),
                session_log="proof_checker hook exception",
            )

        if isinstance(result, dict):
            return result
        return _success_result(
            result,
            session_log="proof_checker hook result",
        )

    return Tool(
        name="proof_checker",
        description="Run a configured formal proof checker on a claim or theorem.",
        parameters={
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "Claim or theorem text to validate"},
                "context": {
                    "type": "string",
                    "description": "Optional context (definitions, assumptions, file path)",
                },
            },
            "required": ["claim"],
        },
        handler=_handler,
    )


def _register_builtin_tools() -> None:
    register_domain_tool("memory_search", _build_memory_search_tool, replace=True)
    register_domain_tool("test_runner", _build_test_runner_tool, replace=True)
    register_domain_tool("proof_checker", _build_proof_checker_tool, replace=True)


_register_builtin_tools()
