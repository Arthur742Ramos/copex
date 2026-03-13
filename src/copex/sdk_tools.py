"""SDK-native domain tool registration for Copex."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from copilot import Tool

from copex.memory import ProjectMemory
from copex.pdf_analyze import (
    PdfAnalyzeError,
    export_pdf_screenshots,
    pdf_support_available,
    prepare_pdf_analysis_payload,
)
from copex.security import SecurityError, sanitize_command, validate_path

DomainToolFactory = Callable[[Path], Tool]
ProofCheckerHook = Callable[[dict[str, Any], Path], dict[str, Any] | str]
TestRunnerHook = Callable[[dict[str, Any], Path], dict[str, Any] | str]

_MAX_TEXT_RESULT_CHARS = 8_000
_MAX_MEMORY_ENTRY_CHARS = 400
_DEFAULT_TEST_TIMEOUT_SECONDS = 300.0
_ALLOWED_PYTHON_TEST_MODULES = frozenset({"pytest", "unittest"})
_ALLOWED_DIRECT_TEST_RUNNERS = frozenset({"pytest", "tox", "nox"})
_ALLOWED_SUBCOMMAND_TEST_RUNNERS = frozenset(
    {"bun", "cargo", "dotnet", "go", "make", "npm", "pnpm", "uv", "yarn"}
)

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
    binary_results: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "resultType": "success",
        "textResultForLlm": _clip_text(text),
    }
    if binary_results:
        result["binaryResultsForLlm"] = list(binary_results)
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


def _is_test_name(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized == "test" or normalized.startswith("test:")


def _validate_test_command(argv: list[str]) -> list[str]:
    sanitized = sanitize_command(argv)
    program = Path(sanitized[0]).name.lower()
    tail = sanitized[1:]

    if program in _ALLOWED_DIRECT_TEST_RUNNERS:
        return sanitized

    if program in {"python", "python3"}:
        if len(tail) >= 2 and tail[0] == "-m" and tail[1].lower() in _ALLOWED_PYTHON_TEST_MODULES:
            return sanitized
        raise SecurityError(
            "Only python -m pytest/unittest test commands are allowed.",
            violation_type="invalid_test_command",
            context={"command": " ".join(sanitized[:4])},
        )

    if program == "uv":
        if tail and tail[0] == "run":
            _validate_test_command(tail[1:])
            return sanitized
        raise SecurityError(
            "Only uv run <test command> invocations are allowed.",
            violation_type="invalid_test_command",
            context={"command": " ".join(sanitized[:4])},
        )

    if program not in _ALLOWED_SUBCOMMAND_TEST_RUNNERS:
        raise SecurityError(
            "Only supported project test commands are allowed.",
            violation_type="invalid_test_command",
            context={"command": " ".join(sanitized[:4])},
        )

    if not tail:
        raise SecurityError(
            "Test command is missing a test subcommand.",
            violation_type="invalid_test_command",
            context={"command": program},
        )

    head = tail[0].lower()
    if program in {"npm", "pnpm"}:
        if _is_test_name(head):
            return sanitized
        if len(tail) >= 2 and head == "run" and _is_test_name(tail[1]):
            return sanitized
        raise SecurityError(
            "Only npm/pnpm test scripts are allowed.",
            violation_type="invalid_test_command",
            context={"command": " ".join(sanitized[:4])},
        )

    if program == "yarn":
        if _is_test_name(head):
            return sanitized
        if len(tail) >= 2 and head == "run" and _is_test_name(tail[1]):
            return sanitized
        raise SecurityError(
            "Only yarn test scripts are allowed.",
            violation_type="invalid_test_command",
            context={"command": " ".join(sanitized[:4])},
        )

    if program == "make":
        if _is_test_name(head):
            return sanitized
        raise SecurityError(
            "Only make test targets are allowed.",
            violation_type="invalid_test_command",
            context={"command": " ".join(sanitized[:4])},
        )

    if head != "test":
        raise SecurityError(
            "Only test subcommands are allowed for this runner.",
            violation_type="invalid_test_command",
            context={"command": " ".join(sanitized[:4])},
        )
    return sanitized


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
                candidate = working_dir / candidate
            try:
                command_cwd = validate_path(
                    candidate,
                    base_dir=working_dir.resolve(),
                    allow_absolute=True,
                )
            except SecurityError as exc:
                return _failure_result(
                    "test_runner cwd must stay within the configured working directory.",
                    error=str(exc),
                )

        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return _failure_result("Invalid test command syntax.", error=str(exc))
        if not argv:
            return _failure_result("Parameter 'command' cannot be empty.", error="empty argv")

        try:
            argv = _validate_test_command(argv)
        except SecurityError as exc:
            return _failure_result(
                "Only supported project test commands are allowed.",
                error=str(exc),
                session_log=f"test_runner rejected command={command!r}",
            )

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
                    "description": (
                        "Supported test command to execute "
                        "(examples: 'pytest -q', 'python -m pytest', 'uv run pytest -q')"
                    ),
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


_js_repl_manager: Any | None = None
_js_repl_logger = logging.getLogger(__name__)
_pdf_tools_logger = logging.getLogger(__name__)


def _get_js_repl_manager(node_path: str | None = None) -> Any:
    """Return the module-level JSReplManager singleton, creating it lazily."""
    global _js_repl_manager
    current_node_path = getattr(_js_repl_manager, "node_path", None) if _js_repl_manager else None
    if _js_repl_manager is None or (node_path and current_node_path != node_path):
        from copex.js_repl import JSReplManager

        if _js_repl_manager is not None and getattr(_js_repl_manager, "running", False):
            raise RuntimeError("JS REPL manager is already running with a different Node.js executable")
        _js_repl_manager = JSReplManager(node_path=node_path)
    return _js_repl_manager


def _build_js_repl_tool(working_dir: Path, *, node_path: str | None = None) -> Tool:
    async def _handler(invocation: dict[str, Any]) -> dict[str, Any]:
        args = invocation.get("arguments") or {}
        code = str(args.get("code", "")).strip()
        if not code:
            return _failure_result(
                "Parameter 'code' is required for js_repl.",
                error="missing code",
            )

        manager = _get_js_repl_manager(node_path)
        try:
            result = await manager.execute(code)
        except Exception as exc:
            return _failure_result(
                "JavaScript execution failed.",
                error=str(exc),
                session_log="js_repl execution error",
            )

        console_lines = result.get("console", [])
        output_parts = []
        if console_lines:
            output_parts.append("Console:\n" + "\n".join(console_lines))
        if result.get("error"):
            output_parts.append(f"Error: {result['error']}")
            text = "\n\n".join(output_parts) if output_parts else result["error"]
            return _failure_result(
                text,
                error=result["error"],
                session_log="js_repl error",
            )
        if result.get("result") is not None:
            output_parts.append(f"Result: {result['result']}")
        text = "\n\n".join(output_parts) if output_parts else "(no output)"
        return _success_result(
            text,
            session_log=f"js_repl ok len={len(code)}",
            telemetry={"code_length": len(code)},
        )

    return Tool(
        name="js_repl",
        description=(
            "Execute JavaScript code in a persistent Node.js REPL. "
            "Variables and functions persist across calls. "
            "Use for calculations, data processing, prototyping, or testing JS logic."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "JavaScript code to execute (top-level await supported)",
                },
            },
            "required": ["code"],
        },
        handler=_handler,
    )


def _build_js_repl_reset_tool(working_dir: Path, *, node_path: str | None = None) -> Tool:
    async def _handler(invocation: dict[str, Any]) -> dict[str, Any]:
        manager = _get_js_repl_manager(node_path)
        try:
            await manager.reset()
        except Exception as exc:
            return _failure_result(
                "Failed to reset JavaScript REPL.",
                error=str(exc),
                session_log="js_repl_reset error",
            )
        return _success_result(
            "JavaScript REPL context has been reset. All variables and state cleared.",
            session_log="js_repl_reset ok",
        )

    return Tool(
        name="js_repl_reset",
        description="Reset the persistent JavaScript REPL, clearing all variables and state.",
        parameters={"type": "object", "properties": {}},
        handler=_handler,
    )


def _build_pdf_analyze_tool(working_dir: Path) -> Tool:
    async def _handler(invocation: dict[str, Any]) -> dict[str, Any]:
        args = invocation.get("arguments") or {}
        path = str(args.get("path", "")).strip()
        prompt = str(args.get("prompt", "")).strip()
        pages = args.get("pages")
        page_spec = str(pages).strip() if pages is not None else None
        if page_spec == "":
            page_spec = None

        if not path:
            return _failure_result(
                "Parameter 'path' is required for pdf_analyze.",
                error="missing path",
            )
        if not prompt:
            return _failure_result(
                "Parameter 'prompt' is required for pdf_analyze.",
                error="missing prompt",
            )

        try:
            payload = prepare_pdf_analysis_payload(
                path,
                prompt=prompt,
                pages=page_spec,
                working_dir=working_dir,
            )
        except PdfAnalyzeError as exc:
            return _failure_result(
                f"Unable to prepare PDF analysis: {exc}",
                error=str(exc),
                session_log="pdf_analyze error",
            )

        return _success_result(
            payload.prompt_text,
            session_log=(
                f"pdf_analyze ok path={payload.display_path} pages={len(payload.selected_pages)}"
            ),
            telemetry={
                "path": payload.display_path,
                "page_count": len(payload.selected_pages),
                "total_image_bytes": payload.total_image_bytes,
            },
            binary_results=payload.binary_results,
        )

    return Tool(
        name="pdf_analyze",
        description=(
            "Render PDF pages to images for the current model to inspect text, graphics, charts, "
            "tables, and figures."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the PDF file to analyze",
                },
                "prompt": {
                    "type": "string",
                    "description": "Analysis request for the rendered PDF pages",
                },
                "pages": {
                    "type": "string",
                    "description": "Optional page selection like '1-3,5'; defaults to all pages",
                },
            },
            "required": ["path", "prompt"],
        },
        handler=_handler,
    )


def _build_pdf_screenshot_tool(working_dir: Path) -> Tool:
    async def _handler(invocation: dict[str, Any]) -> dict[str, Any]:
        args = invocation.get("arguments") or {}
        path = str(args.get("path", "")).strip()
        pages = str(args.get("pages", "")).strip()
        output_dir = str(args.get("output_dir", "")).strip()

        if not path:
            return _failure_result(
                "Parameter 'path' is required for pdf_screenshot.",
                error="missing path",
            )
        if not pages:
            return _failure_result(
                "Parameter 'pages' is required for pdf_screenshot.",
                error="missing pages",
            )
        if not output_dir:
            return _failure_result(
                "Parameter 'output_dir' is required for pdf_screenshot.",
                error="missing output_dir",
            )

        try:
            written_paths = export_pdf_screenshots(
                path,
                pages=pages,
                output_dir=output_dir,
                working_dir=working_dir,
            )
        except PdfAnalyzeError as exc:
            return _failure_result(
                f"Unable to render PDF screenshots: {exc}",
                error=str(exc),
                session_log="pdf_screenshot error",
            )

        written_display = [str(output_path.relative_to(working_dir.resolve())) for output_path in written_paths]
        return _success_result(
            "Rendered PDF screenshots:\n" + "\n".join(f"- {path}" for path in written_display),
            session_log=f"pdf_screenshot ok path={path} pages={len(written_paths)}",
            telemetry={"path": path, "page_count": len(written_paths), "output_dir": output_dir},
        )

    return Tool(
        name="pdf_screenshot",
        description="Render selected PDF pages to PNG files inside the working directory.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the PDF file to render",
                },
                "pages": {
                    "type": "string",
                    "description": "Page selection like '1-3,5'",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory where rendered PNG files should be written",
                },
            },
            "required": ["path", "pages", "output_dir"],
        },
        handler=_handler,
    )


def register_pdf_tools() -> bool:
    """Conditionally register PDF tools if PyMuPDF is available."""

    if not pdf_support_available():
        _pdf_tools_logger.warning("PyMuPDF not found; pdf tools will not be available")
        return False

    register_domain_tool("pdf_analyze", _build_pdf_analyze_tool, replace=True)
    register_domain_tool("pdf_screenshot", _build_pdf_screenshot_tool, replace=True)
    return True


def register_js_repl_tools(node_path: str | None = None) -> bool:
    """Conditionally register JS REPL tools if Node.js is available.

    Returns True if registration succeeded, False otherwise.
    """
    from copex.js_repl import resolve_node_path

    resolved_node_path = resolve_node_path(node_path)
    if not resolved_node_path:
        if node_path:
            _js_repl_logger.warning(
                "Configured Node.js executable not found for js_repl: %s",
                node_path,
            )
        else:
            _js_repl_logger.warning("Node.js not found; js_repl tools will not be available")
        return False

    register_domain_tool(
        "js_repl",
        lambda working_dir: _build_js_repl_tool(working_dir, node_path=resolved_node_path),
        replace=True,
    )
    register_domain_tool(
        "js_repl_reset",
        lambda working_dir: _build_js_repl_reset_tool(working_dir, node_path=resolved_node_path),
        replace=True,
    )
    return True


async def shutdown_js_repl() -> None:
    """Stop the JS REPL kernel if running. Safe to call at exit."""
    global _js_repl_manager
    if _js_repl_manager is not None:
        try:
            await _js_repl_manager.stop()
        except Exception:
            _js_repl_logger.debug("Error stopping JS REPL manager", exc_info=True)
        _js_repl_manager = None


def _register_builtin_tools() -> None:
    register_domain_tool("memory_search", _build_memory_search_tool, replace=True)
    register_domain_tool("test_runner", _build_test_runner_tool, replace=True)
    register_domain_tool("proof_checker", _build_proof_checker_tool, replace=True)


_register_builtin_tools()
