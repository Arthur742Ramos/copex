"""Tests for the persistent JavaScript REPL tool."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from copex.js_repl import JSReplError, JSReplManager
from copex.sdk_tools import (
    _build_js_repl_reset_tool,
    _build_js_repl_tool,
    build_domain_tools,
    list_domain_tools,
    register_js_repl_tools,
    shutdown_js_repl,
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def _invocation(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": "session-1",
        "tool_call_id": "call-1",
        "tool_name": "tool",
        "arguments": args,
    }


# --------------------------------------------------------------------------
# JSReplManager unit tests (with FakeProcess)
# --------------------------------------------------------------------------


class FakeProcess:
    """Simulates an asyncio.subprocess.Process for the JS kernel."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.returncode: int | None = None
        self.pid: int = 12345
        self._responses = list(responses or [])
        self._response_idx = 0
        self.stdin = MagicMock()
        self.stdin.write = MagicMock()
        self.stdin.drain = AsyncMock()
        self.stdout = self._make_stdout()
        self.stderr = MagicMock()

    def _make_stdout(self) -> Any:
        stdout = MagicMock()

        async def _readline() -> bytes:
            if self._response_idx < len(self._responses):
                resp = self._responses[self._response_idx]
                self._response_idx += 1
                return (json.dumps(resp) + "\n").encode()
            # Simulate stream end after all responses
            await asyncio.sleep(60)  # block until cancelled
            return b""

        stdout.readline = _readline
        return stdout

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


@pytest.fixture()
def fake_responses() -> list[dict[str, Any]]:
    return [
        {"id": 1, "result": "42", "error": None, "console": []},
    ]


@pytest.fixture()
def fake_process(fake_responses: list[dict[str, Any]]) -> FakeProcess:
    return FakeProcess(fake_responses)


@pytest.fixture()
def manager() -> JSReplManager:
    return JSReplManager(node_path="/usr/bin/node")


@pytest.mark.asyncio
async def test_manager_start_creates_subprocess(
    manager: JSReplManager, fake_process: FakeProcess
) -> None:
    with patch("copex.js_repl.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = fake_process
        await manager.start()
        assert manager.running
        mock_exec.assert_awaited_once()
        await manager.stop()


@pytest.mark.asyncio
async def test_manager_execute_sends_and_receives(
    manager: JSReplManager, fake_process: FakeProcess
) -> None:
    with patch("copex.js_repl.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = fake_process
        await manager.start()

        result = await manager.execute("21 * 2")
        assert result["result"] == "42"
        assert result["error"] is None

        # Verify the request was written to stdin
        call_args = fake_process.stdin.write.call_args[0][0]
        msg = json.loads(call_args.decode())
        assert msg["code"] == "21 * 2"
        assert "id" in msg

        await manager.stop()


@pytest.mark.asyncio
async def test_manager_reset(manager: JSReplManager) -> None:
    responses = [{"id": 1, "result": "context reset", "error": None, "console": []}]
    proc = FakeProcess(responses)
    with patch("copex.js_repl.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = proc
        await manager.start()

        result = await manager.reset()
        assert result["result"] == "context reset"

        call_args = proc.stdin.write.call_args[0][0]
        msg = json.loads(call_args.decode())
        assert msg["action"] == "reset"

        await manager.stop()


@pytest.mark.asyncio
async def test_manager_auto_restarts_on_crash(manager: JSReplManager) -> None:
    # proc1 has already exited (returncode != None), so _ensure_running re-spawns
    responses2 = [{"id": 1, "result": "second", "error": None, "console": []}]
    proc1 = FakeProcess([])
    proc2 = FakeProcess(responses2)
    proc1.returncode = 1  # simulate crash

    call_count = 0

    async def _mock_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        nonlocal call_count
        call_count += 1
        return proc1 if call_count == 1 else proc2

    with patch("copex.js_repl.asyncio.create_subprocess_exec", side_effect=_mock_exec):
        await manager.start()
        # Process has crashed (returncode set), _ensure_running will restart
        # After restart, request_id resets to 0 internally, so next id=1
        manager._request_id = 0
        result = await manager.execute("1 + 1")
        assert result["result"] == "second"
        assert call_count == 2
        await manager.stop()


@pytest.mark.asyncio
async def test_manager_stop_idempotent(manager: JSReplManager) -> None:
    await manager.stop()  # should not raise
    assert not manager.running


@pytest.mark.asyncio
async def test_manager_context_manager() -> None:
    responses = [{"id": 1, "result": "ok", "error": None, "console": []}]
    proc = FakeProcess(responses)
    with patch("copex.js_repl.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = proc
        async with JSReplManager(node_path="/usr/bin/node") as mgr:
            assert mgr.running
            result = await mgr.execute("'ok'")
            assert result["result"] == "ok"
        # After exiting context, process should be stopped
        assert not mgr.running


# --------------------------------------------------------------------------
# Domain tool handler tests
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_js_repl_manager() -> None:
    """Reset the module-level singleton before each test."""
    import copex.sdk_tools as _sdk

    _sdk._js_repl_manager = None


def test_js_repl_tool_handler_missing_code(tmp_path: Path) -> None:
    tool = _build_js_repl_tool(tmp_path)
    result = run(tool.handler(_invocation({})))
    assert result["resultType"] == "failure"
    assert "code" in result["textResultForLlm"].lower()


def test_js_repl_tool_handler_success(tmp_path: Path) -> None:
    mock_manager = MagicMock()
    mock_manager.execute = AsyncMock(
        return_value={"result": "42", "error": None, "console": []}
    )

    with patch("copex.sdk_tools._get_js_repl_manager", return_value=mock_manager):
        tool = _build_js_repl_tool(tmp_path)
        result = run(tool.handler(_invocation({"code": "21 * 2"})))

    assert result["resultType"] == "success"
    assert "42" in result["textResultForLlm"]


def test_js_repl_tool_handler_with_console(tmp_path: Path) -> None:
    mock_manager = MagicMock()
    mock_manager.execute = AsyncMock(
        return_value={"result": None, "error": None, "console": ["hello", "world"]}
    )

    with patch("copex.sdk_tools._get_js_repl_manager", return_value=mock_manager):
        tool = _build_js_repl_tool(tmp_path)
        result = run(tool.handler(_invocation({"code": "console.log('hello'); console.log('world')"})))

    assert result["resultType"] == "success"
    assert "hello" in result["textResultForLlm"]
    assert "world" in result["textResultForLlm"]


def test_js_repl_tool_handler_error(tmp_path: Path) -> None:
    mock_manager = MagicMock()
    mock_manager.execute = AsyncMock(
        return_value={"result": None, "error": "ReferenceError: x is not defined", "console": []}
    )

    with patch("copex.sdk_tools._get_js_repl_manager", return_value=mock_manager):
        tool = _build_js_repl_tool(tmp_path)
        result = run(tool.handler(_invocation({"code": "x"})))

    assert result["resultType"] == "failure"
    assert "ReferenceError" in result["textResultForLlm"]


def test_js_repl_tool_handler_exception(tmp_path: Path) -> None:
    mock_manager = MagicMock()
    mock_manager.execute = AsyncMock(side_effect=JSReplError("timeout"))

    with patch("copex.sdk_tools._get_js_repl_manager", return_value=mock_manager):
        tool = _build_js_repl_tool(tmp_path)
        result = run(tool.handler(_invocation({"code": "while(true){}"})))

    assert result["resultType"] == "failure"
    assert "timeout" in result["error"].lower()


def test_js_repl_reset_tool_handler(tmp_path: Path) -> None:
    mock_manager = MagicMock()
    mock_manager.reset = AsyncMock()

    with patch("copex.sdk_tools._get_js_repl_manager", return_value=mock_manager):
        tool = _build_js_repl_reset_tool(tmp_path)
        result = run(tool.handler(_invocation({})))

    assert result["resultType"] == "success"
    assert "reset" in result["textResultForLlm"].lower()


def test_js_repl_reset_tool_handler_failure(tmp_path: Path) -> None:
    mock_manager = MagicMock()
    mock_manager.reset = AsyncMock(side_effect=JSReplError("kernel dead"))

    with patch("copex.sdk_tools._get_js_repl_manager", return_value=mock_manager):
        tool = _build_js_repl_reset_tool(tmp_path)
        result = run(tool.handler(_invocation({})))

    assert result["resultType"] == "failure"
    assert "kernel dead" in result["error"].lower()


# --------------------------------------------------------------------------
# Registration & conditional loading
# --------------------------------------------------------------------------


def test_register_js_repl_tools_with_node() -> None:
    with patch("copex.sdk_tools.shutil.which", return_value="/usr/bin/node"):
        assert register_js_repl_tools() is True
    assert "js_repl" in list_domain_tools()
    assert "js_repl_reset" in list_domain_tools()


def test_register_js_repl_tools_without_node() -> None:
    with patch("copex.sdk_tools.shutil.which", return_value=None):
        assert register_js_repl_tools() is False
    # Tools should not be registered when node is missing
    # (unless a previous test registered them)


def test_register_js_repl_tools_builds_tools(tmp_path: Path) -> None:
    with patch("copex.sdk_tools.shutil.which", return_value="/usr/bin/node"):
        register_js_repl_tools()
    tools = build_domain_tools(["js_repl", "js_repl_reset"], working_dir=tmp_path)
    assert len(tools) == 2
    assert tools[0].name == "js_repl"
    assert tools[1].name == "js_repl_reset"


# --------------------------------------------------------------------------
# Config integration
# --------------------------------------------------------------------------


def test_config_js_repl_default_false() -> None:
    from copex.config import CopexConfig

    config = CopexConfig()
    assert config.js_repl is False


def test_config_js_repl_enabled() -> None:
    from copex.config import CopexConfig

    config = CopexConfig(js_repl=True)
    assert config.js_repl is True


def test_config_to_session_options_includes_tools_when_enabled() -> None:
    from copex.config import CopexConfig

    config = CopexConfig(js_repl=True)
    with patch("copex.sdk_tools.shutil.which", return_value="/usr/bin/node"):
        opts = config.to_session_options()
    assert "tools" in opts
    tool_names = [t.name for t in opts["tools"]]
    assert "js_repl" in tool_names
    assert "js_repl_reset" in tool_names


def test_config_to_session_options_no_tools_when_disabled() -> None:
    from copex.config import CopexConfig

    config = CopexConfig(js_repl=False)
    opts = config.to_session_options()
    assert "tools" not in opts


def test_config_to_session_options_no_tools_when_no_node() -> None:
    from copex.config import CopexConfig

    config = CopexConfig(js_repl=True)
    with patch("copex.sdk_tools.shutil.which", return_value=None):
        opts = config.to_session_options()
    assert "tools" not in opts


# --------------------------------------------------------------------------
# Shutdown helper
# --------------------------------------------------------------------------


def test_shutdown_js_repl_when_none() -> None:
    import copex.sdk_tools as _sdk

    _sdk._js_repl_manager = None
    run(shutdown_js_repl())
    assert _sdk._js_repl_manager is None


def test_shutdown_js_repl_stops_manager() -> None:
    import copex.sdk_tools as _sdk

    mock_manager = MagicMock()
    mock_manager.stop = AsyncMock()
    _sdk._js_repl_manager = mock_manager

    run(shutdown_js_repl())
    mock_manager.stop.assert_awaited_once()
    assert _sdk._js_repl_manager is None
