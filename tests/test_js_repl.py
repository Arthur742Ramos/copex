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

    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        *,
        stderr_output: bytes = b"",
    ) -> None:
        self.returncode: int | None = None
        self.pid: int = 12345
        self._responses = list(responses or [])
        self._response_idx = 0
        self._write_count = 0
        self.stdin = MagicMock()
        self.stdin.write = MagicMock(side_effect=self._write)
        self.stdin.drain = AsyncMock()
        self.stdout = self._make_stdout()
        self.stderr = MagicMock()
        self.stderr.read = AsyncMock(return_value=stderr_output)

    def _write(self, _data: bytes) -> None:
        self._write_count += 1

    def _make_stdout(self) -> Any:
        stdout = MagicMock()

        async def _readline() -> bytes:
            if self._response_idx < len(self._responses):
                while self._write_count <= self._response_idx:
                    await asyncio.sleep(0)
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
async def test_manager_start_drains_stderr(
    manager: JSReplManager, fake_process: FakeProcess
) -> None:
    with patch("copex.js_repl.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = fake_process
        await manager.start()
        await asyncio.sleep(0)
        fake_process.stderr.read.assert_awaited_once()
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
    # proc1 crashes after start(), so _ensure_running re-spawns before execute()
    responses2 = [{"id": 1, "result": "second", "error": None, "console": []}]
    proc1 = FakeProcess([])
    proc2 = FakeProcess(responses2)

    call_count = 0

    async def _mock_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        nonlocal call_count
        call_count += 1
        return proc1 if call_count == 1 else proc2

    with patch("copex.js_repl.asyncio.create_subprocess_exec", side_effect=_mock_exec):
        await manager.start()
        proc1.returncode = 1  # simulate crash after start
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
    with patch("copex.js_repl.shutil.which", return_value="/usr/bin/node"):
        assert register_js_repl_tools() is True
    assert "js_repl" in list_domain_tools()
    assert "js_repl_reset" in list_domain_tools()


def test_register_js_repl_tools_without_node() -> None:
    with patch("copex.js_repl.shutil.which", return_value=None):
        assert register_js_repl_tools() is False
    # Tools should not be registered when node is missing
    # (unless a previous test registered them)


def test_register_js_repl_tools_builds_tools(tmp_path: Path) -> None:
    with patch("copex.js_repl.shutil.which", return_value="/usr/bin/node"):
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


def test_config_js_repl_node_path_normalizes_blank() -> None:
    from copex.config import CopexConfig

    config = CopexConfig(js_repl_node_path="   ")
    assert config.js_repl_node_path is None


def test_config_to_session_options_includes_tools_when_enabled() -> None:
    from copex.config import CopexConfig

    config = CopexConfig(js_repl=True)
    with patch("copex.js_repl.shutil.which", return_value="/usr/bin/node"):
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
    with patch("copex.js_repl.shutil.which", return_value=None):
        opts = config.to_session_options()
    assert "tools" not in opts


def test_config_to_session_options_uses_custom_node_path() -> None:
    from copex.config import CopexConfig

    config = CopexConfig(js_repl=True, js_repl_node_path="/custom/node")
    fake_tool = MagicMock(name="js_repl_tool")
    with (
        patch("copex.sdk_tools.register_js_repl_tools", return_value=True) as mock_register,
        patch("copex.sdk_tools.build_domain_tools", return_value=[fake_tool]) as mock_build,
    ):
        opts = config.to_session_options()

    mock_register.assert_called_once_with("/custom/node")
    mock_build.assert_called_once()
    assert opts["tools"] == [fake_tool]


def test_config_to_session_options_rejects_missing_custom_node_path() -> None:
    from copex.config import CopexConfig

    config = CopexConfig(js_repl=True, js_repl_node_path="/missing/node")
    with patch("copex.sdk_tools.register_js_repl_tools", return_value=False):
        with pytest.raises(ValueError, match="js_repl_node_path"):
            config.to_session_options()


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


def test_shutdown_js_repl_suppresses_exception() -> None:
    import copex.sdk_tools as _sdk

    mock_manager = MagicMock()
    mock_manager.stop = AsyncMock(side_effect=RuntimeError("boom"))
    _sdk._js_repl_manager = mock_manager

    run(shutdown_js_repl())  # should not raise
    assert _sdk._js_repl_manager is None


# --------------------------------------------------------------------------
# _get_js_repl_manager lazy singleton
# --------------------------------------------------------------------------


def test_get_js_repl_manager_creates_lazily() -> None:
    import copex.sdk_tools as _sdk

    _sdk._js_repl_manager = None
    manager = _sdk._get_js_repl_manager()
    assert manager is not None
    from copex.js_repl import JSReplManager

    assert isinstance(manager, JSReplManager)
    # Calling again returns same instance
    assert _sdk._get_js_repl_manager() is manager
    _sdk._js_repl_manager = None  # cleanup


# --------------------------------------------------------------------------
# JSReplManager - idempotent start
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manager_start_idempotent(
    manager: JSReplManager, fake_process: FakeProcess
) -> None:
    with patch("copex.js_repl.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = fake_process
        await manager.start()
        await manager.start()  # second call is a no-op
        assert mock_exec.await_count == 1
        await manager.stop()


# --------------------------------------------------------------------------
# JSReplManager - timeout paths
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manager_execute_timeout() -> None:
    """execute() raises JSReplError when the kernel never responds."""
    proc = FakeProcess([])  # no responses → future never resolves
    mgr = JSReplManager(node_path="/usr/bin/node")
    with patch("copex.js_repl.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = proc
        await mgr.start()
        with pytest.raises(JSReplError, match="timed out"):
            await mgr.execute("slow()", timeout=0.05)
        await mgr.stop()


@pytest.mark.asyncio
async def test_manager_reset_timeout() -> None:
    """reset() raises JSReplError when the kernel never responds."""
    proc = FakeProcess([])
    mgr = JSReplManager(node_path="/usr/bin/node")
    with patch("copex.js_repl.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = proc
        await mgr.start()
        with pytest.raises(JSReplError, match="reset timed out"):
            await mgr.reset(timeout=0.05)
        await mgr.stop()


@pytest.mark.asyncio
async def test_manager_start_surfaces_stderr_when_kernel_exits_immediately() -> None:
    proc = FakeProcess(stderr_output=b"SyntaxError: bad import\n")
    proc.returncode = 1
    mgr = JSReplManager(node_path="/usr/bin/node")

    with patch("copex.js_repl.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = proc
        with pytest.raises(JSReplError, match="SyntaxError: bad import"):
            await mgr.start()


# --------------------------------------------------------------------------
# JSReplManager - stop with stubborn process (kill path)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manager_stop_kills_on_timeout() -> None:
    """When terminate + wait times out, stop() calls kill()."""
    proc = FakeProcess([{"id": 1, "result": "ok", "error": None, "console": []}])
    mgr = JSReplManager(node_path="/usr/bin/node")

    # Make wait() hang so it times out
    async def _slow_wait() -> int:
        await asyncio.sleep(60)
        return 0

    proc.wait = _slow_wait

    with patch("copex.js_repl.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = proc
        await mgr.start()
        await mgr.stop()
        # kill() should have been called since wait() timed out
        assert proc.returncode == -9


@pytest.mark.asyncio
async def test_manager_stop_handles_process_lookup_error() -> None:
    """ProcessLookupError during terminate/kill is handled gracefully."""
    proc = FakeProcess([{"id": 1, "result": "ok", "error": None, "console": []}])
    mgr = JSReplManager(node_path="/usr/bin/node")

    # Make terminate raise ProcessLookupError (process already gone)
    def _raise_plookup() -> None:
        raise ProcessLookupError()

    proc.terminate = _raise_plookup

    with patch("copex.js_repl.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = proc
        await mgr.start()
        await mgr.stop()  # should not raise
        assert not mgr.running


# --------------------------------------------------------------------------
# JSReplManager - stop fails pending futures
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manager_stop_fails_pending_futures() -> None:
    """Pending futures get ConnectionError when stop() is called."""
    proc = FakeProcess([])  # no responses
    mgr = JSReplManager(node_path="/usr/bin/node")

    with patch("copex.js_repl.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = proc
        await mgr.start()

        # Manually inject a pending future
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        mgr._pending[999] = fut

        await mgr.stop()
        assert fut.done()
        with pytest.raises(ConnectionError, match="stopped"):
            fut.result()


# --------------------------------------------------------------------------
# JSReplManager - reader loop edge cases
# --------------------------------------------------------------------------


class NonJsonFakeProcess(FakeProcess):
    """FakeProcess that sends non-JSON output before real responses."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        super().__init__(responses)

    def _make_stdout(self) -> Any:
        stdout = MagicMock()
        lines = [b"not json at all\n"]  # non-JSON output first
        for resp in self._responses:
            lines.append((json.dumps(resp) + "\n").encode())

        idx = 0

        async def _readline() -> bytes:
            nonlocal idx
            while self._write_count == 0:
                await asyncio.sleep(0)
            if idx < len(lines):
                data = lines[idx]
                idx += 1
                return data
            await asyncio.sleep(60)
            return b""

        stdout.readline = _readline
        return stdout


@pytest.mark.asyncio
async def test_reader_loop_skips_non_json() -> None:
    """Non-JSON lines from the kernel are skipped without error."""
    responses = [{"id": 1, "result": "ok", "error": None, "console": []}]
    proc = NonJsonFakeProcess(responses)
    mgr = JSReplManager(node_path="/usr/bin/node")

    with patch("copex.js_repl.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = proc
        await mgr.start()

        result = await mgr.execute("1")
        assert result["result"] == "ok"
        await mgr.stop()


class EofFakeProcess(FakeProcess):
    """FakeProcess whose stdout returns empty bytes immediately (kernel exit)."""

    def _make_stdout(self) -> Any:
        stdout = MagicMock()

        async def _readline() -> bytes:
            return b""  # immediate EOF

        stdout.readline = _readline
        return stdout


@pytest.mark.asyncio
async def test_reader_loop_handles_eof_and_fails_pending() -> None:
    """When kernel exits (EOF), pending futures get ConnectionError."""
    proc = EofFakeProcess([])
    mgr = JSReplManager(node_path="/usr/bin/node")

    with patch("copex.js_repl.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = proc
        await mgr.start()

        # Give reader loop a moment to process the EOF
        await asyncio.sleep(0.05)

        # Manually inject a pending future
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        mgr._pending[999] = fut

        # Reader loop already exited — stop will clean up pending
        await mgr.stop()

# --------------------------------------------------------------------------
# Tool handler edge cases
# --------------------------------------------------------------------------


def test_js_repl_tool_handler_no_result_no_console(tmp_path: Path) -> None:
    """Handler returns '(no output)' when kernel returns nothing."""
    mock_manager = MagicMock()
    mock_manager.execute = AsyncMock(
        return_value={"result": None, "error": None, "console": []}
    )

    with patch("copex.sdk_tools._get_js_repl_manager", return_value=mock_manager):
        tool = _build_js_repl_tool(tmp_path)
        result = run(tool.handler(_invocation({"code": "void 0"})))

    assert result["resultType"] == "success"
    assert "(no output)" in result["textResultForLlm"]


def test_js_repl_tool_handler_console_and_result(tmp_path: Path) -> None:
    """Handler shows both console output and result when both present."""
    mock_manager = MagicMock()
    mock_manager.execute = AsyncMock(
        return_value={"result": "3", "error": None, "console": ["log line"]}
    )

    with patch("copex.sdk_tools._get_js_repl_manager", return_value=mock_manager):
        tool = _build_js_repl_tool(tmp_path)
        result = run(tool.handler(_invocation({"code": "console.log('log line'); 1+2"})))

    assert result["resultType"] == "success"
    assert "log line" in result["textResultForLlm"]
    assert "3" in result["textResultForLlm"]


def test_js_repl_tool_handler_uses_custom_node_path(tmp_path: Path) -> None:
    mock_manager = MagicMock()
    mock_manager.execute = AsyncMock(
        return_value={"result": "42", "error": None, "console": []}
    )

    with patch("copex.sdk_tools._get_js_repl_manager", return_value=mock_manager) as mock_get:
        tool = _build_js_repl_tool(tmp_path, node_path="/custom/node")
        run(tool.handler(_invocation({"code": "21 * 2"})))

    mock_get.assert_called_once_with("/custom/node")


def test_js_repl_tool_handler_error_with_console(tmp_path: Path) -> None:
    """Handler includes console output even when there's an error."""
    mock_manager = MagicMock()
    mock_manager.execute = AsyncMock(
        return_value={"result": None, "error": "SyntaxError: bad", "console": ["before error"]}
    )

    with patch("copex.sdk_tools._get_js_repl_manager", return_value=mock_manager):
        tool = _build_js_repl_tool(tmp_path)
        result = run(tool.handler(_invocation({"code": "console.log('before error'); {{{"})))

    assert result["resultType"] == "failure"
    assert "before error" in result["textResultForLlm"]
    assert "SyntaxError" in result["textResultForLlm"]


# --------------------------------------------------------------------------
# Integration tests (require Node.js)
# --------------------------------------------------------------------------


_node_available = shutil.which("node") is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not _node_available, reason="Node.js not available")
async def test_integration_basic_execution() -> None:
    """Kernel evaluates simple expressions correctly."""
    async with JSReplManager() as mgr:
        result = await mgr.execute("2 + 3")
        assert result["result"] == "5"
        assert result["error"] is None


@pytest.mark.asyncio
@pytest.mark.skipif(not _node_available, reason="Node.js not available")
async def test_integration_state_persistence() -> None:
    """Variables persist across execute() calls."""
    async with JSReplManager() as mgr:
        await mgr.execute("var x = 10")
        result = await mgr.execute("x * 2")
        assert result["result"] == "20"


@pytest.mark.asyncio
@pytest.mark.skipif(not _node_available, reason="Node.js not available")
async def test_integration_let_const_promotion() -> None:
    """let/const are promoted to var so they persist across calls."""
    async with JSReplManager() as mgr:
        await mgr.execute("let a = 5")
        await mgr.execute("const b = 3")
        result = await mgr.execute("a + b")
        assert result["result"] == "8"


@pytest.mark.asyncio
@pytest.mark.skipif(not _node_available, reason="Node.js not available")
async def test_integration_console_capture() -> None:
    """console.log output is captured and returned."""
    async with JSReplManager() as mgr:
        result = await mgr.execute("console.log('hello'); console.warn('oops')")
        assert "hello" in result["console"]
        assert "[warn] oops" in result["console"]


@pytest.mark.asyncio
@pytest.mark.skipif(not _node_available, reason="Node.js not available")
async def test_integration_reset_clears_state() -> None:
    """reset() clears the VM context."""
    async with JSReplManager() as mgr:
        await mgr.execute("var counter = 42")
        result = await mgr.reset()
        assert result["result"] == "context reset"

        # counter should no longer exist
        result = await mgr.execute("typeof counter")
        assert result["result"] == "undefined"


@pytest.mark.asyncio
@pytest.mark.skipif(not _node_available, reason="Node.js not available")
async def test_integration_error_handling() -> None:
    """Errors in JS code are reported as error field."""
    async with JSReplManager() as mgr:
        result = await mgr.execute("undeclaredVar.prop")
        assert result["error"] is not None
        assert "not defined" in result["error"] or "ReferenceError" in result["error"]


@pytest.mark.asyncio
@pytest.mark.skipif(not _node_available, reason="Node.js not available")
async def test_integration_function_persistence() -> None:
    """Functions assigned to var persist across calls."""
    async with JSReplManager() as mgr:
        await mgr.execute("var double = function(n) { return n * 2 }")
        result = await mgr.execute("double(21)")
        assert result["result"] == "42"


@pytest.mark.asyncio
@pytest.mark.skipif(not _node_available, reason="Node.js not available")
async def test_integration_empty_code() -> None:
    """Empty code produces an error."""
    async with JSReplManager() as mgr:
        result = await mgr.execute("   ")
        assert result["error"] is not None
