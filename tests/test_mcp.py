from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from copex.mcp import _ALLOWED_COMMAND_RE, _BLOCKED_ARG_CHARS_RE


class TestMCPArgValidation:
    def test_allows_simple_args(self):
        for arg in ["--verbose", "value", "my-server", "1.0.0"]:
            assert _BLOCKED_ARG_CHARS_RE.search(arg) is None

    def test_allows_windows_paths(self):
        for path in [r"C:\Users\me\server", r"D:\Program Files\app.exe"]:
            assert _BLOCKED_ARG_CHARS_RE.search(path) is None

    def test_allows_spaces_in_args(self):
        assert _BLOCKED_ARG_CHARS_RE.search("/path/to/my app") is None

    def test_allows_unix_paths(self):
        for path in ["/usr/local/bin/node", "./relative/path", "../parent"]:
            assert _BLOCKED_ARG_CHARS_RE.search(path) is None

    def test_allows_at_sign_and_equals(self):
        for arg in ["@scope/package", "--key=value", "user@host"]:
            assert _BLOCKED_ARG_CHARS_RE.search(arg) is None

    def test_blocks_semicolon(self):
        assert _BLOCKED_ARG_CHARS_RE.search("arg; rm -rf /") is not None

    def test_blocks_pipe(self):
        assert _BLOCKED_ARG_CHARS_RE.search("arg | cat") is not None

    def test_blocks_ampersand(self):
        assert _BLOCKED_ARG_CHARS_RE.search("arg && evil") is not None

    def test_blocks_dollar_sign(self):
        assert _BLOCKED_ARG_CHARS_RE.search("$HOME") is not None

    def test_blocks_backtick(self):
        assert _BLOCKED_ARG_CHARS_RE.search("`whoami`") is not None

    def test_blocks_parentheses(self):
        assert _BLOCKED_ARG_CHARS_RE.search("$(cmd)") is not None

    def test_blocks_angle_brackets(self):
        assert _BLOCKED_ARG_CHARS_RE.search("> /etc/passwd") is not None
        assert _BLOCKED_ARG_CHARS_RE.search("< /etc/shadow") is not None

    def test_blocks_null_byte(self):
        assert _BLOCKED_ARG_CHARS_RE.search("arg\x00evil") is not None

    def test_blocks_newline(self):
        assert _BLOCKED_ARG_CHARS_RE.search("arg\nevil") is not None

    def test_command_regex_still_strict(self):
        assert _ALLOWED_COMMAND_RE.match("npx") is not None
        assert _ALLOWED_COMMAND_RE.match("/usr/bin/node") is not None
        assert _ALLOWED_COMMAND_RE.match("command with spaces") is None
        assert _ALLOWED_COMMAND_RE.match(r"C:\path") is None


class TestStdioTransportEOF:
    @pytest.mark.asyncio
    async def test_eof_fails_pending_futures(self):
        """When reader loop hits EOF, pending futures get ConnectionError."""
        from copex.mcp import MCPServerConfig, StdioTransport

        transport = StdioTransport(MCPServerConfig(name="test", command="echo"))

        loop = asyncio.get_running_loop()
        future1 = loop.create_future()
        future2 = loop.create_future()
        transport._pending = {1: future1, 2: future2}

        mock_process = AsyncMock()
        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(return_value=b"")
        mock_process.stdout = mock_stdout
        transport._process = mock_process

        await transport._reader_loop()

        assert future1.done()
        assert future2.done()

        with pytest.raises(ConnectionError, match="MCP transport closed unexpectedly"):
            future1.result()
        with pytest.raises(ConnectionError, match="MCP transport closed unexpectedly"):
            future2.result()

    @pytest.mark.asyncio
    async def test_eof_skips_already_done_futures(self):
        """Already-completed futures should not be touched on EOF."""
        from copex.mcp import MCPServerConfig, StdioTransport

        transport = StdioTransport(MCPServerConfig(name="test", command="echo"))

        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        done_future.set_result("already done")
        pending_future = loop.create_future()
        transport._pending = {1: done_future, 2: pending_future}

        mock_process = AsyncMock()
        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(return_value=b"")
        mock_process.stdout = mock_stdout
        transport._process = mock_process

        await transport._reader_loop()

        assert done_future.result() == "already done"
        assert pending_future.done()
        with pytest.raises(ConnectionError):
            pending_future.result()

    @pytest.mark.asyncio
    async def test_eof_clears_pending_dict(self):
        """After EOF, self._pending should be empty."""
        from copex.mcp import MCPServerConfig, StdioTransport

        transport = StdioTransport(MCPServerConfig(name="test", command="echo"))

        loop = asyncio.get_running_loop()
        transport._pending = {1: loop.create_future()}

        mock_process = AsyncMock()
        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(return_value=b"")
        mock_process.stdout = mock_stdout
        transport._process = mock_process

        await transport._reader_loop()

        assert len(transport._pending) == 0


class TestStdioTransportSecurity:
    @pytest.mark.asyncio
    async def test_connect_filters_env_allowlist_and_keeps_explicit_config_env(self, monkeypatch):
        from copex.mcp import MCPServerConfig, StdioTransport

        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("HOME", "/home/test")
        monkeypatch.setenv("SECRET_TOKEN", "should-not-leak")

        captured: dict[str, object] = {}

        async def _fake_create_subprocess_exec(*_args, **kwargs):
            captured["env"] = kwargs.get("env")
            process = AsyncMock()
            process.stdout = AsyncMock()
            process.stdin = AsyncMock()
            process.terminate = lambda: None
            process.wait = AsyncMock(return_value=0)
            process.kill = lambda: None
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

        transport = StdioTransport(
            MCPServerConfig(
                name="test",
                command="echo",
                env={"CUSTOM_TOKEN": "ok", "PATH": "/custom/bin"},
            )
        )
        transport._initialize = AsyncMock()
        transport._reader_loop = AsyncMock()

        await transport.connect()
        await transport.disconnect()

        env = captured.get("env")
        assert isinstance(env, dict)
        assert env["PATH"] == "/custom/bin"
        assert env["HOME"] == "/home/test"
        assert env["CUSTOM_TOKEN"] == "ok"
        assert "SECRET_TOKEN" not in env


class TestStdioTransportFraming:
    @pytest.mark.asyncio
    async def test_reader_loop_supports_content_length_framing(self):
        from copex.mcp import MCPServerConfig, StdioTransport

        transport = StdioTransport(MCPServerConfig(name="test", command="echo"))
        loop = asyncio.get_running_loop()
        pending = loop.create_future()
        transport._pending = {1: pending}

        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}).encode("utf-8")
        mock_process = AsyncMock()
        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[
                f"Content-Length: {len(payload)}\r\n".encode(),
                b"\r\n",
                b"",
            ]
        )
        mock_stdout.readexactly = AsyncMock(return_value=payload)
        mock_process.stdout = mock_stdout
        transport._process = mock_process

        await transport._reader_loop()

        assert pending.done()
        assert pending.result() == {"ok": True}

    @pytest.mark.asyncio
    async def test_reader_loop_falls_back_to_newline_json(self):
        from copex.mcp import MCPServerConfig, StdioTransport

        transport = StdioTransport(MCPServerConfig(name="test", command="echo"))
        loop = asyncio.get_running_loop()
        pending = loop.create_future()
        transport._pending = {7: pending}

        line = json.dumps({"jsonrpc": "2.0", "id": 7, "result": "legacy"}).encode("utf-8") + b"\n"
        mock_process = AsyncMock()
        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(side_effect=[line, b""])
        mock_process.stdout = mock_stdout
        transport._process = mock_process

        await transport._reader_loop()

        assert pending.done()
        assert pending.result() == "legacy"
