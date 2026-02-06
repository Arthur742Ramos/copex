from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock

from copex.mcp import _BLOCKED_ARG_CHARS_RE, _ALLOWED_COMMAND_RE


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
        from copex.mcp import StdioTransport, MCPServerConfig

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
        from copex.mcp import StdioTransport, MCPServerConfig

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
        from copex.mcp import StdioTransport, MCPServerConfig

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
