from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from copex.mcp import MCPServerConfig, StdioTransport


class _FakeStdin:
    def __init__(self):
        self.writes: list[bytes] = []

    def write(self, data: bytes):
        self.writes.append(data)

    async def drain(self):
        return None


class _FakeStdout:
    def __init__(self, lines):
        self._lines = iter(lines)

    async def readline(self):
        item = next(self._lines)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeProcess:
    def __init__(self, lines):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(lines)
        self.terminate_called = False
        self.kill_called = False
        self.wait = AsyncMock(return_value=0)

    def terminate(self):
        self.terminate_called = True

    def kill(self):
        self.kill_called = True


@pytest.mark.asyncio
async def test_stdio_transport_connect_disconnect_lifecycle():
    config = MCPServerConfig(name="demo", command="python3", args=["-V"])
    process = _FakeProcess([b""])

    with (
        patch("copex.mcp.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as mock_exec,
        patch.object(StdioTransport, "_initialize", new=AsyncMock(return_value=None)),
    ):
        transport = StdioTransport(config)
        await transport.connect()
        await transport.disconnect()

    mock_exec.assert_awaited_once()
    assert process.terminate_called is True


@pytest.mark.asyncio
async def test_disconnect_kills_process_when_wait_times_out():
    config = MCPServerConfig(name="demo", command="python3")
    process = _FakeProcess([b""])
    process.wait = AsyncMock(side_effect=asyncio.TimeoutError)
    transport = StdioTransport(config)
    transport._process = process
    transport._reader_task = asyncio.create_task(asyncio.sleep(10))

    await transport.disconnect()

    assert process.terminate_called is True
    assert process.kill_called is True


@pytest.mark.asyncio
async def test_send_resolves_pending_future_with_response():
    transport = StdioTransport(MCPServerConfig(name="demo", command="python3"))

    async def _fake_write(message):
        transport._pending[message["id"]].set_result({"ok": True})

    with patch.object(transport, "_write", side_effect=_fake_write):
        result = await transport.send({"jsonrpc": "2.0", "method": "ping"})

    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_reader_loop_handles_malformed_then_valid_json():
    transport = StdioTransport(MCPServerConfig(name="demo", command="python3"))
    transport._process = _FakeProcess([
        b"not-json\n",
        b'{"id": 1, "result": {"ok": true}}\n',
        b"",
    ])

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    transport._pending = {1: future}

    await transport._reader_loop()

    assert future.result() == {"ok": True}


@pytest.mark.asyncio
async def test_reader_loop_recovers_from_reader_exception():
    transport = StdioTransport(MCPServerConfig(name="demo", command="python3"))
    transport._process = _FakeProcess([
        RuntimeError("read error"),
        b'{"id": 1, "result": {"ok": true}}\n',
        b"",
    ])

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    transport._pending = {1: future}

    await transport._reader_loop()

    assert future.result() == {"ok": True}


@pytest.mark.asyncio
async def test_reader_loop_eof_sets_connection_error_on_pending_future():
    transport = StdioTransport(MCPServerConfig(name="demo", command="python3"))
    transport._process = _FakeProcess([
        b"not-json\n",
        b"",
    ])

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    transport._pending = {1: future}

    await transport._reader_loop()

    with pytest.raises(ConnectionError, match="closed unexpectedly"):
        future.result()


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="Content-Length framing is not fully supported by newline-based reader loop yet.",
)
async def test_reader_loop_parses_content_length_framed_messages():
    transport = StdioTransport(MCPServerConfig(name="demo", command="python3"))
    transport._process = _FakeProcess([
        b"Content-Length: 72\r\n",
        b"\r\n",
        b"{\n",
        b'  "jsonrpc": "2.0",\n',
        b'  "id": 1,\n',
        b'  "result": {"ok": true}\n',
        b"}\n",
        b"",
    ])

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    transport._pending = {1: future}

    await transport._reader_loop()

    assert future.result() == {"ok": True}
