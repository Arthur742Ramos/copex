"""Tests for CopilotCLI subprocess wrapper."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from copex.cli_client import CopilotCLI
from copex.client import StreamChunk
from copex.config import CopexConfig
from copex.exceptions import CopexError
from copex.models import Model, ReasoningEffort

# Use a model that supports reasoning so CopexConfig doesn't normalize to NONE
_DEFAULT_MODEL = Model.CLAUDE_OPUS_4_6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cli(
    cli_path: str = "/usr/bin/copilot",
    model: Model = _DEFAULT_MODEL,
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    **kwargs,
) -> CopilotCLI:
    """Build a CopilotCLI with a fake cli_path so it never looks up the real binary."""
    cfg = CopexConfig(
        cli_path=cli_path,
        model=model,
        reasoning_effort=reasoning_effort,
        **kwargs,
    )
    return _make_cli_from_config(cfg)


def _make_cli_from_config(cfg: CopexConfig) -> CopilotCLI:
    with patch("copex.cli_client.find_copilot_cli", return_value=cfg.cli_path or "/usr/bin/copilot"):
        return CopilotCLI(cfg)


class _AsyncLineIterator:
    """Async iterator over a list of byte lines, mimicking subprocess stdout."""

    def __init__(self, lines: list[bytes]):
        self._iter = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def _fake_process(stdout_lines: list[bytes], returncode: int = 0, stderr: bytes = b""):
    """Create a mock asyncio subprocess."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.stdout = _AsyncLineIterator(stdout_lines)

    stderr_reader = AsyncMock()
    stderr_reader.read = AsyncMock(return_value=stderr)
    proc.stderr = stderr_reader

    proc.wait = AsyncMock()
    proc.kill = AsyncMock()
    return proc


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_config(self):
        cli = _make_cli()
        assert cli.config.model == _DEFAULT_MODEL
        assert cli._cli_path == "/usr/bin/copilot"
        assert cli._config_dir is None
        assert cli._has_session is False

    def test_custom_model(self):
        cli = _make_cli(model=Model.GPT_5_1_CODEX)
        assert cli.config.model == Model.GPT_5_1_CODEX

    def test_raises_when_cli_not_found(self):
        cfg = CopexConfig(cli_path=None)
        with patch("copex.cli_client.find_copilot_cli", return_value=None):
            with pytest.raises(CopexError, match="Copilot CLI not found"):
                CopilotCLI(cfg)

    def test_uses_find_copilot_cli_when_no_path(self):
        cfg = CopexConfig(cli_path=None)
        with patch("copex.cli_client.find_copilot_cli", return_value="/opt/copilot"):
            cli = CopilotCLI(cfg)
            assert cli._cli_path == "/opt/copilot"


# ---------------------------------------------------------------------------
# _make_config_dir
# ---------------------------------------------------------------------------


class TestMakeConfigDir:
    def test_creates_dir_with_config_json(self):
        cli = _make_cli(reasoning_effort=ReasoningEffort.HIGH)
        config_dir = cli._make_config_dir()
        try:
            p = Path(config_dir) / "config.json"
            assert p.exists()
            data = json.loads(p.read_text())
            assert data["reasoning_effort"] == "high"
            assert data["banner"] == "never"
        finally:
            import shutil
            shutil.rmtree(config_dir, ignore_errors=True)

    def test_reasoning_effort_values(self):
        for effort in [ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH]:
            cli = _make_cli(reasoning_effort=effort)
            config_dir = cli._make_config_dir()
            try:
                data = json.loads((Path(config_dir) / "config.json").read_text())
                assert data["reasoning_effort"] == effort.value
            finally:
                import shutil
                shutil.rmtree(config_dir, ignore_errors=True)

    def test_model_written_to_config(self):
        cli = _make_cli(model=Model.CLAUDE_OPUS_4_6)
        config_dir = cli._make_config_dir()
        try:
            data = json.loads((Path(config_dir) / "config.json").read_text())
            assert data["model"] == Model.CLAUDE_OPUS_4_6.value
        finally:
            import shutil
            shutil.rmtree(config_dir, ignore_errors=True)

    def test_cleanup_on_del(self):
        cli = _make_cli()
        config_dir = cli._make_config_dir()
        cli._config_dir = config_dir
        assert Path(config_dir).exists()
        cli.__del__()
        assert not Path(config_dir).exists()
        assert cli._config_dir is None


# ---------------------------------------------------------------------------
# _build_command
# ---------------------------------------------------------------------------


class TestBuildCommand:
    def test_basic_command(self):
        cli = _make_cli()
        cmd = cli._build_command("hello")
        assert cmd[0] == "/usr/bin/copilot"
        assert "-p" in cmd
        assert cmd[cmd.index("-p") + 1] == "hello"
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == _DEFAULT_MODEL.value
        assert "--no-color" in cmd
        assert "-s" in cmd
        assert "--allow-all" in cmd
        # cleanup
        cli._cleanup()

    def test_streaming_flag(self):
        cli = _make_cli(streaming=True)
        cmd = cli._build_command("test")
        assert "--stream" in cmd
        assert cmd[cmd.index("--stream") + 1] == "on"
        cli._cleanup()

    def test_no_streaming_flag(self):
        cli = _make_cli(streaming=False)
        cmd = cli._build_command("test")
        assert "--stream" not in cmd
        cli._cleanup()

    def test_config_dir_in_command(self):
        cli = _make_cli()
        cmd = cli._build_command("prompt")
        assert "--config-dir" in cmd
        config_dir = cmd[cmd.index("--config-dir") + 1]
        assert Path(config_dir).exists()
        cli._cleanup()

    def test_continue_flag_after_session(self):
        cli = _make_cli()
        cli._has_session = True
        cmd = cli._build_command("follow up")
        assert "--continue" in cmd
        cli._cleanup()

    def test_no_continue_on_first_call(self):
        cli = _make_cli()
        cmd = cli._build_command("first prompt")
        assert "--continue" not in cmd
        cli._cleanup()

    def test_excluded_tools(self):
        cli = _make_cli(excluded_tools=["tool_a", "tool_b"])
        cmd = cli._build_command("prompt")
        indices = [i for i, v in enumerate(cmd) if v == "--excluded-tools"]
        assert len(indices) == 2
        tools = [cmd[i + 1] for i in indices]
        assert "tool_a" in tools
        assert "tool_b" in tools
        cli._cleanup()

    def test_available_tools(self):
        cli = _make_cli(available_tools=["read_file"])
        cmd = cli._build_command("prompt")
        idx = cmd.index("--available-tools")
        assert cmd[idx + 1] == "read_file"
        cli._cleanup()

    def test_working_directory(self):
        cli = _make_cli(working_directory="/tmp/work")
        cmd = cli._build_command("prompt")
        assert "--add-dir" in cmd
        assert cmd[cmd.index("--add-dir") + 1] == "/tmp/work"
        cli._cleanup()

    def test_instructions_prepended(self):
        cli = _make_cli(instructions="Be helpful")
        cmd = cli._build_command("do something")
        prompt = cmd[cmd.index("-p") + 1]
        assert prompt.startswith("Be helpful")
        assert "do something" in prompt
        cli._cleanup()


# ---------------------------------------------------------------------------
# send()
# ---------------------------------------------------------------------------


class TestSend:
    @pytest.mark.asyncio
    async def test_successful_send(self):
        cli = _make_cli()
        proc = _fake_process([b"Hello ", b"world\n"])

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            response = await cli.send("test prompt")

        assert response.content == "Hello world\n"
        assert response.retries == 0
        cli._cleanup()

    @pytest.mark.asyncio
    async def test_send_with_on_chunk(self):
        cli = _make_cli()
        proc = _fake_process([b"line1\n", b"line2\n"])
        chunks: list[StreamChunk] = []

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            response = await cli.send("test", on_chunk=chunks.append)

        assert len(chunks) == 2
        assert all(c.type == "message" for c in chunks)
        assert chunks[0].delta == "line1\n"
        assert chunks[1].delta == "line2\n"
        cli._cleanup()

    @pytest.mark.asyncio
    async def test_send_nonzero_exit_code(self):
        cli = _make_cli()
        proc = _fake_process(
            stdout_lines=[b"partial"],
            returncode=1,
            stderr=b"something went wrong",
        )

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with pytest.raises(CopexError, match="CLI exited with code 1"):
                await cli.send("test")

        cli._cleanup()

    @pytest.mark.asyncio
    async def test_send_sets_has_session(self):
        cli = _make_cli()
        assert cli._has_session is False
        proc = _fake_process([b"ok\n"])

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            await cli.send("hello")

        assert cli._has_session is True
        cli._cleanup()

    @pytest.mark.asyncio
    async def test_send_empty_prompt(self):
        cli = _make_cli()
        proc = _fake_process([b"response\n"])

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            response = await cli.send("")

        assert response.content == "response\n"
        cli._cleanup()

    @pytest.mark.asyncio
    async def test_send_streaming_metrics(self):
        cli = _make_cli()
        proc = _fake_process([b"chunk1", b"chunk2"])

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            response = await cli.send("test")

        assert response.streaming_metrics is not None
        assert response.streaming_metrics.total_chunks == 2
        assert response.streaming_metrics.message_chunks == 2
        assert response.streaming_metrics.total_bytes > 0
        cli._cleanup()


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        cli = _make_cli()
        proc = _fake_process([b"A", b"B", b"C"])

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            chunks = []
            async for chunk in cli.stream("test"):
                chunks.append(chunk)

        assert len(chunks) == 3
        assert [c.delta for c in chunks] == ["A", "B", "C"]
        cli._cleanup()

    @pytest.mark.asyncio
    async def test_stream_propagates_error(self):
        cli = _make_cli()
        proc = _fake_process(stdout_lines=[], returncode=1, stderr=b"fail")

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with pytest.raises(CopexError, match="CLI exited with code 1"):
                async for _ in cli.stream("test"):
                    pass

        cli._cleanup()


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------


class TestChat:
    @pytest.mark.asyncio
    async def test_chat_returns_content(self):
        cli = _make_cli()
        proc = _fake_process([b"answer\n"])

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await cli.chat("question")

        assert result == "answer\n"
        cli._cleanup()


# ---------------------------------------------------------------------------
# Resource cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    @pytest.mark.asyncio
    async def test_context_manager_creates_and_cleans(self):
        cli = _make_cli()
        async with cli:
            assert cli._config_dir is not None
            config_dir = cli._config_dir
            assert Path(config_dir).exists()
        assert cli._config_dir is None
        assert not Path(config_dir).exists()

    @pytest.mark.asyncio
    async def test_stop_cleans_config_dir(self):
        cli = _make_cli()
        await cli.start()
        config_dir = cli._config_dir
        assert config_dir is not None
        await cli.stop()
        assert cli._config_dir is None
        assert not Path(config_dir).exists()

    def test_cleanup_idempotent(self):
        cli = _make_cli()
        cli._cleanup()  # no config dir set — should not raise
        cli._cleanup()  # still fine


# ---------------------------------------------------------------------------
# new_session
# ---------------------------------------------------------------------------


class TestNewSession:
    def test_new_session_resets_flag(self):
        cli = _make_cli()
        cli._has_session = True
        cli.new_session()
        assert cli._has_session is False


# ---------------------------------------------------------------------------
# MCP config resolution
# ---------------------------------------------------------------------------


class TestMCPConfig:
    def test_mcp_config_file_in_command(self, tmp_path):
        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text('{"servers":{}}')
        cli = _make_cli(mcp_config_file=str(mcp_file))
        cmd = cli._build_command("test")
        assert "--additional-mcp-config" in cmd
        assert str(mcp_file) in cmd
        cli._cleanup()

    def test_mcp_servers_inline(self):
        cli = _make_cli(mcp_servers=[{"name": "my-server", "command": "node", "args": ["server.js"]}])
        cmd = cli._build_command("test")
        assert "--additional-mcp-config" in cmd
        # Should have written a temp mcp_servers.json
        idx = cmd.index("--additional-mcp-config")
        mcp_path = Path(cmd[idx + 1])
        assert mcp_path.exists()
        data = json.loads(mcp_path.read_text())
        assert "my-server" in data["servers"]
        cli._cleanup()
