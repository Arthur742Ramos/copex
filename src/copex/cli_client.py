"""CLI subprocess wrapper that bypasses the SDK to access all models.

Uses ``copilot -p "prompt" --model X`` (the non-headless code path) so that
models like Opus 4.6 / 4.6-fast are used directly without fallback.

Reasoning effort is injected via a temporary ``--config-dir`` containing a
``config.json`` with the desired ``reasoning_effort`` value.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from copex.backoff import AdaptiveRetry, BackoffStrategy, ErrorCategory
from copex.client import Response, StreamChunk, StreamingMetrics
from copex.config import CopexConfig, find_copilot_cli
from copex.exceptions import CopexError

logger = logging.getLogger(__name__)


class CopilotCLI:
    """CLI subprocess wrapper that bypasses the SDK to access all models."""

    def __init__(self, config: CopexConfig | None = None) -> None:
        self.config = config or CopexConfig()
        self._cli_path = self.config.cli_path or find_copilot_cli()
        if not self._cli_path:
            raise CopexError("Copilot CLI not found. Install with: npm i -g @github/copilot")
        self._config_dir: str | None = None
        self._has_session = False
        self._retry = AdaptiveRetry(
            strategies={
                ErrorCategory.TRANSIENT: BackoffStrategy(
                    base_delay=self.config.retry.base_delay,
                    max_delay=self.config.retry.max_delay,
                    multiplier=self.config.retry.exponential_base,
                    jitter=0.2,
                    max_retries=self.config.retry.max_retries,
                ),
                ErrorCategory.UNKNOWN: BackoffStrategy(
                    base_delay=self.config.retry.base_delay,
                    max_delay=self.config.retry.max_delay,
                    multiplier=self.config.retry.exponential_base,
                    jitter=0.2,
                    max_retries=self.config.retry.max_retries,
                ),
            },
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """No-op — CLI client doesn't need a persistent connection."""
        if self._config_dir is None:
            self._config_dir = self._make_config_dir()

    async def stop(self) -> None:
        """Clean up temporary config directory."""
        self._cleanup()

    async def __aenter__(self) -> CopilotCLI:
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()

    def _cleanup(self) -> None:
        if self._config_dir:
            shutil.rmtree(self._config_dir, ignore_errors=True)
            self._config_dir = None

    def __del__(self) -> None:
        self._cleanup()

    # ------------------------------------------------------------------
    # Config-dir hack for reasoning effort
    # ------------------------------------------------------------------

    def _make_config_dir(self) -> str:
        config_dir = tempfile.mkdtemp(prefix="copex-cli-")
        config: dict[str, Any] = {
            "reasoning_effort": self.config.reasoning_effort.value,
            "banner": "never",
            "model": self.config.model.value,
        }

        # Copy MCP config if user specified one
        if self.config.mcp_config_file:
            src = Path(self.config.mcp_config_file)
            if src.exists():
                dst = Path(config_dir) / src.name
                shutil.copy2(src, dst)

        config_path = Path(config_dir) / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config_dir

    # ------------------------------------------------------------------
    # Build CLI command
    # ------------------------------------------------------------------

    def _build_command(self, prompt: str) -> list[str]:
        assert self._cli_path is not None
        config_dir = self._config_dir or self._make_config_dir()
        if self._config_dir is None:
            self._config_dir = config_dir

        # Prepend custom instructions to the prompt if set
        effective_prompt = prompt
        instructions = self._resolve_instructions()
        if instructions:
            effective_prompt = f"{instructions}\n\n{prompt}"

        cmd = [
            self._cli_path,
            "-p", effective_prompt,
            "--model", self.config.model.value,
            "--config-dir", config_dir,
            "--no-color",
            "-s",
            "--allow-all",
        ]

        if self.config.streaming:
            cmd.extend(["--stream", "on"])

        # Session continuity
        if self._has_session:
            cmd.append("--continue")

        # MCP configuration
        mcp_path = self._resolve_mcp_config()
        if mcp_path:
            cmd.extend(["--additional-mcp-config", mcp_path])

        # Tool filtering
        if self.config.excluded_tools:
            for tool in self.config.excluded_tools:
                cmd.extend(["--excluded-tools", tool])
        if self.config.available_tools:
            for tool in self.config.available_tools:
                cmd.extend(["--available-tools", tool])

        # Working directory
        if self.config.working_directory:
            cmd.extend(["--add-dir", self.config.working_directory])

        return cmd

    def _resolve_instructions(self) -> str | None:
        if self.config.instructions:
            return self.config.instructions
        if self.config.instructions_file:
            p = Path(self.config.instructions_file)
            if p.exists():
                return p.read_text(encoding="utf-8")
        return None

    def _resolve_mcp_config(self) -> str | None:
        """Return path to an MCP config file for ``--additional-mcp-config``."""
        if self.config.mcp_config_file:
            p = Path(self.config.mcp_config_file)
            if p.exists():
                return str(p)

        if self.config.mcp_servers:
            # Build a temporary MCP config JSON from inline server dicts
            assert self._config_dir is not None
            mcp_path = Path(self._config_dir) / "mcp_servers.json"
            servers: dict[str, Any] = {}
            for i, srv in enumerate(self.config.mcp_servers):
                name = srv.get("name", f"server-{i}")
                servers[name] = {k: v for k, v in srv.items() if k != "name"}
            mcp_path.write_text(
                json.dumps({"servers": servers}, indent=2), encoding="utf-8"
            )
            return str(mcp_path)

        return None

    # ------------------------------------------------------------------
    # Subprocess execution
    # ------------------------------------------------------------------

    async def _run_streaming(
        self,
        cmd: list[str],
        on_chunk: Callable[[StreamChunk], None] | None = None,
    ) -> tuple[str, StreamingMetrics]:
        cwd = self.config.working_directory or None
        metrics = StreamingMetrics(_start_time=time.monotonic())

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        content: list[str] = []
        assert process.stdout is not None

        try:
            async for line in process.stdout:
                text = line.decode("utf-8", errors="replace")
                content.append(text)

                now = time.monotonic()
                if metrics.first_chunk_time is None:
                    metrics.first_chunk_time = now
                metrics.last_chunk_time = now
                metrics.total_chunks += 1
                metrics.total_bytes += len(text)
                metrics.message_chunks += 1

                if on_chunk:
                    on_chunk(StreamChunk(type="message", delta=text))

            await process.wait()
        except BaseException:
            process.kill()
            await process.wait()
            raise

        if process.returncode != 0:
            assert process.stderr is not None
            stderr_bytes = await process.stderr.read()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")
            raise CopexError(
                f"CLI exited with code {process.returncode}: {stderr_text}",
                context={"returncode": process.returncode},
            )

        self._has_session = True
        return "".join(content), metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(
        self,
        prompt: str,
        *,
        tools: list[Any] | None = None,
        on_chunk: Callable[[StreamChunk], None] | None = None,
        metrics: Any | None = None,
    ) -> Response:
        """Send a prompt to the Copilot CLI with automatic retry.

        Args:
            prompt: The prompt to send.
            tools: Ignored (CLI manages its own tools via ``--allow-all``).
            on_chunk: Optional callback for streaming chunks.
            metrics: Ignored (kept for interface compatibility).

        Returns:
            Response object with content and metadata.
        """
        cmd = self._build_command(prompt)
        retry_count = 0

        async def _attempt() -> tuple[str, StreamingMetrics]:
            nonlocal retry_count
            try:
                return await self._run_streaming(cmd, on_chunk)
            except CopexError:
                retry_count += 1
                raise

        output, streaming_metrics = await self._retry.execute(_attempt)

        return Response(
            content=output,
            reasoning="",
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            retries=retry_count,
            streaming_metrics=streaming_metrics,
        )

    async def stream(
        self,
        prompt: str,
        *,
        tools: list[Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from the Copilot CLI.

        Yields StreamChunk objects as they arrive.
        """
        queue: asyncio.Queue[StreamChunk | None | BaseException] = asyncio.Queue()

        def on_chunk(chunk: StreamChunk) -> None:
            queue.put_nowait(chunk)

        async def sender() -> None:
            try:
                await self.send(prompt, tools=tools, on_chunk=on_chunk)
                queue.put_nowait(None)
            except BaseException as e:
                queue.put_nowait(e)

        task = asyncio.create_task(sender())

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def chat(self, prompt: str) -> str:
        """Simple interface — send prompt, get response content."""
        response = await self.send(prompt)
        return response.content

    def new_session(self) -> None:
        """Reset session state so the next call starts fresh."""
        self._has_session = False
