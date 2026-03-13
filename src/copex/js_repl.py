"""Persistent JavaScript REPL manager for Copex.

Manages a long-running Node.js subprocess that evaluates JavaScript code
in a persistent VM context.  Communication uses newline-delimited JSON
over stdin/stdout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KERNEL_PATH = Path(__file__).parent / "kernel.js"
_DEFAULT_TIMEOUT = 35.0  # slightly above kernel's internal 30 s
_MAX_STDERR_CHARS = 1_200


def resolve_node_path(node_path: str | None = None) -> str | None:
    """Resolve a Node.js executable from an explicit path or PATH lookup."""
    if node_path:
        candidate = str(Path(node_path).expanduser())
        return shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
    return shutil.which("node")


class JSReplError(Exception):
    """Raised when the JS REPL kernel reports an error."""


class JSReplManager:
    """Async lifecycle manager for the Node.js REPL kernel."""

    def __init__(self, *, node_path: str | None = None) -> None:
        self._node = resolve_node_path(node_path) or node_path or "node"
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[bytes] | None = None
        self._lock = asyncio.Lock()
        self._expected_shutdown = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the kernel subprocess (idempotent)."""
        async with self._lock:
            if self._process is not None and self._process.returncode is None:
                return
            self._process = await asyncio.create_subprocess_exec(
                self._node,
                str(_KERNEL_PATH),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            process = self._process
            self._reader_task = asyncio.create_task(self._reader_loop(process))
            if self._process.stderr is not None:
                self._stderr_task = asyncio.create_task(self._process.stderr.read())
            await asyncio.sleep(0.01)
            if process.returncode is not None:
                await self._await_stderr_capture(timeout=0.1)
                error_message = self._build_process_error_message(expected_shutdown=False)
                await self._stop_unlocked(expected_shutdown=False)
                raise JSReplError(error_message)
            logger.debug("JS REPL kernel started (pid=%s)", self._process.pid)

    async def stop(self) -> None:
        """Terminate the kernel subprocess."""
        async with self._lock:
            await self._stop_unlocked()

    async def _stop_unlocked(self, *, expected_shutdown: bool = True) -> None:
        self._expected_shutdown = expected_shutdown
        reader_task = self._reader_task
        stderr_task = self._stderr_task
        process = self._process
        self._reader_task = None
        self._stderr_task = None
        self._process = None

        if process is not None:
            if process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass

        for task in (reader_task, stderr_task):
            if task is None:
                continue
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Fail all pending futures
        pending = dict(self._pending)
        self._pending.clear()
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("JS REPL kernel stopped"))
        self._expected_shutdown = False

    @property
    def running(self) -> bool:
        return (
            self._process is not None
            and self._process.returncode is None
            and self._reader_task is not None
            and not self._reader_task.done()
        )

    @property
    def node_path(self) -> str:
        """Return the Node.js executable path configured for the manager."""
        return self._node

    # ------------------------------------------------------------------
    # Request / response
    # ------------------------------------------------------------------

    async def execute(self, code: str, *, timeout: float = _DEFAULT_TIMEOUT) -> dict[str, Any]:
        """Execute *code* in the persistent JS context.

        Returns a dict with keys: ``result``, ``error``, ``console``.
        """
        return await self._send_request(
            {"code": code},
            timeout=timeout,
            timeout_error=f"JS execution timed out after {timeout:.0f}s",
        )

    async def reset(self, *, timeout: float = 10.0) -> dict[str, Any]:
        """Reset the VM context, clearing all state."""
        return await self._send_request(
            {"action": "reset"},
            timeout=timeout,
            timeout_error="JS REPL reset timed out",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _send_request(
        self,
        payload: dict[str, Any],
        *,
        timeout: float,
        timeout_error: str,
    ) -> dict[str, Any]:
        last_error = "JS REPL kernel disconnected"
        for attempt in range(2):
            try:
                await self._ensure_running()
            except JSReplError as exc:
                last_error = str(exc)
                if attempt == 0:
                    await self._restart()
                    continue
                raise

            self._request_id += 1
            req_id = self._request_id
            msg = json.dumps({"id": req_id, **payload}) + "\n"

            loop = asyncio.get_running_loop()
            future: asyncio.Future[dict[str, Any]] = loop.create_future()
            self._pending[req_id] = future

            try:
                assert self._process is not None and self._process.stdin is not None
                self._process.stdin.write(msg.encode())
                await self._process.stdin.drain()
                return await self._wait_for_response(future, timeout=timeout)
            except asyncio.TimeoutError:
                self._pending.pop(req_id, None)
                raise JSReplError(timeout_error) from None
            except ConnectionError as exc:
                self._pending.pop(req_id, None)
                last_error = str(exc)
            except (AssertionError, BrokenPipeError, ConnectionResetError, OSError):
                last_error = self._build_process_error_message(expected_shutdown=False)
                self._pending.pop(req_id, None)
                future.cancel()

            if attempt == 0:
                await self._restart()
                continue
            raise JSReplError(last_error)

        raise JSReplError(last_error)

    async def _ensure_running(self) -> None:
        """Auto-(re)start the kernel if it crashed or was never started."""
        if not self.running:
            async with self._lock:
                await self._stop_unlocked(expected_shutdown=False)
            await self.start()
            if not self.running:
                raise JSReplError(self._build_process_error_message(expected_shutdown=False))

    async def _restart(self) -> None:
        """Restart the kernel process after an unexpected disconnect."""
        async with self._lock:
            await self._stop_unlocked(expected_shutdown=False)
        await self.start()

    async def _await_stderr_capture(self, *, timeout: float) -> None:
        task = self._stderr_task
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            return

    def _stderr_snapshot(self) -> str:
        task = self._stderr_task
        if task is None or not task.done():
            return ""
        try:
            stderr_bytes = task.result()
        except asyncio.CancelledError:
            return ""
        except Exception:
            logger.debug("JS REPL stderr capture failed", exc_info=True)
            return ""

        text = stderr_bytes.decode("utf-8", errors="replace").strip()
        if not text:
            return ""
        compact = " | ".join(line.strip() for line in text.splitlines() if line.strip())
        if len(compact) <= _MAX_STDERR_CHARS:
            return compact
        return f"...{compact[-(_MAX_STDERR_CHARS - 3):]}"

    async def _wait_for_response(
        self,
        future: asyncio.Future[dict[str, Any]],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        reader_task = self._reader_task
        if reader_task is None:
            raise ConnectionError(self._build_process_error_message(expected_shutdown=False))

        done, _ = await asyncio.wait(
            {future, reader_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if future in done:
            return future.result()
        if reader_task in done:
            error_message = self._build_process_error_message(expected_shutdown=False)
            if not future.done():
                future.set_exception(ConnectionError(error_message))
            raise ConnectionError(error_message)
        raise asyncio.TimeoutError

    def _build_process_error_message(self, *, expected_shutdown: bool) -> str:
        base = "JS REPL kernel stopped" if expected_shutdown else "JS REPL kernel exited unexpectedly"
        stderr = self._stderr_snapshot()
        if stderr:
            return f"{base}: {stderr}"
        return base

    async def _reader_loop(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    logger.debug("JS REPL: non-JSON output: %s", line[:200])
                    continue

                req_id = msg.get("id")
                if req_id is not None and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        fut.set_result(msg)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.debug("JS REPL reader loop error", exc_info=True)
        finally:
            pending = dict(self._pending)
            self._pending.clear()
            await self._await_stderr_capture(timeout=0.1)
            error_message = self._build_process_error_message(expected_shutdown=self._expected_shutdown)
            for fut in pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError(error_message))

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> JSReplManager:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
