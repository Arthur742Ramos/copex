"""Persistent Copilot CLI server lifecycle management."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PersistentServerState:
    pid: int
    host: str
    port: int
    started_at: str
    cli_path: str


class PersistentCopilotServer:
    """Manage a long-lived headless Copilot CLI server process."""

    def __init__(
        self,
        *,
        cli_path: str,
        host: str = "127.0.0.1",
        port: int = 8765,
        state_file: Path | None = None,
        log_level: str = "warning",
        github_token: str | None = None,
        use_logged_in_user: bool = True,
        cwd: Path | None = None,
        startup_timeout: float = 12.0,
    ) -> None:
        self.cli_path = cli_path
        self.host = host
        self.port = port
        self.state_file = state_file or (Path.home() / ".copex" / "cli-server.json")
        self.log_level = log_level
        self.github_token = github_token
        self.use_logged_in_user = use_logged_in_user
        self.cwd = cwd
        self.startup_timeout = startup_timeout

    def resolve_cli_url(self, *, auto_start: bool = True) -> str | None:
        """Return a healthy CLI URL, starting the server when requested."""

        state = self._load_state()
        if state and self._state_is_healthy(state):
            return self._format_cli_url(state.host, state.port)

        if state:
            self._delete_state_file()

        if not auto_start:
            return None

        started = self.start()
        return self._format_cli_url(started.host, started.port)

    def start(self) -> PersistentServerState:
        """Start the persistent server and save fresh state."""

        state = self._load_state()
        if state and self._state_is_healthy(state):
            return state

        if state:
            self._delete_state_file()

        process = self._spawn_process()
        started = PersistentServerState(
            pid=process.pid,
            host=self.host,
            port=self.port,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            cli_path=self.cli_path,
        )
        if not self._wait_for_healthy(started.host, started.port):
            self._terminate_pid(started.pid)
            raise RuntimeError(
                f"Timed out waiting for persistent Copilot server at "
                f"{started.host}:{started.port} (pid={started.pid})"
            )
        self._save_state(started)
        return started

    def stop(self) -> bool:
        """Stop the persistent server process from saved state."""

        state = self._load_state()
        if state is None:
            self._delete_state_file()
            return False

        terminated = self._terminate_pid(state.pid)
        self._delete_state_file()
        return terminated

    def status(self) -> dict[str, Any]:
        """Return current persisted server status."""

        state = self._load_state()
        if state is None:
            return {"running": False}
        healthy = self._state_is_healthy(state)
        return {
            "running": healthy,
            "pid": state.pid,
            "host": state.host,
            "port": state.port,
            "cli_url": self._format_cli_url(state.host, state.port),
            "started_at": state.started_at,
            "cli_path": state.cli_path,
        }

    def _format_cli_url(self, host: str, port: int) -> str:
        return f"{host}:{port}"

    def _load_state(self) -> PersistentServerState | None:
        if not self.state_file.exists():
            return None
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None

        try:
            return PersistentServerState(
                pid=int(raw["pid"]),
                host=str(raw["host"]),
                port=int(raw["port"]),
                started_at=str(raw.get("started_at", "")),
                cli_path=str(raw.get("cli_path", self.cli_path)),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _save_state(self, state: PersistentServerState) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": state.pid,
            "host": state.host,
            "port": state.port,
            "started_at": state.started_at,
            "cli_path": state.cli_path,
        }
        self.state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _delete_state_file(self) -> None:
        try:
            self.state_file.unlink()
        except FileNotFoundError:
            return
        except OSError:
            return

    def _state_is_healthy(self, state: PersistentServerState) -> bool:
        return self._pid_is_alive(state.pid) and self._is_server_healthy(state.host, state.port)

    def _pid_is_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _is_server_healthy(self, host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    def _wait_for_healthy(self, host: str, port: int) -> bool:
        deadline = time.monotonic() + max(1.0, self.startup_timeout)
        while time.monotonic() < deadline:
            if self._is_server_healthy(host, port):
                return True
            time.sleep(0.2)
        return False

    def _spawn_process(self) -> subprocess.Popen[bytes]:
        if not Path(self.cli_path).exists():
            raise RuntimeError(f"Copilot CLI not found at {self.cli_path}")

        args = [self.cli_path, "--headless", "--log-level", self.log_level, "--port", str(self.port)]
        env = dict(os.environ)
        if self.github_token:
            env["COPILOT_SDK_AUTH_TOKEN"] = self.github_token
            args.extend(["--auth-token-env", "COPILOT_SDK_AUTH_TOKEN"])
        if not self.use_logged_in_user:
            args.append("--no-auto-login")

        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": env,
        }
        if self.cwd is not None:
            popen_kwargs["cwd"] = str(self.cwd)

        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        return subprocess.Popen(args, **popen_kwargs)

    def _terminate_pid(self, pid: int) -> bool:
        if not self._pid_is_alive(pid):
            return False
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return False

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not self._pid_is_alive(pid):
                return True
            time.sleep(0.1)

        if hasattr(signal, "SIGKILL"):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                return False
            return not self._pid_is_alive(pid)
        return False


def resolve_persistent_cli_url(
    *,
    enabled: bool,
    cli_path: str,
    host: str,
    port: int,
    state_file: Path,
    auto_start: bool,
    log_level: str,
    github_token: str | None,
    use_logged_in_user: bool,
    cwd: Path | None = None,
) -> str | None:
    """Resolve a persistent CLI URL using the configured server manager."""

    if not enabled:
        return None
    manager = PersistentCopilotServer(
        cli_path=cli_path,
        host=host,
        port=port,
        state_file=state_file,
        log_level=log_level,
        github_token=github_token,
        use_logged_in_user=use_logged_in_user,
        cwd=cwd,
    )
    return manager.resolve_cli_url(auto_start=auto_start)
