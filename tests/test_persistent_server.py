from __future__ import annotations

import json
from pathlib import Path

from copex.persistent_server import PersistentCopilotServer, PersistentServerState


class _FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid


def _write_state(path: Path, *, pid: int = 1234, host: str = "127.0.0.1", port: int = 8765) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "host": host,
                "port": port,
                "started_at": "2026-03-04T00:00:00Z",
                "cli_path": "/usr/bin/copilot",
            }
        ),
        encoding="utf-8",
    )


def test_resolve_cli_url_reuses_healthy_state(tmp_path: Path, monkeypatch) -> None:
    state_file = tmp_path / "server.json"
    _write_state(state_file, pid=5555, port=9988)

    manager = PersistentCopilotServer(
        cli_path="/usr/bin/copilot",
        state_file=state_file,
        host="127.0.0.1",
        port=9988,
    )
    monkeypatch.setattr(manager, "_state_is_healthy", lambda _state: True)
    monkeypatch.setattr(manager, "start", lambda: (_ for _ in ()).throw(AssertionError("unexpected start")))

    cli_url = manager.resolve_cli_url(auto_start=True)

    assert cli_url == "127.0.0.1:9988"


def test_resolve_cli_url_starts_server_when_missing(tmp_path: Path, monkeypatch) -> None:
    state_file = tmp_path / "server.json"
    manager = PersistentCopilotServer(
        cli_path="/usr/bin/copilot",
        state_file=state_file,
        host="127.0.0.1",
        port=8765,
    )
    monkeypatch.setattr(
        manager,
        "start",
        lambda: PersistentServerState(
            pid=6789,
            host="127.0.0.1",
            port=8765,
            started_at="2026-03-04T00:00:00Z",
            cli_path="/usr/bin/copilot",
        ),
    )

    cli_url = manager.resolve_cli_url(auto_start=True)

    assert cli_url == "127.0.0.1:8765"


def test_start_writes_state_file(tmp_path: Path, monkeypatch) -> None:
    state_file = tmp_path / "server.json"
    manager = PersistentCopilotServer(
        cli_path="/usr/bin/copilot",
        state_file=state_file,
        host="127.0.0.1",
        port=7878,
    )
    monkeypatch.setattr(manager, "_spawn_process", lambda: _FakeProcess(pid=2468))
    monkeypatch.setattr(manager, "_wait_for_healthy", lambda _host, _port: True)

    state = manager.start()

    assert state.pid == 2468
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["pid"] == 2468
    assert payload["port"] == 7878


def test_stop_removes_state_file(tmp_path: Path, monkeypatch) -> None:
    state_file = tmp_path / "server.json"
    _write_state(state_file, pid=3333)
    manager = PersistentCopilotServer(
        cli_path="/usr/bin/copilot",
        state_file=state_file,
    )
    monkeypatch.setattr(manager, "_terminate_pid", lambda _pid: True)

    stopped = manager.stop()

    assert stopped is True
    assert not state_file.exists()
