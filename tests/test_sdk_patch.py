from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import copex.sdk_patch as sdk_patch


@pytest.mark.asyncio
async def test_patched_start_cli_server_starts_stderr_drain_thread(monkeypatch) -> None:
    process = SimpleNamespace(stderr=MagicMock(), stdout=MagicMock())
    client = SimpleNamespace(
        options={
            "cli_path": "/usr/bin/copilot",
            "log_level": "warning",
            "github_token": None,
            "use_logged_in_user": True,
            "use_stdio": True,
            "cwd": None,
            "env": {},
            "port": 0,
        },
        _process=None,
    )

    monkeypatch.setattr("os.path.exists", lambda _path: True)

    with (
        patch("subprocess.Popen", return_value=process) as popen,
        patch("copex.sdk_patch.threading.Thread") as thread_cls,
    ):
        await sdk_patch._patched_start_cli_server(client)

    popen.assert_called_once()
    thread_cls.assert_called_once()
    thread_cls.return_value.start.assert_called_once()
