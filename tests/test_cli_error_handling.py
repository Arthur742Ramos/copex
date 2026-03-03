"""Regression tests for actionable CLI error handling output."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

import copex.cli_plan as cli_plan
import copex.cli_squad as cli_squad
from copex.config import CopexConfig


def test_run_plan_reports_actionable_error_and_stops_client(monkeypatch, capsys):
    """Plan mode should surface error type + guidance and still stop the client."""
    client = MagicMock()
    client.start = AsyncMock()
    client.stop = AsyncMock()
    monkeypatch.setattr(cli_plan, "make_client", lambda _config: client)
    monkeypatch.setattr(
        cli_plan.PlanExecutor,
        "generate_plan",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    with pytest.raises(typer.Exit):
        asyncio.run(
            cli_plan._run_plan(
                config=CopexConfig(),
                task="test task",
                execute=False,
                review=False,
                resume=False,
                output=None,
                from_step=1,
                load_plan=None,
                max_iterations=1,
                visualize=None,
            )
        )

    output = capsys.readouterr().out
    assert "Plan failed (RuntimeError): boom" in output
    assert "Run without --execute to inspect steps" in output
    client.stop.assert_awaited_once()


def test_run_squad_reports_actionable_error_in_text_mode(capsys):
    """Squad text mode should print error type and a follow-up hint."""
    with patch(
        "copex.squad_team.SquadTeam.load_squad_file",
        side_effect=RuntimeError("broken squad state"),
    ):
        with pytest.raises(typer.Exit):
            asyncio.run(
                cli_squad._run_squad(
                    CopexConfig(),
                    "prompt",
                    json_output=False,
                )
            )

    output = capsys.readouterr().out
    assert "Squad failed (RuntimeError): broken squad state" in output
    assert "Tip: run with --json for structured error output." in output


def test_run_squad_returns_machine_readable_json_error(capsys):
    """Squad JSON mode should keep machine-readable error payloads."""
    with patch(
        "copex.squad_team.SquadTeam.load_squad_file",
        side_effect=RuntimeError("broken squad state"),
    ):
        with pytest.raises(typer.Exit):
            asyncio.run(
                cli_squad._run_squad(
                    CopexConfig(),
                    "prompt",
                    json_output=True,
                )
            )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "success": False,
        "error": "broken squad state",
        "agents": [],
    }
