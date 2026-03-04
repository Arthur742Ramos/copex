from __future__ import annotations

from typer.testing import CliRunner

import copex.cli as cli
import copex.cli_plan as cli_plan
import copex.cli_squad as cli_squad
from copex.model_router import ModelRoute, PromptTaskType
from copex.models import ReasoningEffort


def test_plan_command_auto_routes_model(monkeypatch) -> None:
    captured: dict[str, str] = {}

    async def _fake_run_plan(*, config, **_kwargs):
        captured["model"] = config.model.value
        captured["reasoning"] = config.reasoning_effort.value

    monkeypatch.setattr(
        cli_plan,
        "route_model_for_prompt",
        lambda *_args, **_kwargs: ModelRoute(
            task_type=PromptTaskType.CODING,
            model="gpt-5.3-codex",
            reasoning_effort=ReasoningEffort.XHIGH,
        ),
    )
    monkeypatch.setattr(cli_plan, "_run_plan", _fake_run_plan)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["plan", "Fix schema migration bug"])

    assert result.exit_code == 0
    assert captured["model"] == "gpt-5.3-codex"
    assert captured["reasoning"] == "xhigh"


def test_squad_command_auto_routes_model(monkeypatch) -> None:
    captured: dict[str, str] = {}

    async def _fake_run_squad(config, _prompt, **_kwargs):
        captured["model"] = config.model.value
        captured["reasoning"] = config.reasoning_effort.value

    monkeypatch.setattr(cli.CopexConfig, "to_client_options", lambda self: {})
    monkeypatch.setattr(
        cli_squad,
        "route_model_for_prompt",
        lambda *_args, **_kwargs: ModelRoute(
            task_type=PromptTaskType.CODING,
            model="gpt-5.3-codex",
            reasoning_effort=ReasoningEffort.XHIGH,
        ),
    )
    monkeypatch.setattr(cli_squad, "_run_squad", _fake_run_squad)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["squad", "Fix failing integration tests"])

    assert result.exit_code == 0
    assert captured["model"] == "gpt-5.3-codex"
    assert captured["reasoning"] == "xhigh"
