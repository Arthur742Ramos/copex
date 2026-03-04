from __future__ import annotations

from typer.testing import CliRunner

import copex.cli as cli
from copex.model_router import ModelRoute, PromptTaskType
from copex.models import ReasoningEffort


def test_chat_auto_routes_model_and_reasoning(monkeypatch) -> None:
    captured: dict[str, str] = {}

    async def _fake_run_chat(config, prompt, *_args, **_kwargs):
        captured["model"] = config.model.value
        captured["reasoning"] = config.reasoning_effort.value
        captured["prompt"] = prompt

    monkeypatch.setattr(cli.CopexConfig, "to_client_options", lambda self: {})
    monkeypatch.setattr(
        cli,
        "route_model_for_prompt",
        lambda *_args, **_kwargs: ModelRoute(
            task_type=PromptTaskType.CODING,
            model="gpt-5.3-codex",
            reasoning_effort=ReasoningEffort.XHIGH,
        ),
    )
    monkeypatch.setattr(cli, "_run_chat", _fake_run_chat)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["chat", "Fix tests in src/api.py"])

    assert result.exit_code == 0
    assert captured["model"] == "gpt-5.3-codex"
    assert captured["reasoning"] == "xhigh"


def test_agent_auto_routes_model_and_reasoning(monkeypatch) -> None:
    captured: dict[str, str] = {}

    async def _fake_run_agent(config, _prompt, *, max_turns, json_output):  # noqa: ARG001
        captured["model"] = config.model.value
        captured["reasoning"] = config.reasoning_effort.value

    monkeypatch.setattr(cli.CopexConfig, "to_client_options", lambda self: {})
    monkeypatch.setattr(
        cli,
        "route_model_for_prompt",
        lambda *_args, **_kwargs: ModelRoute(
            task_type=PromptTaskType.CODING,
            model="gpt-5.3-codex",
            reasoning_effort=ReasoningEffort.XHIGH,
        ),
    )
    monkeypatch.setattr(cli, "_run_agent", _fake_run_agent)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["agent", "Refactor this module"])

    assert result.exit_code == 0
    assert captured["model"] == "gpt-5.3-codex"
    assert captured["reasoning"] == "xhigh"


def test_ralph_auto_routes_model_and_reasoning(monkeypatch) -> None:
    captured: dict[str, str] = {}

    async def _fake_run_ralph(config, _prompt, _max_iterations, _completion_promise):
        captured["model"] = config.model.value
        captured["reasoning"] = config.reasoning_effort.value

    monkeypatch.setattr(
        cli,
        "route_model_for_prompt",
        lambda *_args, **_kwargs: ModelRoute(
            task_type=PromptTaskType.CODING,
            model="gpt-5.3-codex",
            reasoning_effort=ReasoningEffort.XHIGH,
        ),
    )
    monkeypatch.setattr(cli, "_run_ralph", _fake_run_ralph)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["ralph", "Implement feature X"])

    assert result.exit_code == 0
    assert captured["model"] == "gpt-5.3-codex"
    assert captured["reasoning"] == "xhigh"


def test_main_prompt_auto_routes_when_model_unspecified(monkeypatch) -> None:
    captured: dict[str, str] = {}

    async def _fake_run_chat(config, prompt, *_args, **_kwargs):
        captured["model"] = config.model.value
        captured["reasoning"] = config.reasoning_effort.value
        captured["prompt"] = prompt

    monkeypatch.setattr(cli.CopexConfig, "to_client_options", lambda self: {})
    monkeypatch.setattr(
        cli,
        "route_model_for_prompt",
        lambda *_args, **_kwargs: ModelRoute(
            task_type=PromptTaskType.CODING,
            model="gpt-5.3-codex",
            reasoning_effort=ReasoningEffort.XHIGH,
        ),
    )
    monkeypatch.setattr(cli, "_run_chat", _fake_run_chat)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["-p", "Fix parser bug", "--no-squad"])

    assert result.exit_code == 0
    assert captured["model"] == "gpt-5.3-codex"
    assert captured["reasoning"] == "xhigh"
