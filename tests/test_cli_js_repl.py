from __future__ import annotations

from typer.testing import CliRunner

import copex.cli as cli


def test_chat_js_repl_flags_set_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _fake_run_chat(config, prompt, *_args, **_kwargs):
        captured["js_repl"] = config.js_repl
        captured["js_repl_node_path"] = config.js_repl_node_path
        captured["prompt"] = prompt

    monkeypatch.setattr(cli, "_run_chat", _fake_run_chat)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "chat",
            "--model",
            "gpt-4.1",
            "--js-repl",
            "--js-repl-node",
            "/custom/node",
            "Run a JS check",
        ],
    )

    assert result.exit_code == 0
    assert captured["js_repl"] is True
    assert captured["js_repl_node_path"] == "/custom/node"
    assert captured["prompt"] == "Run a JS check"


def test_main_prompt_js_repl_flags_set_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _fake_run_chat(config, prompt, *_args, **_kwargs):
        captured["js_repl"] = config.js_repl
        captured["js_repl_node_path"] = config.js_repl_node_path
        captured["prompt"] = prompt

    monkeypatch.setattr(cli, "_run_chat", _fake_run_chat)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "--model",
            "gpt-4.1",
            "-p",
            "Review the snippet",
            "--no-squad",
            "--js-repl",
            "--js-repl-node",
            "/custom/node",
        ],
    )

    assert result.exit_code == 0
    assert captured["js_repl"] is True
    assert captured["js_repl_node_path"] == "/custom/node"
    assert captured["prompt"] == "Review the snippet"


def test_main_interactive_forwards_feature_flags(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_interactive(*_args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "interactive", _fake_interactive)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["--js-repl", "--js-repl-node", "/custom/node", "--pdf-analyze"],
    )

    assert result.exit_code == 0
    assert captured["js_repl"] is True
    assert captured["js_repl_node"] == "/custom/node"
    assert captured["pdf_analyze"] is True
