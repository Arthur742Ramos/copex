from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from copex.cli import app

_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


@pytest.mark.parametrize(
    "argv",
    [
        ["chat", "--reasoning", "invalid", "hello"],
        ["plan", "--reasoning", "invalid", "build api"],
        ["fleet", "--reasoning", "invalid", "task one"],
        ["squad", "--reasoning", "invalid", "do thing"],
    ],
)
def test_reasoning_option_rejects_invalid_values(argv: list[str]) -> None:
    runner = CliRunner()
    result = runner.invoke(app, argv)
    plain_output = _ANSI_ESCAPE_RE.sub("", result.output)

    assert result.exit_code != 0
    assert "--reasoning" in plain_output
    assert "invalid" in plain_output.lower()
