from __future__ import annotations

import pytest
from typer.testing import CliRunner

from copex.cli import app


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

    assert result.exit_code != 0
    assert "--reasoning" in result.output
    assert "invalid" in result.output.lower()
