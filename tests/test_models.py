from __future__ import annotations

import pytest

from copex import models as models_mod
from copex.config import COPILOT_CLI_NOT_FOUND_MESSAGE


def test_discover_models_raises_clear_error_on_cli_lookup_type_error(monkeypatch) -> None:
    monkeypatch.setattr(models_mod, "_discovered_models", None)

    def _raise_type_error() -> str:
        raise TypeError("bad cli lookup")

    monkeypatch.setattr("copex.config.find_copilot_cli", _raise_type_error)

    with pytest.raises(RuntimeError, match=COPILOT_CLI_NOT_FOUND_MESSAGE):
        models_mod.discover_models()


def test_discover_models_falls_back_to_enum_when_cli_missing(monkeypatch) -> None:
    monkeypatch.setattr(models_mod, "_discovered_models", None)
    monkeypatch.setattr("copex.config.find_copilot_cli", lambda: None)

    discovered = models_mod.discover_models()

    assert "gpt-5.3-codex" in discovered
