from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

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


class _FakeCapabilityClient:
    def __init__(self, *, models=None, error: Exception | None = None) -> None:
        self._models = models or []
        self._error = error
        self.stopped = False

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        self.stopped = True

    async def list_models(self):
        if self._error is not None:
            raise self._error
        return self._models


@pytest.mark.asyncio
async def test_refresh_model_capabilities_stops_client_on_success(monkeypatch) -> None:
    client = _FakeCapabilityClient(
        models=[
            SimpleNamespace(
                id="gpt-5.4",
                capabilities=SimpleNamespace(
                    supports=SimpleNamespace(reasoning_effort=True)
                ),
            )
        ]
    )
    fake_module = ModuleType("copilot")
    fake_module.CopilotClient = lambda: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "copilot", fake_module)
    monkeypatch.setattr(models_mod, "_reasoning_support", None)

    result = await models_mod.refresh_model_capabilities()

    assert result["gpt-5.4"] is True
    assert client.stopped is True


@pytest.mark.asyncio
async def test_refresh_model_capabilities_stops_client_on_failure(monkeypatch) -> None:
    client = _FakeCapabilityClient(error=RuntimeError("boom"))
    fake_module = ModuleType("copilot")
    fake_module.CopilotClient = lambda: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "copilot", fake_module)
    monkeypatch.setattr(models_mod, "_reasoning_support", None)

    result = await models_mod.refresh_model_capabilities()

    assert result == models_mod._fallback_reasoning_support()
    assert client.stopped is True
