from __future__ import annotations

import asyncio

import pytest

from copex import model_router
from copex.model_router import (
    PromptTaskType,
    detect_task_type,
    get_runtime_models,
    route_model_for_prompt,
)


@pytest.fixture(autouse=True)
def _reset_task_type_cache() -> None:
    with model_router._task_type_cache_lock:
        model_router._task_type_cache.clear()


@pytest.fixture(autouse=True)
def _disable_llm_classifier(monkeypatch) -> None:
    async def _no_llm(_prompt: str, client_options=None):
        return None

    monkeypatch.setattr(model_router, "_classify_task_type_with_llm", _no_llm)


def test_detect_task_type_coding_prompt() -> None:
    task_type = detect_task_type("Fix failing test in src/auth.py and update API endpoint.")
    assert task_type == PromptTaskType.CODING


def test_detect_task_type_general_prompt() -> None:
    task_type = detect_task_type("Write a creative story about deterministic machines.")
    assert task_type == PromptTaskType.GENERAL


def test_route_model_for_coding_prefers_codex() -> None:
    route = route_model_for_prompt(
        "Refactor this Python function and add pytest coverage.",
        available_models=["claude-opus-4.6-1m", "gpt-5.3-codex"],
    )

    assert route.task_type == PromptTaskType.CODING
    assert route.model == "gpt-5.3-codex"
    assert route.reasoning_effort.value == "xhigh"


def test_route_model_for_general_prefers_opus_1m() -> None:
    route = route_model_for_prompt(
        "Draft a narrative chapter with a reflective tone.",
        available_models=["gpt-5.3-codex", "claude-opus-4.6-1m"],
    )

    assert route.task_type == PromptTaskType.GENERAL
    assert route.model == "claude-opus-4.6-1m"
    assert route.reasoning_effort.value in {"high", "none"}


def test_get_runtime_models_falls_back_to_known_models(monkeypatch) -> None:
    async def _fail(_client_options):
        raise RuntimeError("boom")

    monkeypatch.setattr("copex.model_router._query_runtime_models", _fail)

    models = get_runtime_models(refresh=True)

    assert "gpt-5.3-codex" in models


def test_detect_task_type_uses_llm_for_nuanced_coding_prompt(monkeypatch) -> None:
    async def _classify(_prompt: str, client_options=None):
        return PromptTaskType.CODING

    monkeypatch.setattr(model_router, "_classify_task_type_with_llm", _classify)

    task_type = detect_task_type("Make this code more elegant while preserving behavior.")

    assert task_type == PromptTaskType.CODING


@pytest.mark.asyncio
async def test_detect_task_type_uses_llm_inside_running_loop(monkeypatch) -> None:
    async def _classify(_prompt: str, client_options=None):
        return PromptTaskType.CODING

    monkeypatch.setattr(model_router, "_classify_task_type_with_llm", _classify)
    monkeypatch.setattr(
        model_router,
        "_detect_task_type_with_regex",
        lambda _prompt: PromptTaskType.GENERAL,
    )

    assert detect_task_type("Make this code more elegant while preserving behavior.") == (
        PromptTaskType.CODING
    )


def test_detect_task_type_falls_back_to_regex_on_llm_failure(monkeypatch) -> None:
    async def _fail(_prompt: str, client_options=None):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(model_router, "_classify_task_type_with_llm", _fail)

    task_type = detect_task_type("Fix failing test in src/auth.py")

    assert task_type == PromptTaskType.CODING


def test_detect_task_type_caches_identical_prompts(monkeypatch) -> None:
    calls = 0

    async def _classify(_prompt: str, client_options=None):
        nonlocal calls
        calls += 1
        return PromptTaskType.GENERAL

    monkeypatch.setattr(model_router, "_classify_task_type_with_llm", _classify)

    prompt = "Write a poem about Python."
    assert detect_task_type(prompt) == PromptTaskType.GENERAL
    assert detect_task_type(prompt) == PromptTaskType.GENERAL
    assert calls == 1


def test_route_model_for_prompt_forwards_client_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _detect(_prompt: str, *, client_options=None):
        captured["client_options"] = client_options
        return PromptTaskType.GENERAL

    monkeypatch.setattr(model_router, "detect_task_type", _detect)

    route_model_for_prompt(
        "Draft a narrative chapter with a reflective tone.",
        client_options={"cwd": "/tmp/project"},
        available_models=["claude-opus-4.6-1m"],
    )

    assert captured["client_options"] == {"cwd": "/tmp/project"}
