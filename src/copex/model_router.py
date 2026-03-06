"""Runtime prompt classification and model routing."""

from __future__ import annotations

import asyncio
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any

from copex.models import (
    ReasoningEffort,
    get_available_models,
    normalize_reasoning_effort,
)

_CODING_PATTERNS = (
    re.compile(r"```[\w+-]*\n", re.IGNORECASE),
    re.compile(r"\b(fix|bug|debug|refactor|compile|stack trace|traceback)\b", re.IGNORECASE),
    re.compile(r"\b(test|pytest|unit test|integration test|failing test)\b", re.IGNORECASE),
    re.compile(r"\b(function|class|method|module|api|endpoint|schema|json-rpc)\b", re.IGNORECASE),
    re.compile(r"\b(python|typescript|javascript|rust|go|java|c\\+\\+|shell)\b", re.IGNORECASE),
    re.compile(r"\b(src/|tests?/|\\.py\\b|\\.ts\\b|\\.js\\b|\\.go\\b|\\.rs\\b)\b", re.IGNORECASE),
)

_CODING_MODEL_PREFERENCES = [
    "gpt-5.4",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex",
    "gpt-5.2",
    "gpt-5.1",
    "claude-sonnet-4.6",
]

_GENERAL_MODEL_PREFERENCES = [
    "claude-opus-4.6-1m",
    "claude-opus-4.6",
    "claude-sonnet-4.6",
    "gpt-5.4",
    "gpt-5.3-codex",
    "gpt-5.2",
]

_MODEL_CACHE_TTL_SECONDS = 120.0
_model_cache: list[str] | None = None
_model_cache_refreshed_at = 0.0
_model_cache_lock = threading.Lock()

_TASK_TYPE_CLASSIFIER_MODEL = "claude-opus-4.6-1m"
_TASK_TYPE_CLASSIFIER_TIMEOUT_SECONDS = 4.0
_TASK_TYPE_CACHE_MAX_ENTRIES = 256
_TASK_TYPE_CLASSIFIER_INSTRUCTIONS = (
    "Classify the user's request as exactly one label: "
    "'coding' or 'general/creative'. Reply with only that label."
)
_task_type_cache: OrderedDict[str, PromptTaskType] = OrderedDict()
_task_type_cache_lock = threading.Lock()


class PromptTaskType(str, Enum):
    CODING = "coding"
    GENERAL = "general"


@dataclass(frozen=True)
class ModelRoute:
    task_type: PromptTaskType
    model: str
    reasoning_effort: ReasoningEffort
    warning: str | None = None


def detect_task_type(
    prompt: str, *, client_options: dict[str, Any] | None = None
) -> PromptTaskType:
    """Classify prompt as coding vs general/creative."""

    text = prompt.strip()
    if not text:
        return PromptTaskType.GENERAL
    with _task_type_cache_lock:
        cached = _task_type_cache.get(text)
        if cached is not None:
            _task_type_cache.move_to_end(text)
    if cached is not None:
        return cached

    classified: PromptTaskType | None = None
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is None or not running_loop.is_running():
        try:
            classified = asyncio.run(
                asyncio.wait_for(
                    _classify_task_type_with_llm(text, client_options=client_options),
                    timeout=_TASK_TYPE_CLASSIFIER_TIMEOUT_SECONDS,
                )
            )
        except Exception:
            classified = None

    task_type = classified or _detect_task_type_with_regex(text)
    with _task_type_cache_lock:
        if text not in _task_type_cache and len(_task_type_cache) >= _TASK_TYPE_CACHE_MAX_ENTRIES:
            _task_type_cache.popitem(last=False)
        _task_type_cache[text] = task_type
    return task_type


def _detect_task_type_with_regex(prompt: str) -> PromptTaskType:
    if any(pattern.search(prompt) for pattern in _CODING_PATTERNS):
        return PromptTaskType.CODING
    return PromptTaskType.GENERAL


def _parse_task_type_label(value: str) -> PromptTaskType | None:
    label = value.strip().lower()
    if not label:
        return None
    first = label.splitlines()[0].strip().strip("`'\" \t\r\n.,:;")
    first = re.sub(r"\s*/\s*", "/", first)
    if first == PromptTaskType.CODING.value:
        return PromptTaskType.CODING
    if first in {
        PromptTaskType.GENERAL.value,
        "creative",
        "general/creative",
    }:
        return PromptTaskType.GENERAL
    return None


def _extract_classifier_label(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        message_type = getattr(message, "type", None)
        message_value = message_type.value if hasattr(message_type, "value") else str(message_type)
        if message_value != "assistant.message":
            continue
        content = getattr(getattr(message, "data", None), "content", None)
        if isinstance(content, str) and content.strip():
            return content
    for message in reversed(messages):
        content = getattr(getattr(message, "data", None), "content", None)
        if isinstance(content, str) and content.strip():
            return content
    return None


async def _classify_task_type_with_llm(
    prompt: str, client_options: dict[str, Any] | None = None
) -> PromptTaskType | None:
    from copilot import CopilotClient

    client = CopilotClient(client_options or None)
    session: Any | None = None
    await client.start()
    try:
        session = await client.create_session(
            {
                "model": _TASK_TYPE_CLASSIFIER_MODEL,
                "streaming": False,
                "instructions": _TASK_TYPE_CLASSIFIER_INSTRUCTIONS,
            }
        )
        await session.send({"prompt": prompt})
        messages = await session.get_messages()
        label = _extract_classifier_label(messages)
        if not label:
            return None
        return _parse_task_type_label(label)
    finally:
        if session is not None:
            try:
                await session.destroy()
            except Exception:
                pass
        try:
            await client.stop()
        except Exception:
            pass


async def _query_runtime_models(client_options: dict[str, Any] | None) -> list[str]:
    from copilot import CopilotClient

    client = CopilotClient(client_options or None)
    await client.start()
    try:
        models = await asyncio.wait_for(client.list_models(), timeout=8.0)
        model_ids = [str(model.id) for model in models if model.id]
        return model_ids
    finally:
        try:
            await client.stop()
        except Exception:
            pass


def get_runtime_models(
    *,
    client_options: dict[str, Any] | None = None,
    refresh: bool = False,
) -> list[str]:
    """Return runtime model IDs from SDK model discovery with process-level caching."""

    global _model_cache, _model_cache_refreshed_at

    now = time.monotonic()
    with _model_cache_lock:
        if (
            not refresh
            and _model_cache is not None
            and now - _model_cache_refreshed_at < _MODEL_CACHE_TTL_SECONDS
        ):
            return list(_model_cache)

    try:
        # Avoid asyncio.run() inside existing event loops.
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    discovered: list[str]
    if running_loop and running_loop.is_running():
        discovered = []
    else:
        try:
            discovered = asyncio.run(_query_runtime_models(client_options))
        except Exception:
            discovered = []

    if not discovered:
        discovered = get_available_models()

    with _model_cache_lock:
        _model_cache = list(dict.fromkeys(discovered))
        _model_cache_refreshed_at = time.monotonic()
        return list(_model_cache)


def _pick_model(task_type: PromptTaskType, available_models: list[str]) -> str:
    candidates = (
        _CODING_MODEL_PREFERENCES
        if task_type == PromptTaskType.CODING
        else _GENERAL_MODEL_PREFERENCES
    )
    available = set(available_models)
    for candidate in candidates:
        if candidate in available:
            return candidate
    if available_models:
        return available_models[0]
    return candidates[0]


def route_model_for_prompt(
    prompt: str,
    *,
    client_options: dict[str, Any] | None = None,
    available_models: list[str] | None = None,
) -> ModelRoute:
    """Route to a model/reasoning profile based on prompt type and runtime models."""

    task_type = detect_task_type(prompt, client_options=client_options)
    discovered = available_models or get_runtime_models(client_options=client_options)
    selected = _pick_model(task_type, discovered)
    target_reasoning = (
        ReasoningEffort.XHIGH if task_type == PromptTaskType.CODING else ReasoningEffort.HIGH
    )
    normalized, warning = normalize_reasoning_effort(selected, target_reasoning)
    return ModelRoute(
        task_type=task_type,
        model=selected,
        reasoning_effort=normalized,
        warning=warning,
    )


def option_was_explicit(option_name: str) -> bool:
    """Return True when a CLI option was set explicitly instead of defaulted."""

    try:
        import click
        from click.core import ParameterSource
    except Exception:
        return False

    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return False
    try:
        source = ctx.get_parameter_source(option_name)
    except Exception:
        return False
    return source not in (None, ParameterSource.DEFAULT)
