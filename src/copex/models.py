"""Model and configuration enums.

This module also includes small helpers for validating combinations of model +
"reasoning effort". Not all models support all effort levels.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from enum import Enum

log = logging.getLogger(__name__)


class Model(str, Enum):
    """Available Copilot models."""

    GPT_5_2_CODEX = "gpt-5.2-codex"
    GPT_5_1_CODEX = "gpt-5.1-codex"
    GPT_5_1_CODEX_MAX = "gpt-5.1-codex-max"
    GPT_5_1_CODEX_MINI = "gpt-5.1-codex-mini"
    GPT_5_2 = "gpt-5.2"
    GPT_5_1 = "gpt-5.1"
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_4_1 = "gpt-4.1"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4.5"
    CLAUDE_SONNET_4 = "claude-sonnet-4"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4.5"
    CLAUDE_OPUS_4_5 = "claude-opus-4.5"
    CLAUDE_OPUS_4_6 = "claude-opus-4.6"
    CLAUDE_OPUS_4_6_FAST = "claude-opus-4.6-fast"
    GEMINI_3_PRO = "gemini-3-pro-preview"


# Models that do NOT support reasoning_effort configuration.
# Source: backend ``models.list`` → capabilities.supports.reasoning_effort
_NO_REASONING_MODELS: set[str] = {
    Model.CLAUDE_SONNET_4_5.value,
    Model.CLAUDE_SONNET_4.value,
    Model.CLAUDE_HAIKU_4_5.value,
    Model.CLAUDE_OPUS_4_5.value,
    Model.GEMINI_3_PRO.value,
    Model.GPT_4_1.value,
}


# Dynamic reasoning-support cache populated by ``refresh_model_capabilities()``.
_reasoning_support: dict[str, bool] | None = None


async def refresh_model_capabilities(*, timeout: float = 5.0) -> dict[str, bool]:
    """Query the Copilot backend for per-model reasoning support.

    Results are cached at module level.  On any failure the function falls back
    to the hardcoded ``_NO_REASONING_MODELS`` set and returns that mapping.
    """
    global _reasoning_support
    try:
        from copilot import CopilotClient  # type: ignore[import-untyped]

        client = CopilotClient()
        await client.start()
        models = await asyncio.wait_for(client.list_models(), timeout=timeout)
        result: dict[str, bool] = {}
        for m in models:
            supports = getattr(
                getattr(getattr(m, "capabilities", None), "supports", None),
                "reasoning_effort",
                None,
            )
            result[m.id] = bool(supports)
        _reasoning_support = result
        log.debug("Refreshed model capabilities: %d models", len(result))
        return result
    except Exception:
        log.debug("Backend model query failed; using hardcoded fallback", exc_info=True)
        return _fallback_reasoning_support()


def _fallback_reasoning_support() -> dict[str, bool]:
    """Build a reasoning-support dict from the hardcoded set."""
    result: dict[str, bool] = {}
    for m in Model:
        result[m.value] = m.value not in _NO_REASONING_MODELS
    return result


def model_supports_reasoning(model: Model | str) -> bool:
    """Return True if the model supports the ``reasoning_effort`` parameter.

    Uses dynamically discovered capabilities when available (populated by
    :func:`refresh_model_capabilities`), otherwise falls back to the hardcoded
    ``_NO_REASONING_MODELS`` set.
    """
    model_id = model.value if isinstance(model, Model) else str(model)
    if _reasoning_support is not None:
        if model_id in _reasoning_support:
            return _reasoning_support[model_id]
    return model_id not in _NO_REASONING_MODELS

# ---------------------------------------------------------------------------
# Dynamic model discovery
# ---------------------------------------------------------------------------

_discovered_models: list[str] | None = None

_MODEL_CHOICES_RE = re.compile(
    r'--model\b.*?\(choices:\s*(.+?)\)',
    re.IGNORECASE | re.DOTALL,
)


def discover_models() -> list[str]:
    """Discover available models by parsing ``copilot --help``.

    Results are cached for the lifetime of the process.  Falls back to the
    ``Model`` enum values when the CLI is unavailable.
    """
    global _discovered_models
    if _discovered_models is not None:
        return _discovered_models

    try:
        from copex.config import find_copilot_cli

        cli_path = find_copilot_cli()
        if cli_path is None:
            raise FileNotFoundError("copilot CLI not found")

        result = subprocess.run(
            [cli_path, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"copilot --help exited with code {result.returncode}")
        m = _MODEL_CHOICES_RE.search(result.stdout)
        if m:
            raw = m.group(1)
            models = [s.strip().strip('"').strip("'") for s in raw.split(",")]
            models = [s for s in models if s]
            if models:
                _discovered_models = models
                return _discovered_models
    except (OSError, subprocess.SubprocessError):
        pass

    _discovered_models = [model.value for model in Model]
    return _discovered_models


def resolve_model(model_str: str) -> str:
    """Resolve a model string to a valid model ID.

    Accepts ``Model`` enum values as well as any model ID returned by
    :func:`discover_models`.  Raises ``ValidationError`` for unknown models.
    """
    # Check enum values first
    enum_values = {m.value for m in Model}
    if model_str in enum_values:
        return model_str

    # Check dynamically discovered models
    if model_str in discover_models():
        return model_str

    from copex.exceptions import ValidationError

    raise ValidationError(
        f"Unknown model '{model_str}'. "
        f"Available models: {', '.join(get_available_models())}"
    )


def get_available_models() -> list[str]:
    """Return the union of enum values and dynamically discovered models."""
    enum_values = [m.value for m in Model]
    discovered = discover_models()
    seen: set[str] = set()
    merged: list[str] = []
    for model_id in [*enum_values, *discovered]:
        if model_id not in seen:
            seen.add(model_id)
            merged.append(model_id)
    return merged


def no_reasoning_models() -> set[str]:
    """Return the set of models that do NOT support reasoning_effort.

    Uses dynamically discovered capabilities when available, otherwise returns
    the hardcoded ``_NO_REASONING_MODELS`` set.
    """
    if _reasoning_support is not None:
        return {mid for mid, supports in _reasoning_support.items() if not supports}
    return set(_NO_REASONING_MODELS)


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for supported models."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


_GPT_VERSION_RE = re.compile(r"^gpt-(?P<major>\d+)(?:\.(?P<minor>\d+))?", re.IGNORECASE)


def parse_reasoning_effort(value: str | ReasoningEffort | None) -> ReasoningEffort | None:
    """Parse a reasoning effort string with small aliases.

    Supported aliases:
      - "xh" -> "xhigh"

    Returns None if value is None.
    """
    if value is None:
        return None
    if isinstance(value, ReasoningEffort):
        return value
    v = value.strip().lower()
    if v == "xh":
        v = "xhigh"
    return ReasoningEffort(v)


def model_supports_xhigh(model: Model | str) -> bool:
    """Return True if the model is in the GPT/Codex family that supports xhigh.

    Policy:
      - xhigh is allowed ONLY for GPT-5.2 (and higher GPT/Codex versions).
      - Everything else is capped at high.

    Notes:
      - We treat "gpt-5" (no minor) and "gpt-5-mini" as NOT supporting xhigh.
      - Any gpt major >= 6 supports xhigh.
      - gpt-5.<minor> supports xhigh only when minor >= 2.
    """
    model_id = model.value if isinstance(model, Model) else str(model)
    m = _GPT_VERSION_RE.match(model_id)
    if not m:
        return False

    major = int(m.group("major"))
    minor_str = m.group("minor")
    minor = int(minor_str) if minor_str is not None else None

    if major >= 6:
        return True
    if major < 5:
        return False

    # major == 5
    return minor is not None and minor >= 2


def normalize_reasoning_effort(
    model: Model | str,
    reasoning: ReasoningEffort | str,
    *,
    downgrade_to: ReasoningEffort = ReasoningEffort.HIGH,
) -> tuple[ReasoningEffort, str | None]:
    """Normalize a requested reasoning effort for a given model.

    If an unsupported effort is requested (currently: xhigh on non-supported
    models), we downgrade to "high" and return a warning message.
    Models in ``_NO_REASONING_MODELS`` are forced to ``NONE``.

    Returns:
      (normalized_effort, warning_message)
    """
    effort = parse_reasoning_effort(reasoning)
    if effort is None:
        return downgrade_to, None

    model_id = model.value if isinstance(model, Model) else str(model)

    # Models that don't support reasoning at all
    if not model_supports_reasoning(model_id):
        if effort != ReasoningEffort.NONE:
            return ReasoningEffort.NONE, (
                f"Model '{model_id}' does not support reasoning effort. "
                f"Setting to 'none'."
            )
        return ReasoningEffort.NONE, None

    if effort == ReasoningEffort.XHIGH and not model_supports_xhigh(model):
        return downgrade_to, (
            f"Model '{model_id}' does not support reasoning effort '{effort.value}'. "
            f"Downgrading to '{downgrade_to.value}'."
        )

    return effort, None


class EventType(str, Enum):
    """Copilot session event types."""

    USER_MESSAGE = "user.message"
    ASSISTANT_MESSAGE = "assistant.message"
    ASSISTANT_MESSAGE_DELTA = "assistant.message_delta"
    ASSISTANT_REASONING = "assistant.reasoning"
    ASSISTANT_REASONING_DELTA = "assistant.reasoning_delta"
    ASSISTANT_TURN_END = "assistant.turn_end"
    SESSION_IDLE = "session.idle"
    SESSION_ERROR = "session.error"
    ERROR = "error"
    TOOL_CALL = "tool.call"
    TOOL_EXECUTION_START = "tool.execution_start"
    TOOL_EXECUTION_PARTIAL_RESULT = "tool.execution_partial_result"
    TOOL_EXECUTION_COMPLETE = "tool.execution_complete"
