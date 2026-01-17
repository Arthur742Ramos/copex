"""Model and configuration enums."""

from enum import Enum


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
    GEMINI_3_PRO = "gemini-3-pro-preview"


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for supported models."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class EventType(str, Enum):
    """Copilot session event types."""

    ASSISTANT_MESSAGE = "assistant.message"
    ASSISTANT_MESSAGE_DELTA = "assistant.message_delta"
    ASSISTANT_REASONING = "assistant.reasoning"
    ASSISTANT_REASONING_DELTA = "assistant.reasoning_delta"
    SESSION_IDLE = "session.idle"
    ERROR = "error"
    TOOL_CALL = "tool.call"
