from __future__ import annotations

import pytest

from copex.context import (
    DEFAULT_CONTEXT_BUDGET_TOKENS,
    ConversationTurn,
    SmartContextWindow,
    TokenCounter,
    TurnTokenUsage,
    resolve_model_context_budget,
    smart_boundary_cutoff,
)


def test_resolve_model_context_budget_defaults() -> None:
    assert resolve_model_context_budget("gpt-4.1") == 128_000
    assert resolve_model_context_budget("claude-sonnet-4.5") == 200_000
    assert resolve_model_context_budget("openai/gpt-5") == 200_000
    assert resolve_model_context_budget("anthropic/claude-opus-4.6-1m") == 1_000_000
    assert resolve_model_context_budget("unknown-model") == DEFAULT_CONTEXT_BUDGET_TOKENS


def test_resolve_model_context_budget_override() -> None:
    assert resolve_model_context_budget("gpt-4.1", override=42) == 42
    assert resolve_model_context_budget("gpt-4.1", override=0) == 1


def test_conversation_turn_tracks_token_usage() -> None:
    turn = ConversationTurn(
        index=1,
        user_prompt="hello",
        assistant_response="world",
        user_tokens=12,
        assistant_tokens=8,
    )

    assert turn.token_usage == TurnTokenUsage(user_tokens=12, assistant_tokens=8)
    assert turn.total_tokens == 20


def test_token_counter_uses_fast_approximation_when_encoder_unavailable() -> None:
    counter = TokenCounter("gpt-4.1")
    counter._encoder = None

    assert counter.count("") == 0
    assert counter.count("abcd") == 1
    assert counter.count("abcdefghi") == 3


@pytest.mark.asyncio
async def test_prepare_prompt_triggers_summarization_near_budget_limit() -> None:
    manager = SmartContextWindow("gpt-4.1", context_budget=25_120, recent_turns=2)
    long_user = " ".join(["user-context"] * 90)
    long_assistant = " ".join(["assistant-context"] * 90)

    for i in range(4):
        manager.record_turn(f"{long_user} {i}", f"{long_assistant} {i}")

    calls: list[str] = []

    async def summarize(prompt: str) -> str:
        calls.append(prompt)
        return "- compact summary"

    prepared = await manager.prepare_prompt("Continue from the latest state.", summarize)

    assert calls
    assert prepared.summary_updated is True
    assert prepared.reset_session is True
    assert "Compacted conversation summary" in prepared.prompt
    assert manager.summary.startswith("- compact summary")


def test_smart_boundary_keeps_error_and_fix_together() -> None:
    turns = [
        ConversationTurn(1, "Set up parser", "Done", 10, 10),
        ConversationTurn(2, "Traceback: ValueError", "Investigating failure", 10, 10),
        ConversationTurn(3, "Apply fix", "Fixed by input normalization", 10, 10),
    ]

    # Boundary between error and fix should move forward to keep both together.
    assert smart_boundary_cutoff(turns, 2) == 3


def test_smart_boundary_keeps_code_fences_balanced() -> None:
    turns = [
        ConversationTurn(1, "Show code:\n```python\nprint('hi')", "Continuing...", 10, 10),
        ConversationTurn(2, "Close block", "```\nDone", 10, 10),
    ]

    # Boundary after the first turn would leave an unclosed fence.
    assert smart_boundary_cutoff(turns, 1) == 2


def test_smart_boundary_advances_until_fences_are_balanced() -> None:
    turns = [
        ConversationTurn(1, "Start block\n```python\nx = 1", "Still open", 10, 10),
        ConversationTurn(2, "Discuss output", "No fence yet", 10, 10),
        ConversationTurn(3, "Wrap up", "```\nDone", 10, 10),
    ]

    assert smart_boundary_cutoff(turns, 1) == 3
