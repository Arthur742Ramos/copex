from __future__ import annotations

from copex.context import (
    DEFAULT_CONTEXT_BUDGET_TOKENS,
    ConversationTurn,
    TokenCounter,
    TurnTokenUsage,
    resolve_model_context_budget,
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
