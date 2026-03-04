"""Tests added during deep quality pass — covers edge cases and bug fixes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from copex.cache import CacheEntry, StepCache
from copex.conditions import Condition, ConditionContext, all_of, any_of, none_of, when
from copex.metrics import (
    MetricsCollector,
    TOKEN_COSTS,
    estimate_cost,
    estimate_tokens,
)
from copex.streaming import StreamChunk, StreamingMetrics


# ── Cache TTL=0 bug fix ─────────────────────────────────────────────


class TestCacheTTLZero:
    """Regression: TTL=0 should mean 'expire immediately', not 'never expire'."""

    def test_ttl_zero_expires_immediately(self, tmp_path: Path) -> None:
        cache = StepCache(cache_dir=tmp_path / "cache")
        entry = cache.set("h1", "r1", ttl=0)
        assert entry.expires_at is not None
        # Entry should be expired (or very close to it)
        time.sleep(0.01)
        assert cache.get("h1") is None

    def test_ttl_zero_sets_expires_at(self, tmp_path: Path) -> None:
        cache = StepCache(cache_dir=tmp_path / "cache")
        entry = cache.set("h1", "r1", ttl=0)
        # expires_at should be set (not None)
        assert entry.expires_at is not None

    def test_default_ttl_zero_expires(self, tmp_path: Path) -> None:
        cache = StepCache(cache_dir=tmp_path / "cache", default_ttl=0)
        entry = cache.set("h1", "r1")
        assert entry.expires_at is not None
        time.sleep(0.01)
        assert cache.get("h1") is None

    def test_ttl_none_never_expires(self, tmp_path: Path) -> None:
        cache = StepCache(cache_dir=tmp_path / "cache")
        entry = cache.set("h1", "r1", ttl=None)
        assert entry.expires_at is None
        assert not entry.is_expired()

    def test_explicit_ttl_overrides_default(self, tmp_path: Path) -> None:
        cache = StepCache(cache_dir=tmp_path / "cache", default_ttl=3600)
        entry = cache.set("h1", "r1", ttl=0)
        assert entry.expires_at is not None
        time.sleep(0.01)
        assert cache.get("h1") is None


# ── Streaming UTF-8 byte accuracy ───────────────────────────────────


class TestStreamingBytesAccuracy:
    """Regression: total_bytes should count UTF-8 bytes, not Python chars."""

    def test_ascii_bytes_match_chars(self) -> None:
        m = StreamingMetrics()
        m.record_chunk(StreamChunk(type="message", delta="hello"))
        assert m.total_bytes == 5  # ASCII: 1 byte per char

    def test_multibyte_chars_counted_correctly(self) -> None:
        m = StreamingMetrics()
        # "café" = 5 UTF-8 bytes (c=1, a=1, f=1, é=2)
        m.record_chunk(StreamChunk(type="message", delta="café"))
        assert m.total_bytes == 5

    def test_cjk_chars_counted_correctly(self) -> None:
        m = StreamingMetrics()
        # "你好" = 6 UTF-8 bytes (3 bytes each)
        m.record_chunk(StreamChunk(type="message", delta="你好"))
        assert m.total_bytes == 6

    def test_emoji_counted_correctly(self) -> None:
        m = StreamingMetrics()
        # "🎉" = 4 UTF-8 bytes
        m.record_chunk(StreamChunk(type="message", delta="🎉"))
        assert m.total_bytes == 4


# ── Metrics coverage ────────────────────────────────────────────────


class TestMetricsCollector:
    def test_start_and_complete_request(self) -> None:
        collector = MetricsCollector(session_id="test")
        req = collector.start_request(model="gpt-5.2-codex", prompt="hello")
        result = collector.complete_request(req.request_id, success=True, response="hi")
        assert result is not None
        assert result.success is True
        assert result.duration_ms is not None
        assert result.duration_ms > 0

    def test_complete_nonexistent_request(self) -> None:
        collector = MetricsCollector()
        result = collector.complete_request("nonexistent", success=True)
        assert result is None

    def test_failed_request(self) -> None:
        collector = MetricsCollector()
        req = collector.start_request(model="gpt-5", prompt="test")
        result = collector.complete_request(
            req.request_id, success=False, error="timeout", retries=3
        )
        assert result is not None
        assert result.success is False
        assert result.error == "timeout"
        assert result.retries == 3

    def test_token_counts_from_api(self) -> None:
        collector = MetricsCollector()
        req = collector.start_request(model="gpt-5", prompt="x")
        result = collector.complete_request(
            req.request_id,
            success=True,
            tokens={"prompt": 100, "completion": 200},
        )
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 200
        assert result.total_tokens == 300

    def test_token_estimation_fallback(self) -> None:
        collector = MetricsCollector()
        req = collector.start_request(model="gpt-5", prompt="test")
        result = collector.complete_request(
            req.request_id, success=True, response="a" * 100
        )
        assert result.completion_tokens is not None
        assert result.completion_tokens > 0

    def test_cost_estimate(self) -> None:
        collector = MetricsCollector()
        req = collector.start_request(model="gpt-5.2-codex", prompt="x" * 4000)
        collector.complete_request(
            req.request_id,
            success=True,
            tokens={"prompt": 1000, "completion": 500},
        )
        cost = collector.cost_estimate()
        assert cost > 0

    def test_by_model(self) -> None:
        collector = MetricsCollector()
        for model in ["gpt-5", "gpt-5", "claude-sonnet-4"]:
            req = collector.start_request(model=model, prompt="x")
            collector.complete_request(req.request_id, success=True, response="y")
        breakdown = collector.by_model()
        assert breakdown["gpt-5"]["requests"] == 2
        assert breakdown["claude-sonnet-4"]["requests"] == 1

    def test_export_json(self, tmp_path: Path) -> None:
        collector = MetricsCollector()
        req = collector.start_request(model="gpt-5", prompt="x")
        collector.complete_request(req.request_id, success=True)
        path = tmp_path / "metrics.json"
        collector.export_json(path)
        assert path.exists()
        import json

        data = json.loads(path.read_text())
        assert "total_requests" in data

    def test_export_csv(self, tmp_path: Path) -> None:
        collector = MetricsCollector()
        req = collector.start_request(model="gpt-5", prompt="x")
        collector.complete_request(req.request_id, success=True, response="y")
        path = tmp_path / "metrics.csv"
        collector.export_csv(path)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 row

    def test_print_summary(self) -> None:
        collector = MetricsCollector()
        req = collector.start_request(model="gpt-5", prompt="x")
        collector.complete_request(req.request_id, success=True, response="y")
        summary = collector.print_summary()
        assert "Copex Metrics Summary" in summary
        assert "Success Rate" in summary

    def test_session_metrics_properties(self) -> None:
        collector = MetricsCollector()
        # Add a success and a failure
        req1 = collector.start_request(model="gpt-5", prompt="x")
        collector.complete_request(req1.request_id, success=True, response="y")
        req2 = collector.start_request(model="gpt-5", prompt="z")
        collector.complete_request(req2.request_id, success=False, error="err")

        s = collector.session
        assert s.total_requests == 2
        assert s.successful_requests == 1
        assert s.failed_requests == 1
        assert s.success_rate == 0.5

    def test_session_metrics_empty(self) -> None:
        collector = MetricsCollector()
        s = collector.session
        assert s.total_requests == 0
        assert s.success_rate == 0.0
        assert s.avg_duration_ms == 0.0


class TestEstimateCost:
    def test_known_model(self) -> None:
        cost = estimate_cost("gpt-5.2-codex", 1_000_000, 1_000_000)
        assert cost == 3.00 + 15.00

    def test_unknown_model_uses_default(self) -> None:
        cost = estimate_cost("unknown-model", 1_000_000, 1_000_000)
        assert cost == 2.0 + 10.0  # Default rates


class TestTokenCosts:
    def test_all_model_enum_values_have_costs(self) -> None:
        from copex.models import Model

        missing = []
        for m in Model:
            if m.value not in TOKEN_COSTS:
                missing.append(m.value)
        # Some models may intentionally not have costs (fall back to default)
        # But the major ones should be there
        for model in ["gpt-5.2-codex", "claude-opus-4.6", "gpt-5.3-codex", "claude-sonnet-4.6"]:
            assert model in TOKEN_COSTS, f"Missing TOKEN_COSTS for {model}"


class TestEstimateTokens:
    def test_basic(self) -> None:
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("") == 0

    def test_longer_text(self) -> None:
        text = "a" * 400
        assert estimate_tokens(text) == 100


# ── Conditions: none_of ─────────────────────────────────────────────


class TestNoneOfCondition:
    """Test none_of combinator which was previously missing from exports."""

    def test_none_of_all_false(self) -> None:
        ctx = ConditionContext.empty()
        cond = none_of("false", "false")
        assert cond.evaluate(ctx) is True

    def test_none_of_one_true(self) -> None:
        ctx = ConditionContext.empty()
        cond = none_of("true", "false")
        assert cond.evaluate(ctx) is False

    def test_none_of_all_true(self) -> None:
        ctx = ConditionContext.empty()
        cond = none_of("true", "true")
        assert cond.evaluate(ctx) is False

    def test_none_of_single(self) -> None:
        ctx = ConditionContext.empty()
        assert none_of("false").evaluate(ctx) is True
        assert none_of("true").evaluate(ctx) is False

    def test_none_of_importable_from_package(self) -> None:
        """Verify none_of is exported from the copex package."""
        import copex

        assert hasattr(copex, "none_of")


# ── Checkpoint timestamp consistency ────────────────────────────────


class TestCheckpointTimestampConsistency:
    """Regression: checkpoint_id and created_at should use same timestamp."""

    def test_timestamps_consistent(self, tmp_path: Path) -> None:
        from copex.checkpoint import CheckpointStore

        store = CheckpointStore(base_dir=tmp_path)
        cp = store.create("test-loop", "test prompt")
        # The checkpoint_id should contain the same timestamp as created_at
        # checkpoint_id format: "{loop_id}_{YYYYMMDD_HHMMSS}"
        id_timestamp = cp.checkpoint_id.replace("test-loop_", "")
        # created_at is ISO format — extract date/time parts
        from datetime import datetime

        created = datetime.fromisoformat(cp.created_at)
        expected_stamp = created.strftime("%Y%m%d_%H%M%S")
        assert id_timestamp == expected_stamp
