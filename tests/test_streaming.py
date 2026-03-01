"""Tests for copex.streaming — StreamChunk, StreamingMetrics, ChunkBatcher, Response."""

from __future__ import annotations

import time

import pytest

from copex.streaming import ChunkBatcher, Response, StreamChunk, StreamingMetrics


# ── StreamChunk ──────────────────────────────────────────────────────

class TestStreamChunk:
    def test_defaults(self):
        chunk = StreamChunk(type="message")
        assert chunk.delta == ""
        assert chunk.is_final is False
        assert chunk.tool_id is None

    def test_tool_call_fields(self):
        chunk = StreamChunk(
            type="tool_call",
            tool_id="t1",
            tool_name="run",
            tool_args={"cmd": "ls"},
        )
        assert chunk.tool_name == "run"
        assert chunk.tool_args == {"cmd": "ls"}

    def test_tool_result_fields(self):
        chunk = StreamChunk(
            type="tool_result",
            tool_id="t1",
            tool_result="ok",
            tool_success=True,
            tool_duration=1.5,
        )
        assert chunk.tool_success is True
        assert chunk.tool_duration == 1.5


# ── StreamingMetrics ─────────────────────────────────────────────────

class TestStreamingMetrics:
    def test_initial_state(self):
        m = StreamingMetrics()
        assert m.total_chunks == 0
        assert m.time_to_first_chunk_ms is None
        assert m.chunks_per_second == 0.0
        assert m.throughput_bytes_per_second == 0.0

    def test_record_message_chunk(self):
        m = StreamingMetrics(_start_time=time.monotonic())
        m.record_chunk(StreamChunk(type="message", delta="hello"))
        assert m.total_chunks == 1
        assert m.message_chunks == 1
        assert m.total_bytes == 5
        assert m.first_chunk_time is not None

    def test_record_reasoning_chunk(self):
        m = StreamingMetrics()
        m.record_chunk(StreamChunk(type="reasoning", delta="think"))
        assert m.reasoning_chunks == 1

    def test_record_tool_chunk(self):
        m = StreamingMetrics()
        m.record_chunk(StreamChunk(type="tool_call", delta=""))
        assert m.tool_chunks == 1

    def test_record_tool_result_chunk(self):
        m = StreamingMetrics()
        m.record_chunk(StreamChunk(type="tool_result", delta=""))
        assert m.tool_chunks == 1

    def test_time_to_first_chunk(self):
        now = time.monotonic()
        m = StreamingMetrics(_start_time=now)
        m.first_chunk_time = now + 0.1
        ttfc = m.time_to_first_chunk_ms
        assert ttfc is not None
        assert abs(ttfc - 100.0) < 5.0

    def test_time_to_first_chunk_no_start(self):
        m = StreamingMetrics()
        m.first_chunk_time = time.monotonic()
        assert m.time_to_first_chunk_ms is None

    def test_chunks_per_second_single_chunk(self):
        now = time.monotonic()
        m = StreamingMetrics(first_chunk_time=now, last_chunk_time=now, total_chunks=1)
        # elapsed=0 → returns 0.0 (rate is undefined for zero elapsed time)
        assert m.chunks_per_second == 0.0

    def test_chunks_per_second_multiple(self):
        now = time.monotonic()
        m = StreamingMetrics(
            first_chunk_time=now, last_chunk_time=now + 2.0, total_chunks=10
        )
        assert abs(m.chunks_per_second - 5.0) < 0.01

    def test_throughput_bytes_zero_elapsed(self):
        now = time.monotonic()
        m = StreamingMetrics(
            first_chunk_time=now, last_chunk_time=now, total_bytes=100
        )
        # elapsed=0 → returns 0.0 (rate is undefined for zero elapsed time)
        assert m.throughput_bytes_per_second == 0.0

    def test_throughput_bytes_nonzero(self):
        now = time.monotonic()
        m = StreamingMetrics(
            first_chunk_time=now, last_chunk_time=now + 1.0, total_bytes=1000
        )
        assert abs(m.throughput_bytes_per_second - 1000.0) < 0.01


# ── ChunkBatcher ─────────────────────────────────────────────────────

class TestChunkBatcher:
    def test_non_delta_passthrough(self):
        received = []
        batcher = ChunkBatcher(callback=received.append)
        chunk = StreamChunk(type="tool_call", tool_id="t1", tool_name="run")
        batcher.push(chunk)
        assert len(received) == 1
        assert received[0] is chunk

    def test_final_chunk_passthrough(self):
        received = []
        batcher = ChunkBatcher(callback=received.append)
        chunk = StreamChunk(type="message", delta="done", is_final=True)
        batcher.push(chunk)
        assert len(received) == 1
        assert received[0] is chunk

    def test_batches_same_type_deltas(self):
        received = []
        batcher = ChunkBatcher(callback=received.append)
        batcher.push(StreamChunk(type="message", delta="hel"))
        batcher.push(StreamChunk(type="message", delta="lo"))
        assert len(received) == 0  # Not flushed yet
        batcher.flush()
        assert len(received) == 1
        assert received[0].delta == "hello"
        assert received[0].type == "message"

    def test_type_change_flushes(self):
        received = []
        batcher = ChunkBatcher(callback=received.append)
        batcher.push(StreamChunk(type="message", delta="msg"))
        batcher.push(StreamChunk(type="reasoning", delta="think"))
        # Type changed → first batch flushed
        assert len(received) == 1
        assert received[0].delta == "msg"
        batcher.flush()
        assert len(received) == 2
        assert received[1].delta == "think"

    def test_max_bytes_flushes(self):
        received = []
        batcher = ChunkBatcher(callback=received.append, max_bytes=10)
        batcher.push(StreamChunk(type="message", delta="12345"))
        batcher.push(StreamChunk(type="message", delta="67890"))
        # 10 bytes reached → should auto-flush
        assert len(received) == 1
        assert received[0].delta == "1234567890"

    def test_flush_noop_when_empty(self):
        received = []
        batcher = ChunkBatcher(callback=received.append)
        batcher.flush()
        assert len(received) == 0


# ── Response ─────────────────────────────────────────────────────────

class TestResponse:
    def test_usage_none_when_no_tokens(self):
        r = Response(content="hi")
        assert r.usage is None

    def test_usage_with_tokens(self):
        r = Response(content="hi", prompt_tokens=10, completion_tokens=20)
        assert r.usage == {"prompt_tokens": 10, "completion_tokens": 20}

    def test_usage_partial_tokens(self):
        r = Response(content="hi", prompt_tokens=5)
        assert r.usage == {"prompt_tokens": 5, "completion_tokens": 0}

    def test_defaults(self):
        r = Response(content="test")
        assert r.reasoning is None
        assert r.retries == 0
        assert r.auto_continues == 0
        assert r.raw_events == []

    def test_server_model_and_cost(self):
        r = Response(content="hi", server_model="gpt-5", cost=0.01)
        assert r.server_model == "gpt-5"
        assert r.cost == 0.01


# ── Edge Cases ───────────────────────────────────────────────────────


class TestStreamingEdgeCases:
    def test_record_chunk_empty_delta(self):
        """Empty string delta should not add bytes."""
        m = StreamingMetrics()
        m.record_chunk(StreamChunk(type="message", delta=""))
        assert m.total_chunks == 1
        assert m.total_bytes == 0

    def test_record_chunk_system_type(self):
        """System chunks should not count as message/reasoning/tool."""
        m = StreamingMetrics()
        m.record_chunk(StreamChunk(type="system", delta="info"))
        assert m.total_chunks == 1
        assert m.message_chunks == 0
        assert m.reasoning_chunks == 0
        assert m.tool_chunks == 0

    def test_batcher_empty_delta_not_batched(self):
        """Chunks with empty delta are passed through, not batched."""
        received = []
        batcher = ChunkBatcher(callback=received.append)
        chunk = StreamChunk(type="message", delta="")
        batcher.push(chunk)
        # Empty delta is not "is_delta", so passed through directly
        assert len(received) == 1

    def test_metrics_multiple_chunks_timing(self):
        """Verify first/last chunk times are tracked correctly."""
        m = StreamingMetrics(_start_time=time.monotonic())
        m.record_chunk(StreamChunk(type="message", delta="a"))
        first = m.first_chunk_time
        time.sleep(0.01)
        m.record_chunk(StreamChunk(type="message", delta="b"))
        assert m.first_chunk_time == first  # Should not change
        assert m.last_chunk_time > first
        assert m.total_chunks == 2

    def test_response_usage_both_none(self):
        """Both tokens None should yield None usage."""
        r = Response(content="x", prompt_tokens=None, completion_tokens=None)
        assert r.usage is None

    def test_chunks_per_second_no_first_chunk(self):
        """No first chunk should return 0."""
        m = StreamingMetrics(last_chunk_time=time.monotonic(), total_chunks=5)
        assert m.chunks_per_second == 0.0

    def test_chunks_per_second_no_last_chunk(self):
        """No last chunk should return 0."""
        m = StreamingMetrics(first_chunk_time=time.monotonic(), total_chunks=5)
        assert m.chunks_per_second == 0.0
