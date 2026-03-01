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
