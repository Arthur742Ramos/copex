"""Tests for copex.cache — StepCache, CacheEntry, and global cache functions."""

from __future__ import annotations

import json
import time

import pytest

from copex.cache import CacheEntry, StepCache, clear_global_cache, get_cache


# ── CacheEntry ───────────────────────────────────────────────────────


class TestCacheEntry:
    def test_not_expired_when_no_expiry(self):
        entry = CacheEntry(
            step_hash="abc",
            result="ok",
            created_at=time.time(),
            expires_at=None,
            metadata={},
        )
        assert not entry.is_expired()

    def test_not_expired_when_future(self):
        entry = CacheEntry(
            step_hash="abc",
            result="ok",
            created_at=time.time(),
            expires_at=time.time() + 3600,
            metadata={},
        )
        assert not entry.is_expired()

    def test_expired_when_past(self):
        entry = CacheEntry(
            step_hash="abc",
            result="ok",
            created_at=time.time() - 100,
            expires_at=time.time() - 1,
            metadata={},
        )
        assert entry.is_expired()

    def test_to_dict_roundtrip(self):
        entry = CacheEntry(
            step_hash="abc",
            result="ok",
            created_at=1234.0,
            expires_at=5678.0,
            metadata={"key": "val"},
        )
        d = entry.to_dict()
        restored = CacheEntry.from_dict(d)
        assert restored.step_hash == entry.step_hash
        assert restored.result == entry.result
        assert restored.metadata == entry.metadata


# ── StepCache ────────────────────────────────────────────────────────


class TestStepCache:
    def test_disabled_cache_returns_none(self, tmp_path):
        cache = StepCache(cache_dir=tmp_path / "cache", enabled=False)
        assert cache.get("any") is None

    def test_set_and_get(self, tmp_path):
        cache = StepCache(cache_dir=tmp_path / "cache")
        entry = cache.set("hash1", "result1", metadata={"step": 1})
        assert entry.result == "result1"
        assert cache.get("hash1") is not None
        assert cache.get("hash1").result == "result1"

    def test_get_nonexistent(self, tmp_path):
        cache = StepCache(cache_dir=tmp_path / "cache")
        assert cache.get("nonexistent") is None

    def test_delete(self, tmp_path):
        cache = StepCache(cache_dir=tmp_path / "cache")
        cache.set("hash1", "result1")
        assert cache.delete("hash1")
        assert cache.get("hash1") is None
        assert not cache.delete("hash1")  # Already deleted

    def test_clear(self, tmp_path):
        cache = StepCache(cache_dir=tmp_path / "cache")
        cache.set("h1", "r1")
        cache.set("h2", "r2")
        cleared = cache.clear()
        assert cleared == 2
        assert cache.get("h1") is None

    def test_ttl_expiry(self, tmp_path):
        cache = StepCache(cache_dir=tmp_path / "cache")
        cache.set("hash1", "result1", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("hash1") is None

    def test_default_ttl(self, tmp_path):
        cache = StepCache(cache_dir=tmp_path / "cache", default_ttl=0.01)
        cache.set("hash1", "result1")
        time.sleep(0.02)
        assert cache.get("hash1") is None

    def test_max_entries_enforced(self, tmp_path):
        cache = StepCache(cache_dir=tmp_path / "cache", max_entries=3)
        cache.set("h1", "r1")
        cache.set("h2", "r2")
        cache.set("h3", "r3")
        cache.set("h4", "r4")  # Should evict oldest
        stats = cache.stats()
        assert stats["total_entries"] <= 3

    def test_compute_hash_deterministic(self, tmp_path):
        cache = StepCache(cache_dir=tmp_path / "cache")
        h1 = cache.compute_hash("do something", {"a": 1})
        h2 = cache.compute_hash("do something", {"a": 1})
        assert h1 == h2

    def test_compute_hash_different_inputs(self, tmp_path):
        cache = StepCache(cache_dir=tmp_path / "cache")
        h1 = cache.compute_hash("do something", {"a": 1})
        h2 = cache.compute_hash("do something", {"a": 2})
        assert h1 != h2

    def test_compute_hash_with_files(self, tmp_path):
        cache = StepCache(cache_dir=tmp_path / "cache")
        f1 = tmp_path / "a.txt"
        f1.write_text("hello")
        h1 = cache.compute_hash("step", file_paths=[str(f1)])
        f1.write_text("world")
        h2 = cache.compute_hash("step", file_paths=[str(f1)])
        assert h1 != h2

    def test_compute_hash_missing_file(self, tmp_path):
        cache = StepCache(cache_dir=tmp_path / "cache")
        h = cache.compute_hash("step", file_paths=["/nonexistent/file.txt"])
        assert isinstance(h, str)
        assert len(h) == 16

    def test_stats(self, tmp_path):
        cache = StepCache(cache_dir=tmp_path / "cache")
        cache.set("h1", "r1")
        stats = cache.stats()
        assert stats["total_entries"] == 1
        assert stats["enabled"] is True
        assert stats["expired_entries"] == 0

    def test_set_when_disabled(self, tmp_path):
        cache = StepCache(cache_dir=tmp_path / "cache", enabled=False)
        entry = cache.set("h1", "r1")
        assert entry.result == "r1"
        # Not actually stored
        assert cache.get("h1") is None

    def test_persistence_across_instances(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache1 = StepCache(cache_dir=cache_dir)
        cache1.set("h1", "r1")

        cache2 = StepCache(cache_dir=cache_dir)
        assert cache2.get("h1") is not None
        assert cache2.get("h1").result == "r1"

    def test_corrupt_index_handled(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "index.json").write_text("{bad json")
        cache = StepCache(cache_dir=cache_dir)
        assert cache.get("any") is None


# ── Global cache ─────────────────────────────────────────────────────


class TestGlobalCache:
    def test_get_cache_no_cache(self):
        cache = get_cache(no_cache=True)
        assert not cache.enabled

    def test_clear_global_cache_when_none(self):
        import copex.cache as cache_mod
        old = cache_mod._global_cache
        cache_mod._global_cache = None
        assert clear_global_cache() == 0
        cache_mod._global_cache = old
