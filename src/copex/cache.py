"""
Step Caching - Cache step results to skip unchanged steps on repeated runs.

Caches are keyed by a hash of:
- Step description
- Step inputs (dependencies)
- Relevant file contents (if applicable)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "copex" / "steps"


@dataclass
class CacheEntry:
    """A cached step result."""
    
    step_hash: str
    result: str
    created_at: float
    expires_at: float | None  # None = never expires
    metadata: dict[str, Any]
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(**data)


class StepCache:
    """Cache for step execution results.
    
    Usage:
        cache = StepCache()
        
        # Check cache before executing
        step_hash = cache.compute_hash(step_description, inputs)
        if cached := cache.get(step_hash):
            print(f"Cache hit: {cached.result}")
        else:
            result = execute_step()
            cache.set(step_hash, result)
    """
    
    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        default_ttl: float | None = None,  # TTL in seconds, None = forever
        max_entries: int = 1000,
        enabled: bool = True,
    ) -> None:
        """Initialize the step cache.
        
        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time-to-live in seconds (None = no expiry)
            max_entries: Maximum number of cache entries
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.default_ttl = default_ttl
        self.max_entries = max_entries
        self.enabled = enabled
        
        # Create cache directory
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory index for fast lookups
        self._index: dict[str, CacheEntry] = {}
        self._load_index()
    
    def _load_index(self) -> None:
        """Load cache index from disk."""
        if not self.enabled:
            return
        
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file) as f:
                    data = json.load(f)
                for key, entry_data in data.items():
                    entry = CacheEntry.from_dict(entry_data)
                    if not entry.is_expired():
                        self._index[key] = entry
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load cache index: {e}")
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        if not self.enabled:
            return
        
        index_file = self.cache_dir / "index.json"
        try:
            data = {key: entry.to_dict() for key, entry in self._index.items()}
            with open(index_file, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def compute_hash(
        self,
        description: str,
        inputs: dict[str, Any] | None = None,
        file_paths: list[str | Path] | None = None,
    ) -> str:
        """Compute a hash key for a step.
        
        Args:
            description: Step description/prompt
            inputs: Input parameters for the step
            file_paths: Paths to files whose contents should affect the hash
        
        Returns:
            A hash string uniquely identifying this step configuration
        """
        hasher = hashlib.sha256()
        
        # Add description
        hasher.update(description.encode("utf-8"))
        
        # Add inputs (sorted for consistency)
        if inputs:
            inputs_str = json.dumps(inputs, sort_keys=True, default=str)
            hasher.update(inputs_str.encode("utf-8"))
        
        # Add file contents
        if file_paths:
            for path in sorted(str(p) for p in file_paths):
                try:
                    content = Path(path).read_bytes()
                    hasher.update(content)
                except OSError:
                    # File doesn't exist or can't be read - include path only
                    hasher.update(f"missing:{path}".encode("utf-8"))
        
        return hasher.hexdigest()[:16]  # Truncate for readability
    
    def get(self, step_hash: str) -> CacheEntry | None:
        """Get a cached entry by hash.
        
        Args:
            step_hash: The step hash to look up
        
        Returns:
            CacheEntry if found and not expired, None otherwise
        """
        if not self.enabled:
            return None
        
        entry = self._index.get(step_hash)
        if entry is None:
            return None
        
        if entry.is_expired():
            self.delete(step_hash)
            return None
        
        logger.debug(f"Cache hit for {step_hash}")
        return entry
    
    def set(
        self,
        step_hash: str,
        result: str,
        *,
        ttl: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        """Cache a step result.
        
        Args:
            step_hash: The step hash key
            result: The result to cache
            ttl: Time-to-live in seconds (overrides default)
            metadata: Optional metadata to store with the entry
        
        Returns:
            The created cache entry
        """
        if not self.enabled:
            return CacheEntry(
                step_hash=step_hash,
                result=result,
                created_at=time.time(),
                expires_at=None,
                metadata=metadata or {},
            )
        
        # Enforce max entries
        self._enforce_max_entries()
        
        # Compute expiration
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + effective_ttl if effective_ttl else None
        
        entry = CacheEntry(
            step_hash=step_hash,
            result=result,
            created_at=time.time(),
            expires_at=expires_at,
            metadata=metadata or {},
        )
        
        self._index[step_hash] = entry
        self._save_index()
        
        logger.debug(f"Cached result for {step_hash}")
        return entry
    
    def delete(self, step_hash: str) -> bool:
        """Delete a cache entry.
        
        Args:
            step_hash: The step hash to delete
        
        Returns:
            True if the entry was deleted, False if not found
        """
        if step_hash in self._index:
            del self._index[step_hash]
            self._save_index()
            return True
        return False
    
    def clear(self) -> int:
        """Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = len(self._index)
        self._index.clear()
        self._save_index()
        return count
    
    def _enforce_max_entries(self) -> None:
        """Remove oldest entries if cache exceeds max size."""
        if len(self._index) < self.max_entries:
            return
        
        # Sort by creation time and remove oldest
        sorted_entries = sorted(
            self._index.items(),
            key=lambda x: x[1].created_at,
        )
        
        to_remove = len(self._index) - self.max_entries + 1
        for key, _ in sorted_entries[:to_remove]:
            del self._index[key]
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        expired = sum(1 for e in self._index.values() if e.is_expired())
        return {
            "total_entries": len(self._index),
            "expired_entries": expired,
            "active_entries": len(self._index) - expired,
            "cache_dir": str(self.cache_dir),
            "enabled": self.enabled,
            "max_entries": self.max_entries,
            "default_ttl": self.default_ttl,
        }


# Global cache instance (can be configured)
_global_cache: StepCache | None = None


def get_cache(*, no_cache: bool = False, **kwargs: Any) -> StepCache:
    """Get the global step cache instance.
    
    Args:
        no_cache: If True, returns a disabled cache
        **kwargs: Arguments passed to StepCache if creating new instance
    
    Returns:
        The global StepCache instance
    """
    global _global_cache
    
    if no_cache:
        return StepCache(enabled=False)
    
    if _global_cache is None:
        _global_cache = StepCache(**kwargs)
    
    return _global_cache


def clear_global_cache() -> int:
    """Clear the global cache.
    
    Returns:
        Number of entries cleared
    """
    global _global_cache
    if _global_cache:
        return _global_cache.clear()
    return 0
