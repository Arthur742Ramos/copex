"""
Metrics and Logging - Track token usage, timing, and success rates.

Provides:
- Token usage tracking
- Timing metrics per request
- Success/failure rates
- Cost estimation
- Export to various formats
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import threading


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    request_id: str
    timestamp: str
    model: str
    reasoning_effort: str

    # Timing
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None

    # Tokens (estimated or from API)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # Status
    success: bool = False
    error: str | None = None
    retries: int = 0

    # Content
    prompt_preview: str = ""
    response_preview: str = ""

    def finalize(self, end_time: float | None = None) -> None:
        """Finalize timing metrics."""
        self.end_time = end_time or time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000


@dataclass
class SessionMetrics:
    """Aggregated metrics for a session."""

    session_id: str
    started_at: str
    requests: list[RequestMetrics] = field(default_factory=list)

    @property
    def total_requests(self) -> int:
        return len(self.requests)

    @property
    def successful_requests(self) -> int:
        return sum(1 for r in self.requests if r.success)

    @property
    def failed_requests(self) -> int:
        return sum(1 for r in self.requests if not r.success)

    @property
    def success_rate(self) -> float:
        if not self.requests:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def total_retries(self) -> int:
        return sum(r.retries for r in self.requests)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens or 0 for r in self.requests)

    @property
    def total_duration_ms(self) -> float:
        return sum(r.duration_ms or 0 for r in self.requests)

    @property
    def avg_duration_ms(self) -> float:
        if not self.requests:
            return 0.0
        return self.total_duration_ms / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "total_retries": self.total_retries,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "requests": [
                {
                    "request_id": r.request_id,
                    "timestamp": r.timestamp,
                    "model": r.model,
                    "duration_ms": r.duration_ms,
                    "tokens": r.total_tokens,
                    "success": r.success,
                    "retries": r.retries,
                }
                for r in self.requests
            ],
        }


# Token cost estimates per 1M tokens (as of 2026)
TOKEN_COSTS = {
    "gpt-5.2-codex": {"input": 3.00, "output": 15.00},
    "gpt-5.1-codex": {"input": 2.50, "output": 10.00},
    "gpt-5.1-codex-max": {"input": 5.00, "output": 20.00},
    "gpt-5.1-codex-mini": {"input": 0.50, "output": 2.00},
    "gpt-5.2": {"input": 2.50, "output": 10.00},
    "gpt-5.1": {"input": 2.00, "output": 8.00},
    "gpt-5": {"input": 2.00, "output": 8.00},
    "gpt-5-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 0.30, "output": 1.20},
    "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-haiku-4.5": {"input": 0.80, "output": 4.00},
    "claude-opus-4.5": {"input": 15.00, "output": 75.00},
    "gemini-3-pro-preview": {"input": 1.25, "output": 5.00},
}


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token)."""
    return len(text) // 4


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """
    Estimate cost in USD.

    Args:
        model: Model name
        prompt_tokens: Input tokens
        completion_tokens: Output tokens

    Returns:
        Estimated cost in USD
    """
    costs = TOKEN_COSTS.get(model, {"input": 2.0, "output": 10.0})
    input_cost = (prompt_tokens / 1_000_000) * costs["input"]
    output_cost = (completion_tokens / 1_000_000) * costs["output"]
    return input_cost + output_cost


class MetricsCollector:
    """
    Collects and manages metrics for Copex requests.

    Usage:
        collector = MetricsCollector()

        # Start tracking a request
        req = collector.start_request(model="gpt-5.2-codex", prompt="Hello")

        # ... make request ...

        # Complete tracking
        collector.complete_request(req.request_id, success=True, response="Hi!")

        # Get summary
        print(collector.summary())
    """

    def __init__(self, session_id: str | None = None):
        """Initialize collector."""
        self.session = SessionMetrics(
            session_id=session_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            started_at=datetime.now().isoformat(),
        )
        self._pending: dict[str, RequestMetrics] = {}
        self._lock = threading.Lock()
        self._request_counter = 0

    def start_request(
        self,
        model: str,
        reasoning_effort: str = "xhigh",
        prompt: str = "",
    ) -> RequestMetrics:
        """
        Start tracking a new request.

        Returns:
            RequestMetrics object to track this request
        """
        with self._lock:
            self._request_counter += 1
            request_id = f"req_{self._request_counter}_{int(time.time() * 1000)}"

        metrics = RequestMetrics(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            model=model,
            reasoning_effort=reasoning_effort,
            start_time=time.time(),
            prompt_preview=prompt[:100] if prompt else "",
            prompt_tokens=estimate_tokens(prompt) if prompt else None,
        )

        with self._lock:
            self._pending[request_id] = metrics

        return metrics

    def complete_request(
        self,
        request_id: str,
        success: bool = True,
        response: str = "",
        error: str | None = None,
        retries: int = 0,
        tokens: dict[str, int] | None = None,
    ) -> RequestMetrics | None:
        """
        Complete tracking for a request.

        Args:
            request_id: Request ID from start_request
            success: Whether request succeeded
            response: Response content
            error: Error message if failed
            retries: Number of retries needed
            tokens: Optional token counts {"prompt": N, "completion": N}

        Returns:
            Completed RequestMetrics
        """
        with self._lock:
            metrics = self._pending.pop(request_id, None)

        if not metrics:
            return None

        metrics.finalize()
        metrics.success = success
        metrics.error = error
        metrics.retries = retries
        metrics.response_preview = response[:100] if response else ""

        # Token counts
        if tokens:
            metrics.prompt_tokens = tokens.get("prompt", metrics.prompt_tokens)
            metrics.completion_tokens = tokens.get("completion")
            metrics.total_tokens = (metrics.prompt_tokens or 0) + (metrics.completion_tokens or 0)
        elif response:
            metrics.completion_tokens = estimate_tokens(response)
            metrics.total_tokens = (metrics.prompt_tokens or 0) + metrics.completion_tokens

        with self._lock:
            self.session.requests.append(metrics)

        return metrics

    def summary(self) -> dict[str, Any]:
        """Get session summary."""
        return self.session.to_dict()

    def cost_estimate(self) -> float:
        """Estimate total cost in USD."""
        total = 0.0
        for req in self.session.requests:
            if req.prompt_tokens and req.completion_tokens:
                total += estimate_cost(req.model, req.prompt_tokens, req.completion_tokens)
        return total

    def by_model(self) -> dict[str, dict[str, Any]]:
        """Get metrics grouped by model."""
        by_model: dict[str, list[RequestMetrics]] = defaultdict(list)
        for req in self.session.requests:
            by_model[req.model].append(req)

        result = {}
        for model, requests in by_model.items():
            result[model] = {
                "requests": len(requests),
                "success_rate": sum(1 for r in requests if r.success) / len(requests),
                "total_tokens": sum(r.total_tokens or 0 for r in requests),
                "avg_duration_ms": sum(r.duration_ms or 0 for r in requests) / len(requests),
            }

        return result

    def export_json(self, path: Path | str) -> None:
        """Export metrics to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)

    def export_csv(self, path: Path | str) -> None:
        """Export metrics to CSV file."""
        import csv

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "request_id", "timestamp", "model", "reasoning_effort",
                "duration_ms", "prompt_tokens", "completion_tokens", "total_tokens",
                "success", "retries", "error",
            ])

            for req in self.session.requests:
                writer.writerow([
                    req.request_id, req.timestamp, req.model, req.reasoning_effort,
                    req.duration_ms, req.prompt_tokens, req.completion_tokens, req.total_tokens,
                    req.success, req.retries, req.error or "",
                ])

    def print_summary(self) -> str:
        """Get printable summary string."""
        s = self.session
        lines = [
            f"═══ Copex Metrics Summary ═══",
            f"Session: {s.session_id}",
            f"Started: {s.started_at}",
            f"",
            f"Requests: {s.total_requests} ({s.successful_requests} ok, {s.failed_requests} failed)",
            f"Success Rate: {s.success_rate:.1%}",
            f"Total Retries: {s.total_retries}",
            f"",
            f"Total Tokens: {s.total_tokens:,}",
            f"Estimated Cost: ${self.cost_estimate():.4f}",
            f"",
            f"Total Time: {s.total_duration_ms / 1000:.1f}s",
            f"Avg Time/Request: {s.avg_duration_ms:.0f}ms",
        ]

        # By model breakdown
        by_model = self.by_model()
        if len(by_model) > 1:
            lines.append("")
            lines.append("By Model:")
            for model, stats in by_model.items():
                lines.append(f"  {model}: {stats['requests']} requests, {stats['success_rate']:.0%} success")

        return "\n".join(lines)


# Global collector for convenience
_global_collector: MetricsCollector | None = None


def get_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_collector() -> None:
    """Reset global metrics collector."""
    global _global_collector
    _global_collector = None
