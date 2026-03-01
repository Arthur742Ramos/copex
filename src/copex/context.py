"""Smart context window management for long-running conversations."""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

_ERROR_RE = re.compile(
    r"\b(error|exception|traceback|failed|failure|stack trace|panic|invalid)\b",
    re.IGNORECASE,
)
_FIX_RE = re.compile(
    r"\b(fix|fixed|resolved|solution|patched|workaround|mitigate|corrected)\b",
    re.IGNORECASE,
)

_SYSTEM_RESERVED_TOKENS = 5_000
_REPO_MAP_RESERVED_TOKENS = 10_000
_MEMORY_RESERVED_TOKENS = 5_000
_TOOLS_RESERVED_TOKENS = 5_000


def resolve_model_context_budget(model: str, override: int | None = None) -> int:
    """Resolve total context budget for a model family."""
    if override is not None:
        return max(1, int(override))

    model_id = model.lower()
    if "claude-opus-4.6-1m" in model_id:
        return 1_000_000
    if model_id.startswith("claude"):
        return 200_000
    if model_id.startswith("gpt-4"):
        return 128_000
    if model_id.startswith("gpt-5"):
        return 200_000
    if model_id.startswith("gpt"):
        return 128_000
    if model_id.startswith("gemini"):
        return 1_000_000
    return 128_000


class TokenCounter:
    """Fast token counting with optional tiktoken support."""

    def __init__(self, model: str) -> None:
        self._encoder = None
        try:
            import tiktoken  # type: ignore

            try:
                self._encoder = tiktoken.encoding_for_model(model)
            except Exception:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        ascii_chars = sum(1 for ch in text if ord(ch) < 128)
        non_ascii_chars = len(text) - ascii_chars
        approx = (ascii_chars / 4.0) + (non_ascii_chars / 2.0) + (text.count("\n") * 0.25)
        return max(1, int(math.ceil(approx)))


@dataclass
class ConversationTurn:
    """A user+assistant exchange tracked by the context manager."""

    index: int
    user_prompt: str
    assistant_response: str
    user_tokens: int
    assistant_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.user_tokens + self.assistant_tokens

    @property
    def text(self) -> str:
        return f"User:\n{self.user_prompt}\n\nAssistant:\n{self.assistant_response}"

    @property
    def code_fence_count(self) -> int:
        return self.user_prompt.count("```") + self.assistant_response.count("```")

    @property
    def has_error_signature(self) -> bool:
        return bool(_ERROR_RE.search(self.user_prompt) or _ERROR_RE.search(self.assistant_response))

    @property
    def has_fix_signature(self) -> bool:
        return bool(_FIX_RE.search(self.user_prompt) or _FIX_RE.search(self.assistant_response))


@dataclass(frozen=True)
class ContextBudget:
    """Budget split for smart context management."""

    total_tokens: int
    system_tokens: int = _SYSTEM_RESERVED_TOKENS
    repo_map_tokens: int = _REPO_MAP_RESERVED_TOKENS
    memory_tokens: int = _MEMORY_RESERVED_TOKENS
    tools_tokens: int = _TOOLS_RESERVED_TOKENS

    @property
    def reserved_tokens(self) -> int:
        return self.system_tokens + self.repo_map_tokens + self.memory_tokens + self.tools_tokens

    @property
    def working_tokens(self) -> int:
        return max(1, self.total_tokens - self.reserved_tokens)


@dataclass(frozen=True)
class ContextUsage:
    """Current working-context usage snapshot."""

    used_tokens: int
    budget_tokens: int
    reserved_tokens: int
    summary_tokens: int
    recent_turn_tokens: int


@dataclass(frozen=True)
class PreparedContext:
    """Context preparation result for the next outbound prompt."""

    prompt: str
    reset_session: bool
    summary_updated: bool
    usage: ContextUsage


def smart_boundary_cutoff(turns: Sequence[ConversationTurn], cutoff: int) -> int:
    """Adjust compaction cutoff to avoid splitting logical boundaries."""
    if cutoff <= 0:
        return 0
    if cutoff >= len(turns):
        cutoff = len(turns)

    adjusted = cutoff
    if 0 < adjusted < len(turns):
        left = turns[adjusted - 1]
        right = turns[adjusted]

        # Keep error report + fix together.
        if left.has_error_signature and right.has_fix_signature:
            adjusted += 1
        elif right.has_error_signature and adjusted + 1 < len(turns):
            after = turns[adjusted + 1]
            if after.has_fix_signature:
                adjusted += 2

    # Keep code fences intact across the summarized chunk.
    while adjusted < len(turns):
        fence_count = sum(t.code_fence_count for t in turns[:adjusted])
        if fence_count % 2 == 0:
            break
        adjusted += 1

    return min(adjusted, len(turns))


class SmartContextWindow:
    """Tracks turn history and compacts old context near budget limits."""

    def __init__(
        self,
        model: str,
        *,
        context_budget: int | None = None,
        recent_turns: int = 6,
    ) -> None:
        self._model = model
        self._counter = TokenCounter(model)
        self._budget = ContextBudget(total_tokens=resolve_model_context_budget(model, context_budget))
        self._recent_turns = max(1, recent_turns)
        self._turns: list[ConversationTurn] = []
        self._summary = ""
        self._summary_tokens = 0
        self._turn_index = 0
        self._needs_reseed = False

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def budget(self) -> ContextBudget:
        return self._budget

    def configure(self, model: str, context_budget: int | None = None) -> None:
        """Update model/budget settings and refresh token estimates."""
        total_tokens = resolve_model_context_budget(model, context_budget)
        if model == self._model and total_tokens == self._budget.total_tokens:
            return

        self._model = model
        self._counter = TokenCounter(model)
        self._budget = ContextBudget(total_tokens=total_tokens)
        self._summary_tokens = self._counter.count(self._summary)

        refreshed: list[ConversationTurn] = []
        for turn in self._turns:
            refreshed.append(
                ConversationTurn(
                    index=turn.index,
                    user_prompt=turn.user_prompt,
                    assistant_response=turn.assistant_response,
                    user_tokens=self._counter.count(turn.user_prompt),
                    assistant_tokens=self._counter.count(turn.assistant_response),
                )
            )
        self._turns = refreshed

    def clear(self) -> None:
        """Reset tracked context for a fresh conversation."""
        self._turns.clear()
        self._summary = ""
        self._summary_tokens = 0
        self._turn_index = 0
        self._needs_reseed = False

    def record_turn(self, user_prompt: str, assistant_response: str) -> None:
        """Record a completed turn after a successful response."""
        self._turn_index += 1
        turn = ConversationTurn(
            index=self._turn_index,
            user_prompt=user_prompt,
            assistant_response=assistant_response,
            user_tokens=self._counter.count(user_prompt),
            assistant_tokens=self._counter.count(assistant_response),
        )
        self._turns.append(turn)
        self._needs_reseed = False

    def usage(self, next_prompt: str | None = None) -> ContextUsage:
        """Return current working-context usage."""
        turns_tokens = sum(turn.total_tokens for turn in self._turns)
        next_tokens = self._counter.count(next_prompt or "")
        used = self._summary_tokens + turns_tokens + next_tokens
        return ContextUsage(
            used_tokens=used,
            budget_tokens=self._budget.working_tokens,
            reserved_tokens=self._budget.reserved_tokens,
            summary_tokens=self._summary_tokens,
            recent_turn_tokens=turns_tokens,
        )

    async def prepare_prompt(
        self,
        prompt: str,
        summarize: Callable[[str], Awaitable[str]],
    ) -> PreparedContext:
        """Prepare a prompt, compacting old turns when needed."""
        summary_updated = False
        if self.usage(next_prompt=prompt).used_tokens > self._budget.working_tokens:
            summary_updated = await self._compact_until_within_budget(prompt, summarize)

        usage = self.usage(next_prompt=prompt)
        if self._needs_reseed:
            return PreparedContext(
                prompt=self._build_reseed_prompt(prompt),
                reset_session=True,
                summary_updated=summary_updated,
                usage=usage,
            )
        return PreparedContext(
            prompt=prompt,
            reset_session=False,
            summary_updated=summary_updated,
            usage=usage,
        )

    async def _compact_until_within_budget(
        self,
        next_prompt: str,
        summarize: Callable[[str], Awaitable[str]],
    ) -> bool:
        changed = False
        guard = 0

        while (
            self.usage(next_prompt=next_prompt).used_tokens > self._budget.working_tokens
            and len(self._turns) > self._recent_turns
        ):
            guard += 1
            if guard > 8:
                break

            max_compactable = len(self._turns) - self._recent_turns
            candidates = self._turns[:max_compactable]
            raw_cutoff = max(1, len(candidates) // 2)
            cutoff = smart_boundary_cutoff(candidates, raw_cutoff)
            chunk = candidates[:cutoff]
            if not chunk:
                break

            target_tokens = max(256, self._budget.working_tokens // 5)
            summary_prompt = self._build_summary_prompt(chunk, target_tokens)
            new_summary = (await summarize(summary_prompt)).strip()
            if not new_summary:
                new_summary = self._fallback_summary(chunk)
            new_summary = self._truncate_to_budget(new_summary, target_tokens)
            self._summary = new_summary
            self._summary_tokens = self._counter.count(self._summary)
            self._turns = self._turns[cutoff:]
            self._needs_reseed = True
            changed = True

        return changed

    def _build_summary_prompt(self, turns: Sequence[ConversationTurn], target_tokens: int) -> str:
        turns_text = "\n\n".join(self._format_turn(turn) for turn in turns)
        previous_summary = self._summary.strip() or "(none)"
        return (
            "You are compacting long-running conversation history for a coding assistant.\n"
            f"Target size: <= {target_tokens} tokens.\n\n"
            "Preserve:\n"
            "- key technical decisions and constraints\n"
            "- file paths, commands, APIs, and code snippets that matter\n"
            "- open TODOs and unresolved questions\n"
            "- error messages with their fixes grouped together\n\n"
            "Output concise markdown bullets only.\n\n"
            "Previous summary:\n"
            f"{previous_summary}\n\n"
            "New turns to merge:\n"
            f"{turns_text}"
        )

    def _build_reseed_prompt(self, prompt: str) -> str:
        recent = self._turns[-self._recent_turns :]
        parts = [
            "Continue this coding conversation using the compacted context below.",
            "Use the context as prior history and answer only the current user request.",
        ]
        if self._summary:
            parts.append("### Compacted conversation summary")
            parts.append(self._summary)
        if recent:
            parts.append("### Most recent turns (verbatim)")
            for turn in recent:
                parts.append(self._format_turn(turn))
        parts.append("### Current user prompt")
        parts.append(prompt)
        return "\n\n".join(parts)

    @staticmethod
    def _format_turn(turn: ConversationTurn) -> str:
        return (
            f"[Turn {turn.index}]\n"
            f"User:\n{turn.user_prompt}\n\n"
            f"Assistant:\n{turn.assistant_response}"
        )

    @staticmethod
    def _fallback_summary(turns: Sequence[ConversationTurn]) -> str:
        lines = []
        for turn in turns:
            user = " ".join(turn.user_prompt.strip().split())[:180]
            assistant = " ".join(turn.assistant_response.strip().split())[:220]
            lines.append(f"- Turn {turn.index}: user={user!r}; assistant={assistant!r}")
        return "\n".join(lines)

    def _truncate_to_budget(self, summary: str, max_tokens: int) -> str:
        current = self._counter.count(summary)
        if current <= max_tokens:
            return summary
        ratio = max_tokens / max(current, 1)
        keep_chars = max(64, int(len(summary) * ratio))
        return summary[:keep_chars].rstrip()
