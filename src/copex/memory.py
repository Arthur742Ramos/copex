"""Persistent project memory for Copex."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MEMORY_DIR_NAME = ".copex"
MEMORY_FILE_NAME = "memory.md"
PREFERENCES_FILE_NAME = "preferences.toml"
MAX_MEMORY_SIZE_BYTES = 50 * 1024
MAX_PROMPT_CONTEXT_CHARS = 12_000
MAX_IMPORTED_FILE_CHARS = 6_000
MAX_ENTRY_CHARS = 320
MAX_CONTEXT_ENTRIES = 30

EXTERNAL_GUIDANCE_FILES = (
    "CLAUDE.md",
    ".cursorrules",
    ".aider.conf.yml",
    "AGENTS.md",
)

_ENTRY_LINE_RE = re.compile(r"^- \[(?P<timestamp>[^\]]+)\] \[(?P<kind>[^\]]+)\] (?P<text>.+)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_PREFERENCE_HINT_RE = re.compile(
    r"\b(prefer|preference|always|never|please|instead|don't|do not|must|should|avoid)\b",
    re.IGNORECASE,
)
_DECISION_HINT_RE = re.compile(
    r"\b(decision|decided|architecture|architectural|design choice|trade-?off|approach)\b",
    re.IGNORECASE,
)
_PATTERN_HINT_RE = re.compile(
    r"\b(pattern|convention|coding style|naming|format|type hints?|async|dataclass|testing)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MemoryEntry:
    """A single stored memory entry."""

    timestamp: str
    kind: str
    text: str


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _entry_key(kind: str, text: str) -> str:
    return f"{kind.lower()}::{_normalize_whitespace(text).lower()}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProjectMemory:
    """Manage persistent project memory stored in .copex/memory.md."""

    def __init__(self, root: Path | None = None, *, max_size_bytes: int = MAX_MEMORY_SIZE_BYTES):
        self.root = (root or Path.cwd()).resolve()
        self.max_size_bytes = max_size_bytes

    @property
    def memory_dir(self) -> Path:
        return self.root / MEMORY_DIR_NAME

    @property
    def memory_path(self) -> Path:
        return self.memory_dir / MEMORY_FILE_NAME

    @property
    def preferences_path(self) -> Path:
        return self.memory_dir / PREFERENCES_FILE_NAME

    def _ensure_memory_dir(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def read_memory(self) -> str:
        if not self.memory_path.is_file():
            return ""
        return self.memory_path.read_text(encoding="utf-8")

    def write_memory(self, content: str) -> None:
        self._ensure_memory_dir()
        self.memory_path.write_text(content, encoding="utf-8")

    def clear(self) -> None:
        self.write_memory(self._render_entries([]))

    def parse_entries(self) -> list[MemoryEntry]:
        text = self.read_memory()
        if not text:
            return []

        entries: list[MemoryEntry] = []
        for line in text.splitlines():
            match = _ENTRY_LINE_RE.match(line.strip())
            if not match:
                continue
            entries.append(
                MemoryEntry(
                    timestamp=match.group("timestamp").strip(),
                    kind=match.group("kind").strip().lower(),
                    text=match.group("text").strip(),
                )
            )
        return entries

    def _render_entries(
        self,
        entries: list[MemoryEntry],
        *,
        summary_lines: list[str] | None = None,
    ) -> str:
        lines = ["# Copex Project Memory", ""]
        if summary_lines:
            lines.extend(["## Summary", ""])
            lines.extend([f"- {line}" for line in summary_lines])
            lines.append("")
        lines.extend(["## Entries", ""])
        lines.extend(
            [f"- [{entry.timestamp}] [{entry.kind}] {entry.text}" for entry in entries]
        )
        lines.append("")
        return "\n".join(lines)

    def _deduplicate_entries(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        seen: set[str] = set()
        unique_reverse: list[MemoryEntry] = []
        for entry in reversed(entries):
            key = _entry_key(entry.kind, entry.text)
            if key in seen:
                continue
            seen.add(key)
            unique_reverse.append(entry)
        return list(reversed(unique_reverse))

    def _compact_if_needed(self) -> None:
        if not self.memory_path.is_file():
            return
        if self.memory_path.stat().st_size <= self.max_size_bytes:
            return

        entries = self.parse_entries()
        if not entries:
            legacy = _normalize_whitespace(self.read_memory())
            if legacy:
                entries = [MemoryEntry(_now_iso(), "summary", _clip(legacy, MAX_ENTRY_CHARS))]

        before_count = len(entries)
        entries = self._deduplicate_entries(entries)
        kind_counts: dict[str, int] = {}
        for entry in entries:
            kind_counts[entry.kind] = kind_counts.get(entry.kind, 0) + 1

        condensed = [
            MemoryEntry(entry.timestamp, entry.kind, _clip(entry.text, MAX_ENTRY_CHARS))
            for entry in entries
        ]
        condensed = condensed[-240:]

        summary = [
            f"Compacted at {_now_iso()}",
            f"Entries before compaction: {before_count}",
            f"Unique entries retained: {len(entries)}",
            "Entry types: "
            + ", ".join(f"{kind}={count}" for kind, count in sorted(kind_counts.items())),
        ]

        rendered = self._render_entries(condensed, summary_lines=summary)
        while len(rendered.encode("utf-8")) > self.max_size_bytes and len(condensed) > 20:
            condensed = condensed[len(condensed) // 4 :]
            rendered = self._render_entries(condensed, summary_lines=summary)

        if len(rendered.encode("utf-8")) > self.max_size_bytes:
            tighter = [
                MemoryEntry(entry.timestamp, entry.kind, _clip(entry.text, 180))
                for entry in condensed
            ]
            rendered = self._render_entries(tighter, summary_lines=summary)

        self.write_memory(rendered)

    def add_entry(self, text: str, *, kind: str = "manual", source: str | None = None) -> bool:
        normalized = _normalize_whitespace(text.strip())
        if not normalized:
            return False
        if source:
            normalized = f"{source}: {normalized}"

        entries = self.parse_entries()
        key = _entry_key(kind, normalized)
        if any(_entry_key(entry.kind, entry.text) == key for entry in entries):
            return False

        entries.append(MemoryEntry(timestamp=_now_iso(), kind=kind.lower(), text=normalized))
        self.write_memory(self._render_entries(entries))
        self._compact_if_needed()
        return True

    def detect_external_guidance_files(self) -> list[Path]:
        return [path for name in EXTERNAL_GUIDANCE_FILES if (path := self.root / name).is_file()]

    def import_external_guidance(self) -> list[Path]:
        imported: list[Path] = []
        for path in self.detect_external_guidance_files():
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            clipped = _clip(_normalize_whitespace(content), MAX_IMPORTED_FILE_CHARS)
            if self.add_entry(clipped, kind="import", source=path.name):
                imported.append(path)
        return imported

    def load_preferences(self) -> dict[str, str]:
        if not self.preferences_path.is_file():
            return {}
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # type: ignore

        try:
            with open(self.preferences_path, "rb") as handle:
                data = tomllib.load(handle)
        except (OSError, ValueError):
            return {}

        table = data.get("preferences") if isinstance(data.get("preferences"), dict) else data
        if not isinstance(table, dict):
            return {}

        preferred_model = table.get("preferred_model") or table.get("model")
        reasoning_level = table.get("reasoning_level") or table.get("reasoning")
        coding_style = table.get("coding_style") or table.get("style")

        prefs: dict[str, str] = {}
        if isinstance(preferred_model, (str, int, float, bool)):
            prefs["preferred_model"] = str(preferred_model)
        if isinstance(reasoning_level, (str, int, float, bool)):
            prefs["reasoning_level"] = str(reasoning_level)
        if isinstance(coding_style, (str, int, float, bool)):
            prefs["coding_style"] = str(coding_style)

        for key, value in table.items():
            if key in {"preferred_model", "model", "reasoning_level", "reasoning", "coding_style", "style"}:
                continue
            if isinstance(value, (str, int, float, bool)):
                prefs[str(key)] = str(value)
        return prefs

    def _guidance_context(self) -> list[str]:
        chunks: list[str] = []
        for path in self.detect_external_guidance_files():
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            chunks.append(f"### {path.name}\n{_clip(text, 1_200)}")
        return chunks

    def build_prompt_context(self) -> str | None:
        sections: list[str] = []

        entries = self.parse_entries()
        if entries:
            lines = ["## Project Memory"]
            for entry in entries[-MAX_CONTEXT_ENTRIES:]:
                lines.append(f"- [{entry.kind}] {entry.text}")
            sections.append("\n".join(lines))

        preferences = self.load_preferences()
        if preferences:
            lines = ["## User Preferences (.copex/preferences.toml)"]
            for key, value in sorted(preferences.items()):
                lines.append(f"- {key}: {value}")
            sections.append("\n".join(lines))

        guidance = self._guidance_context()
        if guidance:
            sections.append("## Existing Guidance Files\n\n" + "\n\n".join(guidance))

        context = "\n\n".join(section for section in sections if section.strip()).strip()
        if not context:
            return None
        if len(context) > MAX_PROMPT_CONTEXT_CHARS:
            return f"{context[:MAX_PROMPT_CONTEXT_CHARS].rstrip()}\n\n[Memory context truncated]"
        return context

    def compose_instructions(self, base_instructions: str | None) -> str | None:
        context = self.build_prompt_context()
        base = (base_instructions or "").strip()
        if not context:
            return base or None

        memory_block = (
            "## Persistent Project Memory\n"
            "Use this memory to remain consistent with prior project decisions, patterns, and preferences.\n\n"
            f"{context}"
        )
        if base:
            return f"{base}\n\n{memory_block}"
        return memory_block

    def learn_from_session(self, prompt: str, response: str, *, mode: str | None = None) -> int:
        candidates = extract_learning_candidates(prompt, response)
        if mode:
            candidates = [(kind, f"[{mode}] {text}") for kind, text in candidates]

        added = 0
        for kind, text in candidates:
            if self.add_entry(text, kind=kind):
                added += 1
        return added


def _clean_sentence(text: str) -> str:
    stripped = text.strip().strip("-*•").strip()
    return _normalize_whitespace(stripped)


def extract_learning_candidates(prompt: str, response: str) -> list[tuple[str, str]]:
    """Extract candidate (kind, text) learnings from a successful interaction."""
    candidates: list[tuple[str, str]] = []

    def _collect(source_text: str, *, from_prompt: bool) -> None:
        for raw in _SENTENCE_SPLIT_RE.split(source_text):
            sentence = _clean_sentence(raw)
            if len(sentence) < 25:
                continue
            lowered = sentence.lower()
            if lowered.startswith(("you are ", "task:", "overall task:", "current step:")):
                continue
            if from_prompt and _PREFERENCE_HINT_RE.search(sentence):
                candidates.append(("preference", _clip(sentence, 240)))
                continue
            if _DECISION_HINT_RE.search(sentence):
                candidates.append(("decision", _clip(sentence, 240)))
                continue
            if _PATTERN_HINT_RE.search(sentence):
                candidates.append(("pattern", _clip(sentence, 240)))

    _collect(prompt, from_prompt=True)
    _collect(response, from_prompt=False)

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, text in candidates:
        key = _entry_key(kind, text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((kind, text))
        if len(deduped) >= 8:
            break
    return deduped


def compose_memory_instructions(
    base_instructions: str | None,
    *,
    root: Path | None = None,
) -> str | None:
    """Merge persistent memory context into base instructions."""
    return ProjectMemory(root=root).compose_instructions(base_instructions)


def auto_capture_memory(
    prompt: str,
    response: str,
    *,
    root: Path | None = None,
    mode: str | None = None,
) -> int:
    """Extract and store learnings from a successful interaction."""
    return ProjectMemory(root=root).learn_from_session(prompt, response, mode=mode)
