"""Approval workflow and change preview utilities for file-modifying tool calls."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from copex.constants import PATH_KEYS


class ApprovalMode(str, Enum):
    """Modes controlling how file changes are handled."""

    AUTO_APPROVE = "auto-approve"
    APPROVE = "approve"
    MANUAL = "manual"
    DENY_ALL = "deny-all"
    POLICY_BASED = "policy-based"
    DRY_RUN = "dry-run"


class ApprovalAction(str, Enum):
    """Per-change decision action."""

    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"
    EDIT = "edit"


class DecisionOutcome(str, Enum):
    """Mode-level decision outcome before per-file action."""

    APPROVE = "approve"
    REJECT = "reject"
    REQUIRE_INPUT = "require-input"


@dataclass
class ApprovalState:
    """Mutable approval state for one review session."""

    mode: ApprovalMode
    approve_all: bool = False
    reject_all: bool = False


@dataclass(frozen=True)
class ChangeStatistics:
    """Aggregate statistics for a change set."""

    files_changed: int
    lines_added: int
    lines_removed: int
    touched_paths: list[str] = field(default_factory=list)
    operations: dict[str, int] = field(default_factory=dict)
    risky_operations: dict[str, list[str]] = field(default_factory=dict)
    risk: RiskAssessment = field(default_factory=lambda: RiskAssessment(severity="low"))


@dataclass(frozen=True)
class RiskAssessment:
    """Structured risk severity + reasons."""

    severity: str
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChangePreview:
    """Computed preview for a single file change."""

    file_path: str
    operation: str
    metadata: dict[str, Any]
    summary: str
    unified_diff: str
    lines_added: int
    lines_removed: int
    diff_truncated: bool = False
    diff_lines_total: int = 0
    risk: RiskAssessment = field(default_factory=lambda: RiskAssessment(severity="low"))
    risk_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProposedFileChange:
    """A candidate file change extracted from a tool call."""

    path: Path
    display_path: str
    existed_before: bool
    before_content: str
    after_content: str | None = None
    proposed_diff: str | None = None


@dataclass(frozen=True)
class ReviewedChange:
    """A reviewed change with user decision and optional edited content."""

    proposal: ProposedFileChange
    preview: ChangePreview
    action: ApprovalAction
    apply_change: bool
    edited_content: str | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class AuditEntry:
    """A single audit-log record."""

    timestamp: str
    mode: str
    model: str
    file: str
    diff_summary: str
    lines_added: int
    lines_removed: int
    risk_flags: list[str]
    action: str
    dry_run: bool
    event: str = "decision"
    decision: str = ""
    risk: dict[str, Any] = field(default_factory=dict)
    change_fingerprint: str = ""
    result: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        decision = self.decision or self.action
        risk_payload = self.risk or {"severity": "low", "reasons": list(self.risk_flags)}
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "model": self.model,
            "event": self.event,
            "decision": decision,
            "file": self.file,
            "diff_summary": self.diff_summary,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "risk_flags": self.risk_flags,
            "risk": risk_payload,
            "change_fingerprint": self.change_fingerprint,
            "action": self.action,
            "dry_run": self.dry_run,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AuditEntry:
        risk_payload = payload.get("risk")
        if isinstance(risk_payload, dict):
            severity = str(risk_payload.get("severity", "low"))
            reasons = [str(item) for item in risk_payload.get("reasons", []) if str(item)]
            normalized_risk = {"severity": severity, "reasons": reasons}
            risk_flags = reasons
        else:
            risk_flags = [str(item) for item in payload.get("risk_flags", []) if str(item)]
            normalized_risk = {"severity": "low", "reasons": risk_flags}
        return cls(
            timestamp=str(payload.get("timestamp", "")),
            mode=str(payload.get("mode", "")),
            model=str(payload.get("model", "")),
            file=str(payload.get("file", "")),
            diff_summary=str(payload.get("diff_summary", "")),
            lines_added=int(payload.get("lines_added", 0)),
            lines_removed=int(payload.get("lines_removed", 0)),
            risk_flags=risk_flags,
            action=str(payload.get("action", "")),
            dry_run=bool(payload.get("dry_run", False)),
            event=str(payload.get("event", "decision")),
            decision=str(payload.get("decision", payload.get("action", ""))),
            risk=normalized_risk,
            change_fingerprint=str(payload.get("change_fingerprint", "")),
            result=None if payload.get("result", None) is None else str(payload.get("result")),
            error=None if payload.get("error", None) is None else str(payload.get("error")),
        )


NEW_CONTENT_KEYS = ("content", "new_content", "text", "contents")
OLD_STRING_KEYS = ("old_string", "oldText", "old")
NEW_STRING_KEYS = ("new_string", "newText", "new")
APPEND_KEYS = ("append", "append_content")
WRITE_TOOL_HINTS = ("write", "edit", "create", "delete", "patch", "replace")
APPLY_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)
CONFIG_FILE_NAMES = {
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "setup.cfg",
    "tox.ini",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".env",
    ".env.example",
    "makefile",
}
SECURITY_PATH_MARKERS = (
    "auth",
    "security",
    "secret",
    "token",
    "credential",
    "permission",
    "oauth",
    "ssh",
    "key",
)
PATCH_OP_MAP = {"add": "add", "update": "edit", "delete": "delete"}
PATCH_OPERATION_RE = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+)$", re.MULTILINE)
DEFAULT_MAX_DIFF_LINES = 400
DEFAULT_MAX_DIFF_CHARS = 20_000
SECRET_CONTENT_PATTERNS = (
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd)\b\s*[:=]"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
)
RISK_REASON_SCORES = {
    "large-deletion": 3,
    "secret-like-content": 3,
    "delete-operation": 2,
    "security-sensitive-file": 2,
    "config-file-change": 1,
}


def default_audit_log_path(cwd: Path | None = None) -> Path:
    """Default audit log location."""
    project_root = Path.cwd().resolve(strict=False)
    base = Path(cwd) if cwd is not None else project_root
    if not base.is_absolute():
        base = project_root / base
    resolved_base = base.resolve(strict=False)
    try:
        resolved_base.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"Audit log path must stay within the project directory: {project_root}"
        ) from exc
    return resolved_base / ".copex" / "audit.log"


def normalize_approval_mode(value: ApprovalMode | str | None) -> ApprovalMode:
    """Normalize a mode value into an ``ApprovalMode``."""
    if value is None:
        return ApprovalMode.AUTO_APPROVE
    if isinstance(value, ApprovalMode):
        if value == ApprovalMode.APPROVE:
            return ApprovalMode.MANUAL
        return value
    normalized = str(value).strip().lower()
    aliases = {
        "auto": ApprovalMode.AUTO_APPROVE.value,
        "auto-approve": ApprovalMode.AUTO_APPROVE.value,
        "auto_approve": ApprovalMode.AUTO_APPROVE.value,
        "approve": ApprovalMode.MANUAL.value,
        "interactive": ApprovalMode.MANUAL.value,
        "manual": ApprovalMode.MANUAL.value,
        "deny": ApprovalMode.DENY_ALL.value,
        "deny-all": ApprovalMode.DENY_ALL.value,
        "deny_all": ApprovalMode.DENY_ALL.value,
        "policy": ApprovalMode.POLICY_BASED.value,
        "policy-based": ApprovalMode.POLICY_BASED.value,
        "policy_based": ApprovalMode.POLICY_BASED.value,
        "dry": ApprovalMode.DRY_RUN.value,
        "dry-run": ApprovalMode.DRY_RUN.value,
        "dry_run": ApprovalMode.DRY_RUN.value,
    }
    mapped = aliases.get(normalized, normalized)
    return ApprovalMode(mapped)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _short_text(value: Any, *, limit: int = 300) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
    else:
        text = _stable_json(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _change_fingerprint(preview: ChangePreview) -> str:
    payload = {
        "file": preview.file_path,
        "operation": preview.operation,
        "summary": preview.summary,
        "lines_added": preview.lines_added,
        "lines_removed": preview.lines_removed,
        "risk": {
            "severity": preview.risk.severity,
            "reasons": list(preview.risk.reasons),
        },
    }
    digest = sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return digest[:16]


def _resolve_path(raw_path: str, cwd: Path) -> tuple[Path, str] | None:
    raw = raw_path.strip()
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    resolved = candidate.resolve(strict=False)
    cwd_resolved = cwd.resolve(strict=False)
    try:
        relative = resolved.relative_to(cwd_resolved).as_posix()
    except ValueError:
        return None
    return resolved, relative


def _read_existing(path: Path) -> tuple[str, bool]:
    if not path.exists():
        return "", False
    return path.read_text(encoding="utf-8"), True


def _extract_path(payload: dict[str, Any]) -> str | None:
    for key in PATH_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _first_string(payload: dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _count_diff_lines(diff_text: str) -> tuple[int, int]:
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def _build_unified_diff(path: str, before: str, after: str) -> str:
    import difflib

    before_lines = before.splitlines()
    after_lines = after.splitlines()
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    return "\n".join(diff)


def _contains_secret_like_content(content: str | None) -> bool:
    if not content:
        return False
    return any(pattern.search(content) for pattern in SECRET_CONTENT_PATTERNS)


def _risk_severity(reasons: Sequence[str]) -> str:
    score = sum(RISK_REASON_SCORES.get(reason, 0) for reason in set(reasons))
    if score >= 3:
        return "high"
    if score >= 1:
        return "medium"
    return "low"


def _build_risk_assessment(reasons: Sequence[str]) -> RiskAssessment:
    normalized = sorted(set(reasons))
    return RiskAssessment(severity=_risk_severity(normalized), reasons=normalized)


def _assess_risk(
    path: str,
    lines_removed: int,
    *,
    operation: str,
    after_content: str | None,
) -> list[str]:
    lowered = path.lower()
    flags: list[str] = []
    if operation == "delete":
        flags.append("delete-operation")
    if lines_removed >= 100:
        flags.append("large-deletion")
    if Path(path).name.lower() in CONFIG_FILE_NAMES:
        flags.append("config-file-change")
    if any(marker in lowered for marker in SECURITY_PATH_MARKERS):
        flags.append("security-sensitive-file")
    if _contains_secret_like_content(after_content):
        flags.append("secret-like-content")
    return flags


class RiskAssessor:
    """Assess risk for proposed file changes."""

    def flags(
        self,
        *,
        path: str,
        lines_removed: int,
        operation: str,
        after_content: str | None = None,
    ) -> list[str]:
        return _assess_risk(
            path,
            lines_removed,
            operation=operation,
            after_content=after_content,
        )

    def assess(
        self,
        *,
        path: str,
        lines_removed: int,
        operation: str,
        after_content: str | None = None,
    ) -> RiskAssessment:
        flags = self.flags(
            path=path,
            lines_removed=lines_removed,
            operation=operation,
            after_content=after_content,
        )
        return _build_risk_assessment(flags)

    def assess_preview(self, preview: ChangePreview) -> RiskAssessment:
        return self.assess(
            path=preview.file_path,
            lines_removed=preview.lines_removed,
            operation=preview.operation,
        )


def _build_summary(
    *,
    operation: str,
    lines_added: int,
    lines_removed: int,
) -> str:
    return f"{operation} (+{lines_added}/-{lines_removed})"


def _patch_operation_for_path(proposed_diff: str, path: str) -> str | None:
    expected = path.replace("\\", "/").strip()
    for match in PATCH_OPERATION_RE.finditer(proposed_diff):
        op = match.group(1).strip().lower()
        raw_path = match.group(2).strip().replace("\\", "/")
        if raw_path == expected:
            return PATCH_OP_MAP.get(op)
    return None


def _infer_operation(change: ProposedFileChange) -> str:
    if change.proposed_diff:
        patch_operation = _patch_operation_for_path(change.proposed_diff, change.display_path)
        if patch_operation:
            return patch_operation
    if not change.existed_before and change.after_content is not None:
        return "add"
    if change.existed_before and change.after_content == "":
        return "delete"
    return "edit"


def _build_file_metadata(
    change: ProposedFileChange, *, operation: str, after_content: str | None
) -> dict[str, Any]:
    if operation == "delete":
        exists_after: bool | None = False
    elif operation == "add":
        exists_after = True
    elif after_content is None and change.proposed_diff is not None:
        exists_after = None
    else:
        exists_after = True

    before_bytes = len(change.before_content.encode("utf-8"))
    after_bytes = len(after_content.encode("utf-8")) if after_content is not None else None
    return {
        "path": change.display_path,
        "operation": operation,
        "existed_before": change.existed_before,
        "exists_after": exists_after,
        "before_lines": len(change.before_content.splitlines()),
        "after_lines": len(after_content.splitlines()) if after_content is not None else None,
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
    }


def _truncate_diff(
    unified_diff: str,
    *,
    max_lines: int = DEFAULT_MAX_DIFF_LINES,
    max_chars: int = DEFAULT_MAX_DIFF_CHARS,
) -> tuple[str, bool, int]:
    lines = unified_diff.splitlines()
    total_lines = len(lines)
    truncated = False
    shown_lines = lines

    if max_lines > 0 and len(shown_lines) > max_lines:
        shown_lines = shown_lines[:max_lines]
        truncated = True

    rendered = "\n".join(shown_lines)
    if max_chars > 0 and len(rendered) > max_chars:
        clipped = rendered[:max_chars]
        if "\n" in clipped:
            clipped = clipped.rsplit("\n", 1)[0]
        rendered = clipped
        shown_lines = rendered.splitlines()
        truncated = True

    shown_count = len(shown_lines)
    if truncated:
        notice = f"... [diff truncated: {shown_count}/{total_lines} lines shown]"
        rendered = f"{rendered}\n{notice}" if rendered else notice

    return rendered, truncated, total_lines


def _infer_after_content(
    tool_name: str,
    payload: dict[str, Any],
    before: str,
) -> str | None:
    explicit = _first_string(payload, NEW_CONTENT_KEYS)
    if explicit is not None:
        return explicit

    old_text = _first_string(payload, OLD_STRING_KEYS)
    new_text = _first_string(payload, NEW_STRING_KEYS)
    if old_text is not None and new_text is not None:
        return before.replace(old_text, new_text, 1)

    edits = payload.get("edits")
    if isinstance(edits, list):
        updated = before
        changed = False
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            old_edit = _first_string(edit, OLD_STRING_KEYS)
            new_edit = _first_string(edit, NEW_STRING_KEYS)
            if old_edit is None or new_edit is None:
                continue
            if old_edit in updated:
                updated = updated.replace(old_edit, new_edit, 1)
                changed = True
        if changed:
            return updated

    append_text = _first_string(payload, APPEND_KEYS)
    if append_text is not None:
        return before + append_text

    if "delete" in tool_name.lower():
        return ""

    return None


def _extract_patch_paths(patch_text: str) -> list[str]:
    return [match.group(1).strip() for match in APPLY_PATCH_FILE_RE.finditer(patch_text)]


def extract_proposed_changes(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    cwd: Path | None = None,
) -> list[ProposedFileChange]:
    """Extract candidate file changes from a tool call payload."""
    workdir = (cwd or Path.cwd()).resolve(strict=False)
    lowered_name = tool_name.lower()
    if not any(hint in lowered_name for hint in WRITE_TOOL_HINTS) and "patch" not in tool_args:
        return []

    changes: list[ProposedFileChange] = []
    payloads: list[dict[str, Any]] = []

    nested_changes = tool_args.get("changes")
    if isinstance(nested_changes, list):
        payloads.extend(item for item in nested_changes if isinstance(item, dict))

    nested_files = tool_args.get("files")
    if isinstance(nested_files, list):
        payloads.extend(item for item in nested_files if isinstance(item, dict))

    if not payloads:
        payloads.append(tool_args)

    for payload in payloads:
        raw_path = _extract_path(payload)
        if raw_path is None:
            continue
        resolved_info = _resolve_path(raw_path, workdir)
        if resolved_info is None:
            continue
        resolved_path, display_path = resolved_info
        before, existed_before = _read_existing(resolved_path)
        after_content = _infer_after_content(lowered_name, payload, before)
        changes.append(
            ProposedFileChange(
                path=resolved_path,
                display_path=display_path,
                existed_before=existed_before,
                before_content=before,
                after_content=after_content,
            )
        )

    patch_text = tool_args.get("patch") or tool_args.get("diff")
    if isinstance(patch_text, str) and patch_text.strip():
        for raw_path in _extract_patch_paths(patch_text):
            resolved_info = _resolve_path(raw_path, workdir)
            if resolved_info is None:
                continue
            resolved_path, display_path = resolved_info
            before, existed_before = _read_existing(resolved_path)
            changes.append(
                ProposedFileChange(
                    path=resolved_path,
                    display_path=display_path,
                    existed_before=existed_before,
                    before_content=before,
                    after_content=None,
                    proposed_diff=patch_text,
                )
            )

    # Deduplicate by path while preserving first occurrence.
    deduped: dict[Path, ProposedFileChange] = {}
    for change in changes:
        deduped.setdefault(change.path, change)
    return list(deduped.values())


def build_preview(
    change: ProposedFileChange,
    *,
    max_diff_lines: int = DEFAULT_MAX_DIFF_LINES,
    max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS,
) -> ChangePreview:
    """Build a diff preview for a proposed change."""
    operation = _infer_operation(change)

    if change.proposed_diff is not None:
        unified_diff_raw = change.proposed_diff
        lines_added, lines_removed = _count_diff_lines(unified_diff_raw)
        after_for_metadata = change.after_content
        if after_for_metadata is None and operation == "delete":
            after_for_metadata = ""
    else:
        after = change.after_content if change.after_content is not None else change.before_content
        unified_diff_raw = _build_unified_diff(change.display_path, change.before_content, after)
        lines_added, lines_removed = _count_diff_lines(unified_diff_raw)
        after_for_metadata = after

    unified_diff, diff_truncated, diff_lines_total = _truncate_diff(
        unified_diff_raw,
        max_lines=max_diff_lines,
        max_chars=max_diff_chars,
    )
    summary = _build_summary(
        operation=operation,
        lines_added=lines_added,
        lines_removed=lines_removed,
    )
    metadata = _build_file_metadata(
        change,
        operation=operation,
        after_content=after_for_metadata,
    )
    risk = RiskAssessor().assess(
        path=change.display_path,
        lines_removed=lines_removed,
        operation=operation,
        after_content=after_for_metadata,
    )
    return ChangePreview(
        file_path=change.display_path,
        operation=operation,
        metadata=metadata,
        summary=summary,
        unified_diff=unified_diff,
        lines_added=lines_added,
        lines_removed=lines_removed,
        diff_truncated=diff_truncated,
        diff_lines_total=diff_lines_total,
        risk=risk,
        risk_flags=list(risk.reasons),
    )


def summarize_changes(previews: Iterable[ChangePreview]) -> ChangeStatistics:
    """Compute aggregate change statistics."""
    collected = list(previews)
    touched_paths = [item.file_path for item in collected]
    operations = dict(Counter(item.operation for item in collected))
    risky_operations: dict[str, list[str]] = {}
    for item in collected:
        for reason in item.risk_flags:
            risky_operations.setdefault(reason, [])
            if item.file_path not in risky_operations[reason]:
                risky_operations[reason].append(item.file_path)
    aggregate_risk = _build_risk_assessment(list(risky_operations))
    return ChangeStatistics(
        files_changed=len(collected),
        lines_added=sum(item.lines_added for item in collected),
        lines_removed=sum(item.lines_removed for item in collected),
        touched_paths=touched_paths,
        operations=operations,
        risky_operations=risky_operations,
        risk=aggregate_risk,
    )


class ChangeStats:
    """Helpers for computing file-change summaries."""

    @staticmethod
    def summarize(previews: Iterable[ChangePreview]) -> ChangeStatistics:
        return summarize_changes(previews)

    @staticmethod
    def from_previews(previews: Iterable[ChangePreview]) -> ChangeStatistics:
        return summarize_changes(previews)


class DiffPreview:
    """Generate and render colored unified diff previews."""

    def __init__(
        self,
        *,
        max_diff_lines: int = DEFAULT_MAX_DIFF_LINES,
        max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS,
    ):
        self.max_diff_lines = max_diff_lines
        self.max_diff_chars = max_diff_chars

    def build(self, change: ProposedFileChange) -> ChangePreview:
        return build_preview(
            change,
            max_diff_lines=self.max_diff_lines,
            max_diff_chars=self.max_diff_chars,
        )

    def from_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        cwd: Path | None = None,
    ) -> list[ChangePreview]:
        proposals = extract_proposed_changes(tool_name, tool_args, cwd=cwd)
        return [self.build(proposal) for proposal in proposals]

    @staticmethod
    def colorize(unified_diff: str) -> Text:
        text = Text()
        for line in unified_diff.splitlines():
            style = ""
            if line.startswith("+++") or line.startswith("---"):
                style = "bold cyan"
            elif line.startswith("@@"):
                style = "cyan"
            elif line.startswith("+"):
                style = "green"
            elif line.startswith("-"):
                style = "red"
            text.append(line + "\n", style=style)
        return text

    def render_panel(
        self,
        preview: ChangePreview,
        stats: ChangeStatistics | None = None,
    ) -> Panel:
        metadata = preview.metadata
        header = Text()
        header.append(f"File: {preview.file_path}\n", style="bold")
        header.append(f"Operation: {preview.operation}\n")
        header.append(
            "Metadata: "
            f"lines={metadata.get('before_lines', '?')}→{metadata.get('after_lines', '?')} "
            f"bytes={metadata.get('before_bytes', '?')}→{metadata.get('after_bytes', '?')}\n",
            style="dim",
        )
        header.append(f"Summary: {preview.summary}\n")

        files_changed = stats.files_changed if stats is not None else 1
        lines_added = stats.lines_added if stats is not None else preview.lines_added
        lines_removed = stats.lines_removed if stats is not None else preview.lines_removed
        header.append(
            f"Stats: files={files_changed} +{lines_added} -{lines_removed}\n",
            style="dim",
        )

        if preview.diff_truncated:
            header.append(
                f"Preview: truncated ({preview.diff_lines_total} total diff lines)\n",
                style="yellow",
            )
        if preview.risk.reasons:
            style = "red" if preview.risk.severity == "high" else "yellow"
            header.append(
                f"Risk: {preview.risk.severity} ({', '.join(preview.risk.reasons)})\n",
                style=style,
            )
        else:
            header.append("Risk: low\n", style="green")

        content = Group(header, Text(), self.colorize(preview.unified_diff))
        return Panel(
            content,
            title="Change Preview",
            border_style="yellow",
            expand=True,
        )

    def render(
        self,
        console: Console,
        preview: ChangePreview,
        stats: ChangeStatistics | None = None,
    ) -> None:
        console.print(self.render_panel(preview, stats))


class AuditLogger:
    """Append/query audit entries stored as JSONL."""

    def __init__(self, path: Path | None = None):
        self.path = path or default_audit_log_path()

    def write(self, entry: AuditEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def read(self) -> list[AuditEntry]:
        if not self.path.exists():
            return []
        entries: list[AuditEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(AuditEntry.from_dict(payload))
        return entries

    def query(self, *, last: int = 10, file: str | None = None) -> list[AuditEntry]:
        entries = self.read()
        if file:
            wanted = file.strip().replace("\\", "/")
            entries = [entry for entry in entries if entry.file.replace("\\", "/") == wanted]
        if last > 0:
            entries = entries[-last:]
        return entries


class ApprovalWorkflow:
    """Coordinates preview, decision, and audit behavior for file changes."""

    def __init__(
        self,
        mode: ApprovalMode | str = ApprovalMode.AUTO_APPROVE,
        *,
        model: str = "unknown",
        audit_enabled: bool = False,
        audit_logger: AuditLogger | None = None,
        console: Console | None = None,
        input_func: Callable[[str], str] | None = None,
        edit_func: Callable[[Path, str], str] | None = None,
        policy_func: Callable[[ChangePreview], DecisionOutcome | ApprovalAction | bool | str]
        | None = None,
        decision_prompt_func: Callable[
            [Console, ChangePreview, ChangeStatistics, Callable[[str], str]],
            ApprovalAction | str,
        ]
        | None = None,
    ):
        normalized_mode = normalize_approval_mode(mode)
        if normalized_mode == ApprovalMode.POLICY_BASED and policy_func is None:
            raise ValueError("policy_func is required when approval mode is policy-based")
        if decision_prompt_func is None:
            from copex.interactive import prompt_approval_decision

            decision_prompt_func = prompt_approval_decision
        self.state = ApprovalState(mode=normalized_mode)
        self.model = model
        self.audit_enabled = audit_enabled
        self._audit = audit_logger or AuditLogger()
        self._console = console or Console()
        self._input = input_func or input
        self._edit = edit_func
        self._policy = policy_func
        self._diff_preview = DiffPreview()
        self._decision_prompt = decision_prompt_func

    @property
    def mode(self) -> ApprovalMode:
        return self.state.mode

    def _colorize_diff(self, unified_diff: str) -> Text:
        return self._diff_preview.colorize(unified_diff)

    def _render_preview(self, preview: ChangePreview, stats: ChangeStatistics) -> None:
        self._diff_preview.render(self._console, preview, stats)

    def _normalize_prompt_choice(self, choice: ApprovalAction | str) -> ApprovalAction:
        if isinstance(choice, ApprovalAction):
            return choice
        answer = str(choice).strip().lower()
        if answer in {"a", "all", "approve-all"}:
            self.state.approve_all = True
            return ApprovalAction.APPROVE
        if answer in {"r", "reject-all", "deny-all"}:
            self.state.reject_all = True
            return ApprovalAction.REJECT
        if answer in {"y", "yes", "approve"}:
            return ApprovalAction.APPROVE
        if answer in {"n", "no", "reject", "deny"}:
            return ApprovalAction.REJECT
        if answer in {"d", "defer", "later"}:
            return ApprovalAction.DEFER
        if answer in {"e", "edit"}:
            return ApprovalAction.EDIT
        raise ValueError(f"Unsupported approval choice: {choice}")

    def _prompt_action(self, preview: ChangePreview, stats: ChangeStatistics) -> ApprovalAction:
        if self._decision_prompt is not None:
            choice = self._decision_prompt(self._console, preview, stats, self._input)
            return self._normalize_prompt_choice(choice)

        prompt = (
            f"Approve change for {preview.file_path}? "
            "[y]es/[n]o/[d]efer/[e]dit/[a]pprove-all/[r]eject-all: "
        )
        try:
            return self._normalize_prompt_choice(self._input(prompt))
        except EOFError:
            return ApprovalAction.REJECT
        except ValueError:
            return ApprovalAction.REJECT

    def _coerce_policy_outcome(
        self, value: DecisionOutcome | ApprovalAction | bool | str
    ) -> DecisionOutcome:
        if isinstance(value, DecisionOutcome):
            return value
        if isinstance(value, ApprovalAction):
            if value == ApprovalAction.APPROVE:
                return DecisionOutcome.APPROVE
            if value == ApprovalAction.REJECT:
                return DecisionOutcome.REJECT
            if value == ApprovalAction.DEFER:
                return DecisionOutcome.REQUIRE_INPUT
            return DecisionOutcome.REQUIRE_INPUT
        if isinstance(value, bool):
            return DecisionOutcome.APPROVE if value else DecisionOutcome.REJECT

        normalized = str(value).strip().lower()
        mapping = {
            "approve": DecisionOutcome.APPROVE,
            "allow": DecisionOutcome.APPROVE,
            "reject": DecisionOutcome.REJECT,
            "deny": DecisionOutcome.REJECT,
            "manual": DecisionOutcome.REQUIRE_INPUT,
            "review": DecisionOutcome.REQUIRE_INPUT,
            "prompt": DecisionOutcome.REQUIRE_INPUT,
            "escalate": DecisionOutcome.REQUIRE_INPUT,
        }
        mapped = mapping.get(normalized)
        if mapped is None:
            raise ValueError(f"Unsupported policy decision: {value}")
        return mapped

    def _decide_outcome(self, preview: ChangePreview) -> DecisionOutcome:
        if self.state.approve_all:
            return DecisionOutcome.APPROVE
        if self.state.reject_all:
            return DecisionOutcome.REJECT
        if self.mode == ApprovalMode.AUTO_APPROVE:
            return DecisionOutcome.APPROVE
        if self.mode in {ApprovalMode.DRY_RUN, ApprovalMode.DENY_ALL}:
            return DecisionOutcome.REJECT
        if self.mode == ApprovalMode.POLICY_BASED:
            if self._policy is None:
                raise ValueError("policy_func is required when approval mode is policy-based")
            return self._coerce_policy_outcome(self._policy(preview))
        if self.mode == ApprovalMode.MANUAL:
            return DecisionOutcome.REQUIRE_INPUT
        raise ValueError(f"Unsupported approval mode: {self.mode.value}")

    def _decide_action(self, preview: ChangePreview, stats: ChangeStatistics) -> ApprovalAction:
        outcome = self._decide_outcome(preview)
        if outcome == DecisionOutcome.APPROVE:
            return ApprovalAction.APPROVE
        if outcome == DecisionOutcome.REJECT:
            return ApprovalAction.REJECT
        return self._prompt_action(preview, stats)

    def _edited_content(self, change: ProposedFileChange, fallback: str) -> str:
        if self._edit is not None:
            return self._edit(change.path, fallback)
        return fallback

    def _should_log(self) -> bool:
        return self.audit_enabled or self.mode != ApprovalMode.AUTO_APPROVE

    def _audit_entry_for(
        self,
        *,
        preview: ChangePreview,
        action: ApprovalAction,
        dry_run: bool,
        event: str,
        result: Any = None,
        error: str | None = None,
    ) -> AuditEntry:
        decision = action.value
        risk = {
            "severity": preview.risk.severity,
            "reasons": list(preview.risk.reasons),
        }
        return AuditEntry(
            timestamp=_now_iso(),
            mode=self.mode.value,
            model=self.model,
            event=event,
            decision=decision,
            file=preview.file_path,
            diff_summary=preview.summary,
            lines_added=preview.lines_added,
            lines_removed=preview.lines_removed,
            risk_flags=preview.risk_flags,
            risk=risk,
            change_fingerprint=_change_fingerprint(preview),
            action=decision,
            dry_run=dry_run,
            result=_short_text(result),
            error=_short_text(error),
        )

    def log_execution_event(
        self,
        reviewed: Sequence[ReviewedChange],
        *,
        success: bool,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Write execution audit entries for reviewed changes."""
        if not reviewed or not self._should_log():
            return
        for item in reviewed:
            if item.apply_change:
                status = "applied" if success else "failed"
                event_error = error if not success else None
            else:
                status = "skipped"
                event_error = None
            result_payload = {
                "status": status,
                "tool_result": _short_text(result),
            }
            self._audit.write(
                self._audit_entry_for(
                    preview=item.preview,
                    action=item.action,
                    dry_run=item.dry_run,
                    event="execution",
                    result=result_payload,
                    error=event_error,
                )
            )

    def review_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        cwd: Path | None = None,
    ) -> list[ReviewedChange]:
        """Review a tool call and return per-file decisions."""
        proposals = extract_proposed_changes(tool_name, tool_args, cwd=cwd)
        if not proposals:
            return []

        previews = [build_preview(change) for change in proposals]
        stats = summarize_changes(previews)

        reviewed: list[ReviewedChange] = []
        for proposal, preview in zip(proposals, previews, strict=False):
            if self.mode != ApprovalMode.MANUAL or self._decision_prompt is None:
                self._render_preview(preview, stats)
            action = self._decide_action(preview, stats)
            dry_run = self.mode == ApprovalMode.DRY_RUN
            apply_change = action in {ApprovalAction.APPROVE, ApprovalAction.EDIT} and not dry_run
            edited_content = None
            if action == ApprovalAction.EDIT:
                base = (
                    proposal.after_content
                    if proposal.after_content is not None
                    else proposal.before_content
                )
                edited_content = self._edited_content(proposal, base)
            decision = ReviewedChange(
                proposal=proposal,
                preview=preview,
                action=action,
                apply_change=apply_change,
                edited_content=edited_content,
                dry_run=dry_run,
            )
            reviewed.append(decision)

            if self._should_log():
                self._audit.write(
                    self._audit_entry_for(
                        preview=preview,
                        action=action,
                        dry_run=dry_run,
                        event="decision",
                        result={"status": "approved" if apply_change else "blocked"},
                    )
                )

        return reviewed

    def apply_post_tool_decisions(self, reviewed: Sequence[ReviewedChange]) -> list[str]:
        """Enforce decisions after a tool call executes."""
        messages: list[str] = []
        for item in reviewed:
            proposal = item.proposal
            path = proposal.path

            if not item.apply_change:
                if proposal.existed_before:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(proposal.before_content, encoding="utf-8")
                elif path.exists():
                    path.unlink()
                if item.action == ApprovalAction.DEFER:
                    state = "deferred"
                else:
                    state = "dry-run revert" if item.dry_run else "reverted"
                messages.append(f"{state}: {proposal.display_path}")
                continue

            if item.edited_content is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(item.edited_content, encoding="utf-8")
                messages.append(f"edited: {proposal.display_path}")
            else:
                messages.append(f"approved: {proposal.display_path}")

        return messages


class ApprovalGate(ApprovalWorkflow):
    """Compatibility wrapper exposing approval-gate naming."""

    def __init__(
        self,
        mode: ApprovalMode | str = ApprovalMode.AUTO_APPROVE,
        *,
        audit: bool = False,
        **kwargs: Any,
    ):
        current = bool(kwargs.pop("audit_enabled", False))
        super().__init__(
            mode=mode,
            audit_enabled=current or audit,
            **kwargs,
        )

    def review(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        cwd: Path | None = None,
    ) -> list[ReviewedChange]:
        return self.review_tool_call(tool_name, tool_args, cwd=cwd)

    def enforce(self, reviewed: Sequence[ReviewedChange]) -> list[str]:
        return self.apply_post_tool_decisions(reviewed)
