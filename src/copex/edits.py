"""Structured edit parsing, application, verification, and undo support."""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from copex.security import validate_path
from copex.tools import read_text_file, write_text_file, write_text_file_atomic


class EditFormat(str, Enum):
    """Supported structured edit formats."""

    SEARCH_REPLACE = "search_replace"
    UNIFIED_DIFF = "unified_diff"
    WHOLE_FILE = "whole_file"


@dataclass(slots=True)
class UnifiedHunk:
    """One unified-diff hunk for a file."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EditOperation:
    """Single parsed edit operation."""

    file_path: str
    format: EditFormat
    search: str | None = None
    replace: str | None = None
    content: str | None = None
    hunks: list[UnifiedHunk] = field(default_factory=list)


@dataclass(slots=True)
class VerificationCheck:
    """Result of one verification check."""

    name: str
    ran: bool
    success: bool
    output: str = ""


@dataclass(slots=True)
class VerificationReport:
    """Aggregate verification report."""

    checks: list[VerificationCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.success for check in self.checks if check.ran)

    def to_feedback(self) -> str:
        if self.ok:
            return "Verification passed."
        lines = ["Verification failed. Please fix the following issues:"]
        for check in self.checks:
            if check.ran and not check.success:
                if check.output.strip():
                    lines.append(f"- {check.name}:\n{check.output.strip()}")
                else:
                    lines.append(f"- {check.name}: failed")
        return "\n".join(lines)


@dataclass(slots=True)
class EditBatchResult:
    """Result of applying a parsed batch of edits."""

    applied_files: list[str] = field(default_factory=list)
    failed_files: dict[str, str] = field(default_factory=dict)
    undo_batch_id: str | None = None
    verification: VerificationReport | None = None

    @property
    def success(self) -> bool:
        if self.failed_files:
            return False
        if self.verification is not None and not self.verification.ok:
            return False
        return True

    def verification_feedback(self) -> str:
        if self.verification is None:
            return ""
        return self.verification.to_feedback()


@dataclass(slots=True)
class UndoBatchInfo:
    """One undo batch entry."""

    batch_id: str
    created_at: str
    files: list[str] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.files)


@dataclass(slots=True)
class UndoResult:
    """Result of an undo operation."""

    batch_id: str
    restored_files: list[str] = field(default_factory=list)


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_FILE_HEADER_RE = re.compile(r"^\s*(?:File|Path)\s*:\s*(?P<path>.+?)\s*$")
_MARKDOWN_PATH_RE = re.compile(r"^\s*#+\s*(?P<path>(?:\./)?[A-Za-z0-9_.\-/]+\.[A-Za-z0-9_]+)\s*$")
_PLAIN_PATH_RE = re.compile(r"^\s*(?P<path>(?:\./)?[A-Za-z0-9_.\-/]+\.[A-Za-z0-9_]+)\s*$")


def parse_structured_edits(text: str) -> list[EditOperation]:
    """Parse structured edit operations from AI response text."""
    matches: list[tuple[int, EditOperation]] = []
    matches.extend(_parse_search_replace_operations(text))
    matches.extend(_parse_unified_diff_operations(text))
    matches.extend(_parse_whole_file_operations(text))
    matches.sort(key=lambda item: item[0])
    operations = [op for _, op in matches]
    if not operations:
        raise ValueError("No structured edits found in response.")
    return operations


def apply_edit_text(text: str, *, root: Path | None = None, verify: bool = True) -> EditBatchResult:
    """Parse and apply structured edits."""
    operations = parse_structured_edits(text)
    return apply_edit_operations(operations, root=root, verify=verify)


def apply_edit_operations(
    operations: Iterable[EditOperation],
    *,
    root: Path | None = None,
    verify: bool = True,
) -> EditBatchResult:
    """Apply parsed operations with per-file atomicity and undo snapshots."""
    workspace = (root or Path.cwd()).resolve()
    grouped: dict[Path, list[EditOperation]] = {}
    for op in operations:
        target = _resolve_target_path(workspace, op.file_path)
        grouped.setdefault(target, []).append(op)

    staged: dict[Path, str] = {}
    originals: dict[Path, str] = {}
    existed_before: dict[Path, bool] = {}
    failed_files: dict[str, str] = {}

    for path, file_ops in grouped.items():
        existed = path.is_file()
        existed_before[path] = existed
        original_content = read_text_file(path) if existed else ""
        updated_content = original_content

        try:
            for op in file_ops:
                if op.format is EditFormat.SEARCH_REPLACE:
                    updated_content = _apply_search_replace(
                        updated_content,
                        op.search or "",
                        op.replace or "",
                    )
                elif op.format is EditFormat.WHOLE_FILE:
                    updated_content = op.content or ""
                elif op.format is EditFormat.UNIFIED_DIFF:
                    updated_content = _apply_unified_diff(updated_content, op.hunks)
                else:
                    raise ValueError(f"Unsupported edit format: {op.format}")
        except Exception as exc:
            failed_files[_rel_display(path, workspace)] = str(exc)
            continue

        if updated_content != original_content:
            staged[path] = updated_content
            originals[path] = original_content

    undo_batch_id: str | None = None
    applied_files: list[str] = []
    if staged:
        undo_batch_id = _create_undo_batch(workspace, originals, existed_before)
        for path, content in staged.items():
            try:
                write_text_file_atomic(path, content)
                applied_files.append(_rel_display(path, workspace))
            except Exception as exc:
                failed_files[_rel_display(path, workspace)] = str(exc)

    verification_report: VerificationReport | None = None
    if verify and applied_files:
        changed = [workspace / rel_path for rel_path in applied_files]
        verification_report = run_verification(changed, root=workspace)

    return EditBatchResult(
        applied_files=applied_files,
        failed_files=failed_files,
        undo_batch_id=undo_batch_id,
        verification=verification_report,
    )


def run_verification(paths: Iterable[Path], *, root: Path | None = None) -> VerificationReport:
    """Run syntax + optional lint/type checks for changed Python files."""
    workspace = (root or Path.cwd()).resolve()
    files = [Path(p).resolve() for p in paths]
    py_files = [path for path in files if path.suffix == ".py" and path.exists()]

    checks: list[VerificationCheck] = []
    syntax_errors = _syntax_check(py_files)
    checks.append(
        VerificationCheck(
            name="syntax",
            ran=bool(py_files),
            success=not syntax_errors,
            output="\n".join(syntax_errors),
        )
    )

    rel_py_files = [str(path.relative_to(workspace)) for path in py_files]

    lint_tool = _detect_lint_tool()
    if lint_tool and rel_py_files:
        lint_cmd = _lint_command(lint_tool, rel_py_files)
        checks.append(_run_command_check("lint", lint_cmd, workspace))
    else:
        checks.append(VerificationCheck(name="lint", ran=False, success=True))

    if shutil.which("mypy") and rel_py_files:
        checks.append(_run_command_check("type", ["mypy", *rel_py_files], workspace))
    else:
        checks.append(VerificationCheck(name="type", ran=False, success=True))

    return VerificationReport(checks=checks)


def list_undo_history(root: Path | None = None) -> list[UndoBatchInfo]:
    """Return undo history, newest first."""
    workspace = (root or Path.cwd()).resolve()
    undo_root = _undo_dir(workspace)
    if not undo_root.exists():
        return []

    entries: list[UndoBatchInfo] = []
    for manifest_path in sorted(undo_root.glob("*/manifest.json"), reverse=True):
        try:
            data = json.loads(read_text_file(manifest_path))
        except (json.JSONDecodeError, OSError):
            continue
        entries.append(
            UndoBatchInfo(
                batch_id=data.get("batch_id", manifest_path.parent.name),
                created_at=data.get("created_at", ""),
                files=[item["path"] for item in data.get("files", []) if "path" in item],
            )
        )
    return entries


def undo_last_edit_batch(root: Path | None = None, *, batch_id: str | None = None) -> UndoResult:
    """Restore files from the newest undo batch (or explicit batch id)."""
    workspace = (root or Path.cwd()).resolve()
    history = list_undo_history(workspace)
    if not history:
        raise ValueError("No undo history found.")

    selected_id = batch_id or history[0].batch_id
    manifest_path = _undo_dir(workspace) / selected_id / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Undo batch not found: {selected_id}")

    data = json.loads(read_text_file(manifest_path))
    batch_dir = manifest_path.parent
    restored_files: list[str] = []

    for item in data.get("files", []):
        rel_path = item["path"]
        target = _resolve_target_path(workspace, rel_path)
        existed = bool(item.get("existed", False))
        snapshot_rel = item.get("snapshot")
        if existed:
            if not snapshot_rel:
                raise ValueError(f"Missing snapshot for {rel_path}")
            snapshot_path = batch_dir / snapshot_rel
            content = read_text_file(snapshot_path)
            write_text_file_atomic(target, content)
        else:
            if target.exists() and target.is_file():
                target.unlink()
        restored_files.append(rel_path)

    return UndoResult(batch_id=selected_id, restored_files=restored_files)


def _parse_search_replace_operations(text: str) -> list[tuple[int, EditOperation]]:
    lines = text.splitlines(keepends=True)
    offsets = _line_offsets(lines)
    operations: list[tuple[int, EditOperation]] = []
    current_file: str | None = None
    i = 0

    while i < len(lines):
        stripped = lines[i].rstrip("\r\n")
        header_path = _extract_path_header(stripped)
        if header_path:
            current_file = header_path
            i += 1
            continue

        if stripped.startswith("<<<<<<< SEARCH"):
            inline_path = stripped.replace("<<<<<<< SEARCH", "", 1).strip()
            if inline_path:
                current_file = _clean_path(inline_path)

            if not current_file:
                raise ValueError("SEARCH/REPLACE block missing target file path.")

            start = offsets[i]
            i += 1
            search_lines: list[str] = []
            while i < len(lines) and lines[i].strip() != "=======":
                search_lines.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ValueError(f"Unterminated SEARCH block for {current_file}")

            i += 1  # Skip =======
            replace_lines: list[str] = []
            while i < len(lines) and lines[i].strip() != ">>>>>>> REPLACE":
                replace_lines.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ValueError(f"Unterminated REPLACE block for {current_file}")

            operations.append(
                (
                    start,
                    EditOperation(
                        file_path=current_file,
                        format=EditFormat.SEARCH_REPLACE,
                        search="".join(search_lines),
                        replace="".join(replace_lines),
                    ),
                )
            )
        i += 1
    return operations


def _parse_whole_file_operations(text: str) -> list[tuple[int, EditOperation]]:
    lines = text.splitlines(keepends=True)
    offsets = _line_offsets(lines)
    operations: list[tuple[int, EditOperation]] = []
    i = 0

    while i < len(lines) - 1:
        header = lines[i].rstrip("\r\n")
        file_path = _extract_path_header(header)
        if not file_path:
            i += 1
            continue

        if not lines[i + 1].lstrip().startswith("```"):
            i += 1
            continue

        start = offsets[i]
        j = i + 2
        content_lines: list[str] = []
        while j < len(lines) and lines[j].strip() != "```":
            content_lines.append(lines[j])
            j += 1
        if j >= len(lines):
            break

        content = "".join(content_lines)
        if "diff --git " in content or "<<<<<<< SEARCH" in content:
            i = j + 1
            continue

        operations.append(
            (
                start,
                EditOperation(
                    file_path=file_path,
                    format=EditFormat.WHOLE_FILE,
                    content=content,
                ),
            )
        )
        i = j + 1

    return operations


def _parse_unified_diff_operations(text: str) -> list[tuple[int, EditOperation]]:
    lines = text.splitlines(keepends=True)
    offsets = _line_offsets(lines)
    operations: list[tuple[int, EditOperation]] = []
    i = 0

    while i < len(lines):
        if lines[i].startswith("diff --git "):
            start = offsets[i]
            i += 1
            while i < len(lines) and not lines[i].startswith("--- "):
                i += 1
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                continue
            old_header = lines[i].rstrip("\r\n")
            new_header = lines[i + 1].rstrip("\r\n")
            i += 2
            body: list[str] = []
            while i < len(lines) and not lines[i].startswith("diff --git "):
                if lines[i].startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
                    break
                body.append(lines[i].rstrip("\r\n"))
                i += 1
            op = _build_unified_diff_operation(old_header, new_header, body)
            if op is not None:
                operations.append((start, op))
            continue

        if lines[i].startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            start = offsets[i]
            old_header = lines[i].rstrip("\r\n")
            new_header = lines[i + 1].rstrip("\r\n")
            i += 2
            body: list[str] = []
            while i < len(lines):
                if lines[i].startswith("diff --git "):
                    break
                if lines[i].startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
                    break
                body.append(lines[i].rstrip("\r\n"))
                i += 1
            op = _build_unified_diff_operation(old_header, new_header, body)
            if op is not None:
                operations.append((start, op))
            continue

        i += 1

    return operations


def _build_unified_diff_operation(
    old_header: str,
    new_header: str,
    body: list[str],
) -> EditOperation | None:
    old_path = _clean_diff_header_path(old_header)
    new_path = _clean_diff_header_path(new_header)
    file_path = new_path if new_path != "/dev/null" else old_path
    if not file_path or file_path == "/dev/null":
        return None

    hunks: list[UnifiedHunk] = []
    i = 0
    while i < len(body):
        line = body[i]
        if not line.startswith("@@"):
            i += 1
            continue
        match = _HUNK_HEADER_RE.match(line)
        if not match:
            raise ValueError(f"Invalid unified diff hunk header: {line}")
        old_start = int(match.group(1))
        old_count = int(match.group(2) or "1")
        new_start = int(match.group(3))
        new_count = int(match.group(4) or "1")
        i += 1
        hunk_lines: list[str] = []
        while i < len(body) and not body[i].startswith("@@"):
            current = body[i]
            if current.startswith("\\"):
                i += 1
                continue
            if current.startswith((" ", "+", "-")):
                hunk_lines.append(current)
            i += 1
        hunks.append(
            UnifiedHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                lines=hunk_lines,
            )
        )

    if not hunks:
        return None

    return EditOperation(
        file_path=file_path,
        format=EditFormat.UNIFIED_DIFF,
        hunks=hunks,
    )


def _apply_search_replace(content: str, search: str, replace: str) -> str:
    if not search:
        return content + replace

    idx = content.find(search)
    if idx >= 0:
        return content[:idx] + replace + content[idx + len(search) :]

    span = _find_fuzzy_span(content, search)
    if span is None:
        raise ValueError("SEARCH block not found (exact or fuzzy match).")
    start, end = span
    return content[:start] + replace + content[end:]


def _find_fuzzy_span(content: str, search: str) -> tuple[int, int] | None:
    content_lines = content.splitlines(keepends=True)
    content_stripped = [_strip_line_ending(line) for line in content_lines]
    search_lines = [_strip_line_ending(line) for line in search.splitlines()]
    if not search_lines:
        return None
    if len(search_lines) > len(content_stripped):
        return None

    offsets = _line_offsets(content_lines)
    normalized_search = [_normalize_line(line) for line in search_lines]
    width = len(search_lines)

    for start_idx in range(0, len(content_stripped) - width + 1):
        window = content_stripped[start_idx : start_idx + width]
        normalized_window = [_normalize_line(line) for line in window]
        if normalized_window == normalized_search:
            start = offsets[start_idx]
            end = offsets[start_idx + width]
            return start, end
    return None


def _apply_unified_diff(content: str, hunks: list[UnifiedHunk]) -> str:
    source_lines = content.splitlines(keepends=True)
    newline = _detect_newline(content)
    output_lines: list[str] = []
    cursor = 0

    for hunk in hunks:
        start_idx = max(hunk.old_start - 1, 0)
        if start_idx > len(source_lines):
            raise ValueError("Unified diff hunk start is out of range.")
        if start_idx < cursor:
            start_idx = cursor

        output_lines.extend(source_lines[cursor:start_idx])
        cursor = start_idx

        for line in hunk.lines:
            kind = line[:1]
            payload = line[1:]

            if kind == " ":
                if cursor >= len(source_lines):
                    raise ValueError("Unified diff context exceeded file length.")
                _assert_line_match(source_lines[cursor], payload)
                output_lines.append(source_lines[cursor])
                cursor += 1
            elif kind == "-":
                if cursor >= len(source_lines):
                    raise ValueError("Unified diff deletion exceeded file length.")
                _assert_line_match(source_lines[cursor], payload)
                cursor += 1
            elif kind == "+":
                output_lines.append(payload + newline)
            else:
                raise ValueError(f"Unexpected unified diff line prefix: {kind}")

    output_lines.extend(source_lines[cursor:])
    return "".join(output_lines)


def _resolve_target_path(root: Path, file_path: str) -> Path:
    raw = Path(_clean_path(file_path))
    candidate = raw if raw.is_absolute() else root / raw
    return validate_path(candidate, base_dir=root, must_exist=False, allow_absolute=True)


def _create_undo_batch(
    root: Path,
    originals: dict[Path, str],
    existed_before: dict[Path, bool],
) -> str:
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    batch_dir = _undo_dir(root) / batch_id
    files_dir = batch_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    manifest_files: list[dict[str, object]] = []
    for path, content in originals.items():
        rel = _rel_display(path, root)
        existed = bool(existed_before.get(path, False))
        snapshot_rel: str | None = None
        if existed:
            snapshot_rel = str(Path("files") / rel)
            snapshot_path = batch_dir / snapshot_rel
            write_text_file(snapshot_path, content)
        manifest_files.append(
            {
                "path": rel,
                "snapshot": snapshot_rel,
                "existed": existed,
            }
        )

    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": manifest_files,
    }
    write_text_file(batch_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return batch_id


def _undo_dir(root: Path) -> Path:
    return root / ".copex" / "undo"


def _line_offsets(lines: list[str]) -> list[int]:
    offsets = [0]
    total = 0
    for line in lines:
        total += len(line)
        offsets.append(total)
    return offsets


def _extract_path_header(line: str) -> str | None:
    for pattern in (_FILE_HEADER_RE, _MARKDOWN_PATH_RE, _PLAIN_PATH_RE):
        match = pattern.match(line)
        if match:
            return _clean_path(match.group("path"))
    return None


def _clean_path(path: str) -> str:
    value = path.strip().strip("`")
    if value.startswith(("a/", "b/")) and len(value) > 2:
        return value[2:]
    return value


def _clean_diff_header_path(header: str) -> str:
    _, _, value = header.partition(" ")
    value = value.strip().split("\t", 1)[0]
    return _clean_path(value)


def _rel_display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _detect_lint_tool() -> str | None:
    for tool in ("ruff", "flake8", "pylint"):
        if shutil.which(tool):
            return tool
    return None


def _lint_command(tool: str, files: list[str]) -> list[str]:
    if tool == "ruff":
        return ["ruff", "check", *files]
    if tool == "flake8":
        return ["flake8", *files]
    if tool == "pylint":
        return ["pylint", *files]
    raise ValueError(f"Unsupported lint tool: {tool}")


def _run_command_check(name: str, command: list[str], cwd: Path) -> VerificationCheck:
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd, check=False)
    output = result.stdout
    if result.stderr:
        output = f"{output}\n{result.stderr}".strip()
    return VerificationCheck(
        name=name,
        ran=True,
        success=result.returncode == 0,
        output=output.strip(),
    )


def _syntax_check(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        try:
            ast.parse(read_text_file(path), filename=str(path))
        except SyntaxError as exc:
            line = exc.lineno or 0
            column = exc.offset or 0
            errors.append(f"{path}: line {line}, col {column}: {exc.msg}")
    return errors


def _detect_newline(content: str) -> str:
    if "\r\n" in content:
        return "\r\n"
    return "\n"


def _assert_line_match(actual_line: str, expected_line: str) -> None:
    actual = _strip_line_ending(actual_line)
    if actual != expected_line:
        raise ValueError(f"Unified diff context mismatch: expected {expected_line!r}, got {actual!r}")


def _strip_line_ending(line: str) -> str:
    return line.rstrip("\r\n")


def _normalize_line(line: str) -> str:
    return " ".join(line.strip().split())


__all__ = [
    "EditBatchResult",
    "EditFormat",
    "EditOperation",
    "UndoBatchInfo",
    "UndoResult",
    "VerificationCheck",
    "VerificationReport",
    "apply_edit_operations",
    "apply_edit_text",
    "list_undo_history",
    "parse_structured_edits",
    "run_verification",
    "undo_last_edit_batch",
]
