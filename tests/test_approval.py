from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from copex.approval import (
    ApprovalWorkflow,
    AuditLogger,
    build_preview,
    extract_proposed_changes,
    summarize_changes,
)


def _quiet_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, color_system=None)


def test_diff_preview_generation(tmp_path: Path) -> None:
    file_path = tmp_path / "src" / "foo.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("x = 1\n", encoding="utf-8")

    proposals = extract_proposed_changes(
        "write_file",
        {"path": "src/foo.py", "content": "x = 2\n"},
        cwd=tmp_path,
    )
    assert len(proposals) == 1

    preview = build_preview(proposals[0])
    stats = summarize_changes([preview])

    assert preview.file_path == "src/foo.py"
    assert "--- a/src/foo.py" in preview.unified_diff
    assert "+++ b/src/foo.py" in preview.unified_diff
    assert preview.lines_added == 1
    assert preview.lines_removed == 1
    assert stats.files_changed == 1
    assert stats.lines_added == 1
    assert stats.lines_removed == 1


def test_approval_flow_supports_cherry_pick(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A1\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B1\n", encoding="utf-8")
    answers = iter(["y", "n"])

    workflow = ApprovalWorkflow(
        mode="approve",
        input_func=lambda _prompt: next(answers),
        console=_quiet_console(),
    )

    reviewed = workflow.review_tool_call(
        "write_file",
        {
            "changes": [
                {"path": "a.txt", "content": "A2\n"},
                {"path": "b.txt", "content": "B2\n"},
            ]
        },
        cwd=tmp_path,
    )
    assert [item.apply_change for item in reviewed] == [True, False]

    # Simulate tool execution writing both files before post-decision enforcement.
    (tmp_path / "a.txt").write_text("A2\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B2\n", encoding="utf-8")
    workflow.apply_post_tool_decisions(reviewed)

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "A2\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "B1\n"


def test_dry_run_reverts_changes(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('before')\n", encoding="utf-8")
    workflow = ApprovalWorkflow(mode="dry-run", console=_quiet_console())

    reviewed = workflow.review_tool_call(
        "write_file",
        {"path": "app.py", "content": "print('after')\n"},
        cwd=tmp_path,
    )
    assert len(reviewed) == 1
    assert reviewed[0].apply_change is False
    assert reviewed[0].dry_run is True

    (tmp_path / "app.py").write_text("print('after')\n", encoding="utf-8")
    workflow.apply_post_tool_decisions(reviewed)
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "print('before')\n"


def test_audit_logging_and_query(tmp_path: Path) -> None:
    audit_path = tmp_path / ".copex" / "audit.log"
    logger = AuditLogger(audit_path)
    workflow = ApprovalWorkflow(
        mode="auto-approve",
        audit_enabled=True,
        audit_logger=logger,
        console=_quiet_console(),
    )

    reviewed = workflow.review_tool_call(
        "write_file",
        {"path": "src/foo.py", "content": "print('x')\n"},
        cwd=tmp_path,
    )
    assert reviewed

    all_entries = logger.query(last=10)
    assert len(all_entries) == 1
    assert all_entries[0].mode == "auto-approve"
    assert all_entries[0].file == "src/foo.py"

    file_entries = logger.query(last=10, file="src/foo.py")
    assert len(file_entries) == 1
    assert file_entries[0].diff_summary
