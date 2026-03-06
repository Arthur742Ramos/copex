from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from copex.approval import (
    ApprovalAction,
    ApprovalGate,
    ApprovalMode,
    ApprovalWorkflow,
    AuditLogger,
    ChangeStats,
    DiffPreview,
    ProposedFileChange,
    RiskAssessor,
    build_preview,
    default_audit_log_path,
    extract_proposed_changes,
    normalize_approval_mode,
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


def test_diff_preview_colorize_marks_added_and_removed_lines() -> None:
    renderer = DiffPreview()
    colored = renderer.colorize("--- a/foo.py\n+++ b/foo.py\n@@\n-old\n+new\n")
    assert colored.plain.endswith("+new\n")
    assert any(
        colored.plain[span.start : span.end] == "+new\n" and "green" in str(span.style)
        for span in colored.spans
    )
    assert any(
        colored.plain[span.start : span.end] == "-old\n" and "red" in str(span.style)
        for span in colored.spans
    )


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
    workflow.log_execution_event(reviewed, success=True, result="ok")

    all_entries = logger.query(last=10)
    assert len(all_entries) == 2
    assert all_entries[0].mode == "auto-approve"
    assert all_entries[0].file == "src/foo.py"
    assert all_entries[0].event == "decision"
    assert all_entries[1].event == "execution"
    assert all_entries[1].result

    file_entries = logger.query(last=10, file="src/foo.py")
    assert len(file_entries) == 2
    assert file_entries[0].diff_summary


def test_approval_gate_enables_audit_alias(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / ".copex" / "audit.log")
    gate = ApprovalGate(
        mode="approve",
        audit=True,
        input_func=lambda _prompt: "y",
        audit_logger=logger,
        console=_quiet_console(),
    )
    reviewed = gate.review(
        "write_file", {"path": "src/ok.py", "content": "print('ok')\n"}, cwd=tmp_path
    )
    assert reviewed and reviewed[0].apply_change is True
    gate.log_execution_event(reviewed, success=True, result="ok")
    entries = logger.query(last=10)
    assert len(entries) == 2
    assert entries[0].mode == "manual"


def test_audit_schema_includes_decision_risk_fingerprint_result_and_error(tmp_path: Path) -> None:
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
        {"path": "secrets.txt", "content": "api_key = 'x'\n"},
        cwd=tmp_path,
    )
    workflow.log_execution_event(reviewed, success=False, error="tool failed")

    entries = logger.query(last=10)
    assert len(entries) == 2
    for entry in entries:
        payload = entry.to_dict()
        required = {
            "timestamp",
            "mode",
            "event",
            "decision",
            "risk",
            "change_fingerprint",
            "dry_run",
            "result",
            "error",
        }
        assert required.issubset(payload.keys())
        assert payload["decision"] == "approve"
        assert isinstance(payload["risk"], dict)
        assert payload["change_fingerprint"]

    assert entries[0].event == "decision"
    assert entries[0].error is None
    assert entries[1].event == "execution"
    assert entries[1].error == "tool failed"


def test_audit_logs_reject_and_execution_skip_path(tmp_path: Path) -> None:
    audit_path = tmp_path / ".copex" / "audit.log"
    logger = AuditLogger(audit_path)
    workflow = ApprovalWorkflow(
        mode="deny-all",
        audit_enabled=True,
        audit_logger=logger,
        console=_quiet_console(),
    )

    reviewed = workflow.review_tool_call(
        "write_file",
        {"path": "blocked.txt", "content": "nope\n"},
        cwd=tmp_path,
    )
    assert reviewed and reviewed[0].apply_change is False
    workflow.log_execution_event(reviewed, success=True, result="skipped")

    entries = logger.query(last=10)
    assert len(entries) == 2
    assert entries[0].decision == "reject"
    assert entries[1].decision == "reject"
    assert entries[1].event == "execution"
    assert entries[1].result and "skipped" in entries[1].result


def test_normalize_approval_mode_aliases() -> None:
    assert normalize_approval_mode("auto") == ApprovalMode.AUTO_APPROVE
    assert normalize_approval_mode("approve") == ApprovalMode.MANUAL
    assert normalize_approval_mode("deny") == ApprovalMode.DENY_ALL
    assert normalize_approval_mode("policy") == ApprovalMode.POLICY_BASED
    assert normalize_approval_mode("dry") == ApprovalMode.DRY_RUN
    assert normalize_approval_mode(ApprovalMode.APPROVE) == ApprovalMode.MANUAL


def test_approval_mode_manual_is_distinct() -> None:
    assert ApprovalMode.MANUAL.value == "manual"
    assert ApprovalMode.MANUAL is not ApprovalMode.APPROVE


def test_risk_escalation_for_large_config_delete(tmp_path: Path) -> None:
    before = "".join(f"line-{idx}\n" for idx in range(150))
    preview = build_preview(
        ProposedFileChange(
            path=tmp_path / "pyproject.toml",
            display_path="pyproject.toml",
            existed_before=True,
            before_content=before,
            after_content="",
        )
    )

    assert preview.operation == "delete"
    assert preview.risk.severity == "high"
    assert "delete-operation" in preview.risk.reasons
    assert "large-deletion" in preview.risk.reasons
    assert "config-file-change" in preview.risk.reasons


def test_risk_assessor_flags_security_content() -> None:
    risk = RiskAssessor().assess(
        path="auth/config.py",
        lines_removed=0,
        operation="edit",
        after_content="token = 'x'\n",
    )
    assert risk.severity == "high"
    assert "security-sensitive-file" in risk.reasons
    assert "secret-like-content" in risk.reasons


def test_default_audit_log_path_stays_in_project(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert default_audit_log_path() == tmp_path / ".copex" / "audit.log"


def test_default_audit_log_path_rejects_parent_traversal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="project directory"):
        default_audit_log_path(Path(".."))


def test_change_stats_helper_matches_summary(tmp_path: Path) -> None:
    preview = build_preview(
        ProposedFileChange(
            path=tmp_path / "a.py",
            display_path="a.py",
            existed_before=True,
            before_content="x = 1\n",
            after_content="x = 2\n",
        )
    )
    stats = ChangeStats.from_previews([preview])
    assert stats.files_changed == 1
    assert stats.lines_added == 1
    assert stats.lines_removed == 1


def test_policy_based_mode_uses_risk_to_require_input(tmp_path: Path) -> None:
    secure_file = tmp_path / "auth" / "token.txt"
    secure_file.parent.mkdir(parents=True, exist_ok=True)
    secure_file.write_text("token = 'old'\n", encoding="utf-8")
    answers = iter(["n"])
    workflow = ApprovalWorkflow(
        mode="policy-based",
        policy_func=lambda preview: "manual" if preview.risk.severity == "high" else "approve",
        input_func=lambda _prompt: next(answers),
        console=_quiet_console(),
    )

    reviewed = workflow.review_tool_call(
        "write_file",
        {"path": "auth/token.txt", "content": "api_key = 'new'\n"},
        cwd=tmp_path,
    )

    assert len(reviewed) == 1
    assert reviewed[0].preview.risk.severity == "high"
    assert reviewed[0].action == ApprovalAction.REJECT
    assert reviewed[0].apply_change is False


def test_invalid_manual_prompt_input_defaults_to_reject(tmp_path: Path) -> None:
    workflow = ApprovalWorkflow(
        mode="approve",
        input_func=lambda _prompt: "wat",
        console=_quiet_console(),
    )
    workflow._decision_prompt = None

    preview = build_preview(
        ProposedFileChange(
            path=tmp_path / "src" / "foo.py",
            display_path="src/foo.py",
            existed_before=False,
            before_content="",
            after_content="print('x')\n",
        )
    )
    stats = summarize_changes([preview])

    assert workflow._prompt_action(preview, stats) == ApprovalAction.REJECT
