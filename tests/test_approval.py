from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from copex.approval import (
    ApprovalAction,
    ApprovalMode,
    ApprovalWorkflow,
    ProposedFileChange,
    build_preview,
    normalize_approval_mode,
)
from copex.config import CopexConfig


def _quiet_console() -> Console:
    return Console(file=StringIO(), force_terminal=False, color_system=None)


def _review_single(
    tmp_path: Path,
    *,
    mode: str,
    path: str = "sample.txt",
    content: str = "new-content",
    input_func=None,
    policy_func=None,
):
    workflow = ApprovalWorkflow(
        mode=mode,
        console=_quiet_console(),
        input_func=input_func,
        policy_func=policy_func,
    )
    reviewed = workflow.review_tool_call(
        "write_file",
        {"path": path, "content": content},
        cwd=tmp_path,
    )
    assert len(reviewed) == 1
    return workflow, reviewed[0]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ApprovalMode.AUTO_APPROVE),
        ("auto-approve", ApprovalMode.AUTO_APPROVE),
        ("approve", ApprovalMode.MANUAL),
        ("manual", ApprovalMode.MANUAL),
        ("deny-all", ApprovalMode.DENY_ALL),
        ("policy", ApprovalMode.POLICY_BASED),
        ("dry-run", ApprovalMode.DRY_RUN),
    ],
)
def test_normalize_approval_mode_supports_extended_modes(raw, expected):
    assert normalize_approval_mode(raw) == expected


@pytest.mark.parametrize(
    "mode",
    ["auto-approve", "approve", "manual", "deny-all", "policy-based", "dry-run"],
)
def test_config_accepts_extended_approval_modes(mode):
    config = CopexConfig(approval_mode=mode)
    assert config.approval_mode == mode


def test_auto_approve_mode_approves_change(tmp_path: Path):
    _, decision = _review_single(tmp_path, mode="auto-approve")
    assert decision.action == ApprovalAction.APPROVE
    assert decision.apply_change is True
    assert decision.dry_run is False


def test_manual_mode_uses_user_reject_decision(tmp_path: Path):
    _, decision = _review_single(tmp_path, mode="manual", input_func=lambda _: "n")
    assert decision.action == ApprovalAction.REJECT
    assert decision.apply_change is False


def test_manual_mode_approve_all_sets_state_for_followup_changes(tmp_path: Path):
    prompt_calls: list[str] = []

    def input_func(prompt: str) -> str:
        prompt_calls.append(prompt)
        return "a"

    workflow = ApprovalWorkflow(mode="manual", console=_quiet_console(), input_func=input_func)
    reviewed = workflow.review_tool_call(
        "write_file",
        {
            "changes": [
                {"path": "first.txt", "content": "one"},
                {"path": "second.txt", "content": "two"},
            ]
        },
        cwd=tmp_path,
    )
    assert workflow.state.approve_all is True
    assert len(prompt_calls) == 1
    assert [item.action for item in reviewed] == [ApprovalAction.APPROVE, ApprovalAction.APPROVE]


def test_deny_all_mode_rejects_without_prompt(tmp_path: Path):
    _, decision = _review_single(
        tmp_path,
        mode="deny-all",
        input_func=lambda _: pytest.fail("deny-all should not prompt"),
    )
    assert decision.action == ApprovalAction.REJECT
    assert decision.apply_change is False
    assert decision.dry_run is False


def test_policy_based_mode_uses_policy_decisions(tmp_path: Path):
    def policy(preview):
        return not preview.risk_flags

    _, safe = _review_single(
        tmp_path,
        mode="policy-based",
        path="notes.txt",
        policy_func=policy,
        input_func=lambda _: pytest.fail("policy approve/reject should not prompt"),
    )
    _, risky = _review_single(
        tmp_path,
        mode="policy-based",
        path="auth/token.txt",
        policy_func=policy,
        input_func=lambda _: pytest.fail("policy approve/reject should not prompt"),
    )
    assert safe.action == ApprovalAction.APPROVE
    assert risky.action == ApprovalAction.REJECT


def test_dry_run_mode_rejects_and_sets_dry_run_flag(tmp_path: Path):
    _, decision = _review_single(
        tmp_path,
        mode="dry-run",
        input_func=lambda _: pytest.fail("dry-run should not prompt"),
    )
    assert decision.action == ApprovalAction.REJECT
    assert decision.apply_change is False
    assert decision.dry_run is True


def test_policy_mode_requires_policy_function():
    with pytest.raises(ValueError, match="policy_func"):
        ApprovalWorkflow(mode="policy-based")


@pytest.fixture
def edit_change_fixture(tmp_path: Path) -> ProposedFileChange:
    file_path = tmp_path / "sample.txt"
    return ProposedFileChange(
        path=file_path,
        display_path="sample.txt",
        existed_before=True,
        before_content="old\n",
        after_content="new\n",
    )


def test_build_preview_snapshot_for_edit_change(edit_change_fixture: ProposedFileChange):
    preview = build_preview(edit_change_fixture)
    expected_diff = "\n".join(
        [
            "--- a/sample.txt",
            "+++ b/sample.txt",
            "@@ -1 +1 @@",
            "-old",
            "+new",
        ]
    )
    assert preview.operation == "edit"
    assert preview.summary == "edit (+1/-1)"
    assert preview.unified_diff == expected_diff
    assert preview.metadata == {
        "path": "sample.txt",
        "operation": "edit",
        "existed_before": True,
        "exists_after": True,
        "before_lines": 1,
        "after_lines": 1,
        "before_bytes": 4,
        "after_bytes": 4,
    }
    assert preview.diff_truncated is False


def test_build_preview_operation_and_metadata_for_add_and_delete(tmp_path: Path):
    add_preview = build_preview(
        ProposedFileChange(
            path=tmp_path / "new.txt",
            display_path="new.txt",
            existed_before=False,
            before_content="",
            after_content="alpha\nbeta\n",
        )
    )
    delete_preview = build_preview(
        ProposedFileChange(
            path=tmp_path / "old.txt",
            display_path="old.txt",
            existed_before=True,
            before_content="alpha\nbeta\n",
            after_content="",
        )
    )

    assert add_preview.operation == "add"
    assert add_preview.summary == "add (+2/-0)"
    assert add_preview.metadata["exists_after"] is True
    assert delete_preview.operation == "delete"
    assert delete_preview.summary == "delete (+0/-2)"
    assert delete_preview.metadata["exists_after"] is False


def test_build_preview_truncates_large_diff_safely(tmp_path: Path):
    before = "\n".join(f"old-{i}" for i in range(80)) + "\n"
    after = "\n".join(f"new-{i}" for i in range(80)) + "\n"
    preview = build_preview(
        ProposedFileChange(
            path=tmp_path / "large.txt",
            display_path="large.txt",
            existed_before=True,
            before_content=before,
            after_content=after,
        ),
        max_diff_lines=12,
        max_diff_chars=2_000,
    )

    assert preview.diff_truncated is True
    assert "diff truncated" in preview.unified_diff
    assert preview.unified_diff.endswith(
        f"... [diff truncated: 12/{preview.diff_lines_total} lines shown]"
    )
    assert preview.diff_lines_total > len(preview.unified_diff.splitlines())
