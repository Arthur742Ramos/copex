from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from copex.approval import (
    ApprovalAction,
    ApprovalMode,
    ApprovalWorkflow,
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
