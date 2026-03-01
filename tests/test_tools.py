from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from copex.approval import ApprovalWorkflow, AuditLogger
from copex.tools import ParallelToolConfig, ToolRegistry


@dataclass(frozen=True)
class _FakePreview:
    file_path: str


@dataclass(frozen=True)
class _FakeProposal:
    display_path: str


@dataclass(frozen=True)
class _FakeReview:
    apply_change: bool
    preview: _FakePreview
    proposal: _FakeProposal


class _FakeApprovalWorkflow:
    def __init__(self, reviews: list[_FakeReview]):
        self._reviews = reviews
        self.review_calls: list[tuple[str, dict[str, Any], Path | None]] = []
        self.post_calls: list[list[_FakeReview]] = []
        self.execution_calls: list[dict[str, Any]] = []

    def review_tool_call(
        self, name: str, params: dict[str, Any], *, cwd: Path | None = None
    ) -> list[_FakeReview]:
        self.review_calls.append((name, params, cwd))
        return list(self._reviews)

    def apply_post_tool_decisions(self, reviewed: list[_FakeReview]) -> list[str]:
        self.post_calls.append(list(reviewed))
        return [f"post:{item.preview.file_path}" for item in reviewed]

    def log_execution_event(
        self,
        reviewed: list[_FakeReview],
        *,
        success: bool,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        self.execution_calls.append(
            {
                "reviewed": list(reviewed),
                "success": success,
                "result": result,
                "error": error,
            }
        )


def _quiet_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, color_system=None)


@pytest.mark.asyncio
async def test_execute_skips_mutating_tool_when_all_reviews_rejected() -> None:
    executed = False

    async def write_file(path: str, content: str) -> str:
        nonlocal executed
        executed = True
        return f"{path}:{content}"

    workflow = _FakeApprovalWorkflow(
        [
            _FakeReview(
                apply_change=False,
                preview=_FakePreview(file_path="a.txt"),
                proposal=_FakeProposal(display_path="a.txt"),
            )
        ]
    )
    registry = ToolRegistry(ParallelToolConfig(approval_workflow=workflow, retry_on_error=False))
    registry.add_tool("write_file", write_file)

    result = await registry.execute("write_file", {"path": "a.txt", "content": "x"})

    assert result.success is True
    assert executed is False
    assert "Skipped by approval workflow: a.txt" in str(result.result)
    assert workflow.post_calls == []
    assert len(workflow.execution_calls) == 1
    assert workflow.execution_calls[0]["success"] is True


@pytest.mark.asyncio
async def test_execute_runs_mutating_tool_when_review_approved() -> None:
    executed = False

    async def write_file(path: str, content: str) -> str:
        nonlocal executed
        executed = True
        return f"{path}:{content}"

    workflow = _FakeApprovalWorkflow(
        [
            _FakeReview(
                apply_change=True,
                preview=_FakePreview(file_path="a.txt"),
                proposal=_FakeProposal(display_path="a.txt"),
            )
        ]
    )
    registry = ToolRegistry(ParallelToolConfig(approval_workflow=workflow, retry_on_error=False))
    registry.add_tool("write_file", write_file)

    result = await registry.execute("write_file", {"path": "a.txt", "content": "x"})

    assert result.success is True
    assert executed is True
    assert "a.txt:x" in str(result.result)
    assert "post:a.txt" in str(result.result)
    assert len(workflow.post_calls) == 1
    assert len(workflow.post_calls[0]) == 1
    assert len(workflow.execution_calls) == 1
    assert workflow.execution_calls[0]["success"] is True


@pytest.mark.asyncio
async def test_execute_filters_changes_to_approved_subset_before_execution() -> None:
    captured_changes: list[dict[str, Any]] = []

    async def write_many(changes: list[dict[str, Any]]) -> str:
        captured_changes.extend(changes)
        return "applied"

    workflow = _FakeApprovalWorkflow(
        [
            _FakeReview(
                apply_change=True,
                preview=_FakePreview(file_path="a.txt"),
                proposal=_FakeProposal(display_path="a.txt"),
            ),
            _FakeReview(
                apply_change=False,
                preview=_FakePreview(file_path="b.txt"),
                proposal=_FakeProposal(display_path="b.txt"),
            ),
        ]
    )
    registry = ToolRegistry(ParallelToolConfig(approval_workflow=workflow, retry_on_error=False))
    registry.add_tool("write_many", write_many)

    result = await registry.execute(
        "write_many",
        {
            "changes": [
                {"path": "a.txt", "content": "A"},
                {"path": "b.txt", "content": "B"},
            ]
        },
    )

    assert result.success is True
    assert captured_changes == [{"path": "a.txt", "content": "A"}]
    assert "skipped by approval: b.txt" in str(result.result)
    assert len(workflow.post_calls) == 1
    assert [item.proposal.display_path for item in workflow.post_calls[0]] == ["a.txt"]
    assert len(workflow.execution_calls) == 1
    assert workflow.execution_calls[0]["success"] is True


@pytest.mark.asyncio
async def test_execute_logs_execution_failure_event() -> None:
    async def write_file(path: str, content: str) -> str:
        raise RuntimeError(f"boom:{path}:{content}")

    workflow = _FakeApprovalWorkflow(
        [
            _FakeReview(
                apply_change=True,
                preview=_FakePreview(file_path="a.txt"),
                proposal=_FakeProposal(display_path="a.txt"),
            )
        ]
    )
    registry = ToolRegistry(ParallelToolConfig(approval_workflow=workflow, retry_on_error=False))
    registry.add_tool("write_file", write_file)

    result = await registry.execute("write_file", {"path": "a.txt", "content": "x"})

    assert result.success is False
    assert "boom:a.txt:x" in (result.error or "")
    assert len(workflow.execution_calls) == 1
    assert workflow.execution_calls[0]["success"] is False
    assert "boom:a.txt:x" in str(workflow.execution_calls[0]["error"])


@pytest.mark.asyncio
async def test_execute_dry_run_returns_preview_and_avoids_side_effects(tmp_path: Path) -> None:
    target_file = tmp_path / "safe.txt"
    target_file.write_text("before\n", encoding="utf-8")
    audit_path = tmp_path / ".copex" / "audit.log"
    audit_logger = AuditLogger(audit_path)
    executed = False

    async def write_file(path: str, content: str) -> str:
        nonlocal executed
        executed = True
        (tmp_path / path).write_text(content, encoding="utf-8")
        return "written"

    workflow = ApprovalWorkflow(
        mode="dry-run",
        audit_enabled=True,
        audit_logger=audit_logger,
        console=_quiet_console(),
    )
    registry = ToolRegistry(ParallelToolConfig(approval_workflow=workflow, retry_on_error=False))
    registry.add_tool("write_file", write_file)

    result = await registry.execute("write_file", {"path": "safe.txt", "content": "after\n"})

    assert result.success is True
    assert executed is False
    assert target_file.read_text(encoding="utf-8") == "before\n"
    assert isinstance(result.result, dict)
    assert result.result.get("dry_run") is True
    assert "Dry-run preview (no side effects): safe.txt" == result.result.get("message")
    changes = result.result.get("changes")
    assert isinstance(changes, list)
    assert changes and changes[0]["file"] == "safe.txt"
    assert "summary" in changes[0]
    assert "risk" in changes[0]
    entries = audit_logger.query(last=5)
    assert len(entries) == 2
    assert [entry.event for entry in entries] == ["decision", "execution"]
    assert entries[0].dry_run is True
    assert entries[1].dry_run is True


@pytest.mark.asyncio
async def test_execute_parallel_empty_calls_returns_empty_list() -> None:
    registry = ToolRegistry(ParallelToolConfig(retry_on_error=False))
    assert await registry.execute_parallel([]) == []


@pytest.mark.asyncio
async def test_execute_parallel_fail_fast_marks_remaining_calls_cancelled() -> None:
    async def fail_tool() -> str:
        raise RuntimeError("boom")

    async def slow_tool() -> str:
        await asyncio.sleep(30)
        return "slow"

    registry = ToolRegistry(
        ParallelToolConfig(
            fail_fast=True,
            retry_on_error=False,
            max_concurrent=2,
        )
    )
    registry.add_tool("fail", fail_tool)
    registry.add_tool("slow", slow_tool)

    results = await registry.execute_parallel(
        [
            ("fail", {}),
            ("slow", {}),
        ]
    )

    assert len(results) == 2
    assert results[0].name == "fail"
    assert results[0].success is False
    assert "boom" in (results[0].error or "")
    assert results[1].name == "slow"
    assert results[1].success is False
    assert results[1].error == "Cancelled due to fail_fast"
