"""Tests for visualization module."""

from __future__ import annotations

import pytest

from copex.plan import Plan, PlanStep, StepStatus
from copex.visualization import (
    DependencyEdge,
    ParallelGroup,
    StatusIndicator,
    analyze_dependencies,
    find_parallel_groups,
    render_ascii,
    render_mermaid,
    visualize_plan,
)


def _make_plan(*titles: str) -> Plan:
    """Create a simple plan with the given step titles."""
    steps = [
        PlanStep(
            number=i + 1,
            description=title,
        )
        for i, title in enumerate(titles)
    ]
    return Plan(task="Test goal", steps=steps)


class TestStatusIndicator:
    def test_pending_attributes(self) -> None:
        assert StatusIndicator.PENDING.symbol == "○"
        assert StatusIndicator.PENDING.label == "pending"

    def test_completed_attributes(self) -> None:
        assert StatusIndicator.COMPLETED.symbol == "●"
        assert StatusIndicator.COMPLETED.label == "completed"

    def test_failed_attributes(self) -> None:
        assert StatusIndicator.FAILED.symbol == "✗"
        assert StatusIndicator.FAILED.label == "failed"


class TestDependencyEdge:
    def test_basic_edge(self) -> None:
        edge = DependencyEdge(from_step=1, to_step=2)
        assert edge.from_step == 1
        assert edge.to_step == 2


class TestAnalyzeDependencies:
    def test_sequential_plan(self) -> None:
        plan = _make_plan("Step 1", "Step 2", "Step 3")
        edges = analyze_dependencies(plan)

        # Sequential steps: 1->2, 2->3
        assert len(edges) == 2
        assert edges[0].from_step == 1
        assert edges[0].to_step == 2
        assert edges[1].from_step == 2
        assert edges[1].to_step == 3

    def test_single_step_plan(self) -> None:
        plan = _make_plan("Only Step")
        edges = analyze_dependencies(plan)
        assert len(edges) == 0


class TestFindParallelGroups:
    def test_sequential_no_parallel(self) -> None:
        plan = _make_plan("A", "B", "C")
        groups = find_parallel_groups(plan)

        # Each step is at its own level
        assert len(groups) >= 1

    def test_empty_plan(self) -> None:
        plan = Plan(task="Empty", steps=[])
        groups = find_parallel_groups(plan)
        assert groups == []


class TestRenderAscii:
    def test_basic_ascii_rendering(self) -> None:
        plan = _make_plan("Setup", "Build", "Test")
        output = render_ascii(plan)

        assert isinstance(output, str)
        assert "Setup" in output or "Step 1" in output

    def test_empty_plan_ascii(self) -> None:
        plan = Plan(task="Empty", steps=[])
        output = render_ascii(plan)
        assert isinstance(output, str)


class TestRenderMermaid:
    def test_basic_mermaid_rendering(self) -> None:
        plan = _make_plan("Setup", "Build")
        output = render_mermaid(plan)

        assert isinstance(output, str)
        assert "graph" in output.lower() or "flowchart" in output.lower()

    def test_mermaid_has_steps(self) -> None:
        plan = _make_plan("Alpha", "Beta")
        output = render_mermaid(plan)

        # Should contain step references
        assert "1" in output or "Alpha" in output


class TestVisualizePlan:
    def test_default_format(self) -> None:
        plan = _make_plan("A", "B")
        output = visualize_plan(plan)
        assert isinstance(output, str)

    def test_ascii_format(self) -> None:
        plan = _make_plan("A", "B")
        output = visualize_plan(plan, format="ascii")
        assert isinstance(output, str)

    def test_mermaid_format(self) -> None:
        plan = _make_plan("A", "B")
        output = visualize_plan(plan, format="mermaid")
        assert isinstance(output, str)
        assert "graph" in output.lower() or "flowchart" in output.lower()
