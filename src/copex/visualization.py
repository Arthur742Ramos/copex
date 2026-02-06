"""
Plan Visualization - Display plan dependency graphs and structure.

Provides ASCII and Mermaid diagram output for plan steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from copex.plan import Plan, StepStatus


class StatusIndicator(Enum):
    """Visual status indicators for diagram rendering."""

    PENDING = ("○", "#9e9e9e", "pending")      # Gray
    RUNNING = ("◐", "#2196f3", "running")      # Blue
    COMPLETED = ("●", "#4caf50", "completed")  # Green
    FAILED = ("✗", "#f44336", "failed")        # Red
    SKIPPED = ("◌", "#ff9800", "skipped")      # Orange
    BLOCKED = ("⊘", "#9c27b0", "blocked")      # Purple

    def __init__(self, symbol: str, color: str, label: str) -> None:
        self.symbol = symbol
        self.color = color
        self.label = label


@dataclass
class DependencyEdge:
    """Represents a dependency edge between steps."""
    from_step: int  # Step index (1-indexed)
    to_step: int    # Step index (1-indexed)


@dataclass
class ParallelGroup:
    """A group of steps that can run in parallel."""
    level: int            # Execution level (topological depth)
    step_numbers: list[int]  # Step numbers in this group
    can_parallelize: bool    # True if multiple steps at this level


def analyze_dependencies(plan: Plan) -> list[DependencyEdge]:
    """Analyze step dependencies from plan structure.

    Returns a list of dependency edges where an edge (a, b) means
    step b depends on step a.
    """
    edges: list[DependencyEdge] = []

    for i, step in enumerate(plan.steps, 1):
        # Check for explicit dependencies in step metadata
        if hasattr(step, "depends_on") and step.depends_on:
            for dep in step.depends_on:
                edges.append(DependencyEdge(from_step=dep, to_step=i))
        # Default: sequential dependency (each step depends on previous)
        elif i > 1:
            edges.append(DependencyEdge(from_step=i - 1, to_step=i))

    return edges


def find_parallel_groups(plan: Plan) -> list[ParallelGroup]:
    """Find groups of steps that can run in parallel.

    Uses topological sort levels to identify steps that have no
    dependencies between them and can execute concurrently.

    Args:
        plan: The plan to analyze

    Returns:
        List of ParallelGroup objects, ordered by execution level.
        Each group contains step numbers that can run in parallel.
    """
    if not plan.steps:
        return []

    edges = analyze_dependencies(plan)
    n = len(plan.steps)

    # Build dependency graph
    deps: dict[int, set[int]] = {i: set() for i in range(1, n + 1)}
    for edge in edges:
        deps[edge.to_step].add(edge.from_step)

    # Calculate topological levels (longest path from any root)
    levels: dict[int, int] = {}

    def calc_level(step: int, visited: set[int]) -> int:
        if step in levels:
            return levels[step]
        if step in visited:
            # Cycle detected - assign level 0
            return 0
        visited.add(step)

        if not deps[step]:
            levels[step] = 0
        else:
            levels[step] = 1 + max(calc_level(d, visited) for d in deps[step])

        visited.discard(step)
        return levels[step]

    for i in range(1, n + 1):
        calc_level(i, set())

    # Group by level
    level_groups: dict[int, list[int]] = {}
    for step, level in levels.items():
        level_groups.setdefault(level, []).append(step)

    # Build ParallelGroup objects
    groups: list[ParallelGroup] = []
    for level in sorted(level_groups.keys()):
        steps = sorted(level_groups[level])
        groups.append(ParallelGroup(
            level=level,
            step_numbers=steps,
            can_parallelize=len(steps) > 1,
        ))

    return groups


def _get_status_indicator(step) -> StatusIndicator:
    """Get the status indicator for a step."""
    if hasattr(step, "status"):
        status_str = str(step.status.value).lower() if hasattr(step.status, "value") else str(step.status).lower()
        for indicator in StatusIndicator:
            if indicator.label == status_str:
                return indicator
    return StatusIndicator.PENDING


def render_dag_with_status(
    plan: Plan,
    *,
    format: str = "mermaid",
    show_parallel: bool = True,
    show_legend: bool = True,
) -> str:
    """Render a DAG visualization with status indicators and parallel group detection.

    Args:
        plan: The plan to visualize
        format: Output format ("mermaid", "ascii", or "dot")
        show_parallel: Highlight parallel groups
        show_legend: Include a legend for status indicators

    Returns:
        Formatted diagram string
    """
    if format == "mermaid":
        return _render_dag_mermaid(plan, show_parallel, show_legend)
    elif format == "ascii":
        return _render_dag_ascii(plan, show_parallel, show_legend)
    elif format == "dot":
        return _render_dag_dot(plan, show_parallel)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'mermaid', 'ascii', or 'dot'")


def _render_dag_mermaid(plan: Plan, show_parallel: bool, show_legend: bool) -> str:
    """Render plan as Mermaid diagram with status and parallel groups."""
    if not plan.steps:
        return "graph TD\n    empty[Empty Plan]"

    lines: list[str] = []
    lines.append("graph TD")

    # Title
    lines.append(f"    title[\"📋 {_escape_mermaid(plan.task)}\"]")
    lines.append("    style title fill:#e1f5fe,stroke:#01579b,stroke-width:2px")
    lines.append("")

    # Get parallel groups for highlighting
    parallel_groups = find_parallel_groups(plan) if show_parallel else []
    step_to_group: dict[int, int] = {}
    for group in parallel_groups:
        if group.can_parallelize:
            for step_num in group.step_numbers:
                step_to_group[step_num] = group.level

    # Define nodes with status-based styling
    for step in plan.steps:
        indicator = _get_status_indicator(step)
        desc = _escape_mermaid(_truncate(step.description, 35))
        node_id = f"step{step.number}"

        # Node shape based on status
        if indicator == StatusIndicator.COMPLETED:
            shape = f'(["{indicator.symbol} Step {step.number}: {desc}"])'
        elif indicator == StatusIndicator.FAILED:
            shape = f'{{{{"⚠️ Step {step.number}: {desc}"}}}}'
        elif indicator == StatusIndicator.RUNNING:
            shape = f'[/"{indicator.symbol} Step {step.number}: {desc}"/]'
        else:
            shape = f'["{indicator.symbol} Step {step.number}: {desc}"]'

        lines.append(f"    {node_id}{shape}")

    lines.append("")

    # Define edges
    edges = analyze_dependencies(plan)
    for edge in edges:
        from_status = _get_status_indicator(plan.steps[edge.from_step - 1])
        # Use dotted line if source step failed
        arrow = "-.->" if from_status == StatusIndicator.FAILED else "-->"
        lines.append(f"    step{edge.from_step} {arrow} step{edge.to_step}")

    # Style nodes based on status
    lines.append("")
    lines.append("    %% Status-based styling")
    for step in plan.steps:
        indicator = _get_status_indicator(step)
        lines.append(f"    style step{step.number} fill:{indicator.color},stroke:#333,stroke-width:2px,color:white")

    # Highlight parallel groups
    if show_parallel and parallel_groups:
        lines.append("")
        lines.append("    %% Parallel group subgraphs")
        for group in parallel_groups:
            if group.can_parallelize:
                lines.append(f"    subgraph parallel_{group.level}[\"⚡ Parallel Group L{group.level}\"]")
                for step_num in group.step_numbers:
                    lines.append(f"        step{step_num}")
                lines.append("    end")
                lines.append(f"    style parallel_{group.level} fill:#fff3e0,stroke:#ff9800,stroke-dasharray: 5 5")

    # Legend
    if show_legend:
        lines.append("")
        lines.append("    %% Legend")
        lines.append("    subgraph Legend")
        lines.append("        direction LR")
        for indicator in StatusIndicator:
            if indicator != StatusIndicator.BLOCKED:
                lines.append(f'        legend_{indicator.label}["{indicator.symbol} {indicator.label}"]')
                lines.append(f"        style legend_{indicator.label} fill:{indicator.color},color:white")
        lines.append("    end")

    return "\n".join(lines)


def _render_dag_ascii(plan: Plan, show_parallel: bool, show_legend: bool) -> str:
    """Render plan as ASCII DAG with status indicators."""
    if not plan.steps:
        return "(empty plan)"

    lines: list[str] = []
    width = 70

    # Header
    lines.append("╔" + "═" * (width - 2) + "╗")
    title = f"📋 {_truncate(plan.task, width - 10)}"
    lines.append(f"║ {title:<{width - 4}} ║")
    lines.append("╠" + "═" * (width - 2) + "╣")

    # Get parallel groups
    parallel_groups = find_parallel_groups(plan) if show_parallel else []
    step_to_group: dict[int, ParallelGroup] = {}
    for group in parallel_groups:
        for step_num in group.step_numbers:
            step_to_group[step_num] = group

    # Render steps by level
    current_level = -1
    for step in plan.steps:
        group = step_to_group.get(step.number)
        step_level = group.level if group else 0

        # Level separator for new parallel groups
        if show_parallel and group and group.can_parallelize:
            if step_level != current_level:
                current_level = step_level
                if group.step_numbers[0] == step.number:  # First in group
                    parallel_label = f"─── ⚡ Parallel Group (L{step_level}, {len(group.step_numbers)} steps) "
                    lines.append(f"║ {parallel_label:─<{width - 4}} ║")

        # Status indicator
        indicator = _get_status_indicator(step)

        # Parallel indicator
        parallel_mark = ""
        if show_parallel and group and group.can_parallelize:
            parallel_mark = f" [∥{len(group.step_numbers)}]"

        # Step line
        desc = _truncate(step.description, width - 18 - len(parallel_mark))
        step_line = f"{indicator.symbol} Step {step.number}: {desc}{parallel_mark}"
        lines.append(f"║   {step_line:<{width - 6}} ║")

        # Connection indicator
        if step.number < len(plan.steps):
            next_step = plan.steps[step.number]
            next_group = step_to_group.get(next_step.number)

            # Different connector based on relationship
            if group and next_group and group.level == next_group.level:
                # Same parallel group - horizontal connection
                lines.append(f"║   │ ├─────────────────────────────────{' ' * (width - 43)} ║")
            else:
                # Sequential - vertical connection
                lines.append(f"║   │{' ' * (width - 7)} ║")
                lines.append(f"║   ▼{' ' * (width - 7)} ║")

    lines.append("╚" + "═" * (width - 2) + "╝")

    # Legend
    if show_legend:
        lines.append("")
        legend_parts = [f"{ind.symbol} {ind.label}" for ind in StatusIndicator if ind != StatusIndicator.BLOCKED]
        lines.append("Legend: " + "  │  ".join(legend_parts))
        if show_parallel:
            lines.append("        [∥N] = N parallel steps")

    return "\n".join(lines)


def _render_dag_dot(plan: Plan, show_parallel: bool) -> str:
    """Render plan as GraphViz DOT format."""
    if not plan.steps:
        return 'digraph plan {\n    empty [label="Empty Plan"]\n}'

    lines: list[str] = []
    lines.append("digraph plan {")
    lines.append("    rankdir=TB;")
    lines.append("    node [shape=box, style=rounded];")
    lines.append("")

    # Get parallel groups for rank constraints
    parallel_groups = find_parallel_groups(plan) if show_parallel else []

    # Define nodes
    for step in plan.steps:
        indicator = _get_status_indicator(step)
        label = f"{indicator.symbol} Step {step.number}\\n{_escape_dot(step.description[:30])}"
        color = indicator.color
        lines.append(f'    step{step.number} [label="{label}", fillcolor="{color}", style="filled,rounded", fontcolor="white"];')

    lines.append("")

    # Define edges
    edges = analyze_dependencies(plan)
    for edge in edges:
        lines.append(f"    step{edge.from_step} -> step{edge.to_step};")

    # Add rank constraints for parallel groups
    if show_parallel:
        lines.append("")
        lines.append("    // Parallel group ranks")
        for group in parallel_groups:
            if group.can_parallelize:
                nodes = " ".join(f"step{n}" for n in group.step_numbers)
                lines.append(f"    {{ rank=same; {nodes} }}")

    lines.append("}")
    return "\n".join(lines)


def render_ascii(plan: Plan, *, show_parallel: bool = True) -> str:
    """Render plan as ASCII diagram.

    Args:
        plan: The plan to visualize
        show_parallel: Whether to show parallel step indicators

    Returns:
        ASCII art representation of the plan
    """
    if not plan.steps:
        return "(empty plan)"

    lines: list[str] = []
    lines.append("╔══════════════════════════════════════════════════════════════╗")
    lines.append(f"║  📋 Plan: {_truncate(plan.task, 50):<50} ║")
    lines.append("╠══════════════════════════════════════════════════════════════╣")

    parallel_groups = find_parallel_groups(plan) if show_parallel else []
    group_map: dict[int, int] = {}
    group_sizes: dict[int, int] = {}
    for group in parallel_groups:
        if group.can_parallelize:
            for step_idx in group.step_numbers:
                group_map[step_idx] = group.level
                group_sizes[group.level] = len(group.step_numbers)

    for i, step in enumerate(plan.steps, 1):
        # Status indicator
        indicator = _get_status_indicator(step)
        status = indicator.symbol

        # Parallel indicator
        parallel_info = ""
        if show_parallel and i in group_map:
            group_idx = group_map[i]
            size = group_sizes.get(group_idx, 1)
            if size > 1:
                parallel_info = f" [∥{size}]"

        # Step line
        desc = _truncate(step.description, 48 - len(parallel_info))
        lines.append(f"║  {status} Step {i}: {desc}{parallel_info:<{48 - len(desc)}} ║")

        # Connection line (except for last step)
        if i < len(plan.steps):
            lines.append("║     │                                                        ║")
            lines.append("║     ▼                                                        ║")

    lines.append("╚══════════════════════════════════════════════════════════════╝")

    # Legend
    if show_parallel:
        lines.append("")
        lines.append("Legend: ○ pending  ● done  ◐ running  ✗ failed  [∥N] = N parallel steps")

    return "\n".join(lines)


def render_mermaid(plan: Plan) -> str:
    """Render plan as Mermaid diagram.

    Args:
        plan: The plan to visualize

    Returns:
        Mermaid diagram definition
    """
    # Delegate to the enhanced version
    return render_dag_with_status(plan, format="mermaid", show_parallel=True, show_legend=False)


def render_simple_tree(plan: Plan) -> str:
    """Render plan as simple tree (for terminal output).

    Args:
        plan: The plan to visualize

    Returns:
        Simple tree representation
    """
    if not plan.steps:
        return "📋 (empty plan)"

    lines: list[str] = []
    lines.append(f"📋 {plan.task}")

    for i, step in enumerate(plan.steps, 1):
        is_last = i == len(plan.steps)
        prefix = "└──" if is_last else "├──"
        indicator = _get_status_indicator(step)
        lines.append(f"   {prefix} {indicator.symbol} Step {i}: {step.description}")

    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _escape_mermaid(text: str) -> str:
    """Escape special characters for Mermaid."""
    # Mermaid uses specific syntax, escape quotes and special chars
    return (
        text.replace('"', "'")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("&", "&amp;")
    )


def _escape_dot(text: str) -> str:
    """Escape special characters for GraphViz DOT."""
    return (
        text.replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("<", "\\<")
        .replace(">", "\\>")
    )


# Convenience functions for CLI
def visualize_plan(
    plan: Plan,
    *,
    format: str = "ascii",
    show_parallel: bool = True,
    show_status: bool = True,
) -> str:
    """Visualize a plan in the specified format.

    Args:
        plan: The plan to visualize
        format: Output format ("ascii", "mermaid", "tree", "dag", "dot")
        show_parallel: Whether to show parallel step indicators
        show_status: Whether to show status indicators (for dag format)

    Returns:
        Formatted plan visualization
    """
    if format == "ascii":
        return render_ascii(plan, show_parallel=show_parallel)
    elif format == "mermaid":
        return render_mermaid(plan)
    elif format == "tree":
        return render_simple_tree(plan)
    elif format == "dag":
        return render_dag_with_status(plan, format="mermaid", show_parallel=show_parallel, show_legend=show_status)
    elif format == "dot":
        return render_dag_with_status(plan, format="dot", show_parallel=show_parallel)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'ascii', 'mermaid', 'tree', 'dag', or 'dot'")
