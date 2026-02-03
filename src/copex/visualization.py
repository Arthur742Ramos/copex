"""
Plan Visualization - Display plan dependency graphs and structure.

Provides ASCII and Mermaid diagram output for plan steps.
"""

from __future__ import annotations

from dataclasses import dataclass

from copex.plan import Plan


@dataclass
class DependencyEdge:
    """Represents a dependency edge between steps."""
    from_step: int  # Step index (1-indexed)
    to_step: int    # Step index (1-indexed)


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


def find_parallel_groups(plan: Plan) -> list[list[int]]:
    """Find groups of steps that can run in parallel.

    Returns a list of groups, where each group contains step indices
    that have no dependencies between them.
    """
    edges = analyze_dependencies(plan)
    n = len(plan.steps)

    # Build dependency graph
    deps: dict[int, set[int]] = {i: set() for i in range(1, n + 1)}
    for edge in edges:
        deps[edge.to_step].add(edge.from_step)

    # Group by dependency level (topological sort levels)
    remaining = set(range(1, n + 1))
    groups: list[list[int]] = []

    while remaining:
        # Find all steps with no remaining dependencies
        ready = [
            s for s in remaining
            if not deps[s].intersection(remaining)
        ]
        if not ready:
            # Circular dependency - just take first remaining
            ready = [min(remaining)]

        groups.append(sorted(ready))
        remaining -= set(ready)

    return groups


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
    for group_idx, group in enumerate(parallel_groups):
        for step_idx in group:
            group_map[step_idx] = group_idx

    for i, step in enumerate(plan.steps, 1):
        # Status indicator
        status = "○"  # pending
        if hasattr(step, "status"):
            status_map = {
                "completed": "●",
                "running": "◐",
                "failed": "✗",
                "skipped": "◌",
            }
            status = status_map.get(str(step.status).lower(), "○")

        # Parallel indicator
        parallel_info = ""
        if show_parallel and i in group_map:
            group_idx = group_map[i]
            group = parallel_groups[group_idx]
            if len(group) > 1:
                parallel_info = f" [∥{len(group)}]"

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
    if not plan.steps:
        return "graph TD\n    empty[Empty Plan]"

    lines: list[str] = []
    lines.append("graph TD")
    lines.append(f"    title[📋 {_escape_mermaid(plan.task)}]")
    lines.append("    style title fill:#e1f5fe,stroke:#01579b")
    lines.append("")

    # Define nodes
    for i, step in enumerate(plan.steps, 1):
        desc = _escape_mermaid(_truncate(step.description, 40))
        node_id = f"step{i}"

        # Node shape based on status
        shape_start, shape_end = "[", "]"  # Default rectangle
        if hasattr(step, "status"):
            status = str(step.status).lower()
            if status == "completed":
                shape_start, shape_end = "([", "])"  # Stadium
            elif status == "failed":
                shape_start, shape_end = "{{", "}}"  # Hexagon

        lines.append(f"    {node_id}{shape_start}Step {i}: {desc}{shape_end}")

    lines.append("")

    # Define edges
    edges = analyze_dependencies(plan)
    for edge in edges:
        lines.append(f"    step{edge.from_step} --> step{edge.to_step}")

    # Identify parallel groups for styling
    parallel_groups = find_parallel_groups(plan)
    colors = ["#ffeb3b", "#4caf50", "#2196f3", "#9c27b0", "#ff5722"]

    lines.append("")
    for group_idx, group in enumerate(parallel_groups):
        if len(group) > 1:
            color = colors[group_idx % len(colors)]
            for step_idx in group:
                lines.append(f"    style step{step_idx} fill:{color}")

    return "\n".join(lines)


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
        lines.append(f"   {prefix} Step {i}: {step.description}")

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


# Convenience functions for CLI
def visualize_plan(
    plan: Plan,
    *,
    format: str = "ascii",
    show_parallel: bool = True,
) -> str:
    """Visualize a plan in the specified format.

    Args:
        plan: The plan to visualize
        format: Output format ("ascii", "mermaid", "tree")
        show_parallel: Whether to show parallel step indicators

    Returns:
        Formatted plan visualization
    """
    if format == "ascii":
        return render_ascii(plan, show_parallel=show_parallel)
    elif format == "mermaid":
        return render_mermaid(plan)
    elif format == "tree":
        return render_simple_tree(plan)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'ascii', 'mermaid', or 'tree'")
