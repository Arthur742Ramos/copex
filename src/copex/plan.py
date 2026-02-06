"""
Plan Mode - Step-by-step task planning and execution for Copex.

Provides structured planning capabilities:
- Generate step-by-step plans from task descriptions
- Execute plans step by step with progress tracking (using Ralph loops)
- Interactive review before execution
- Resume execution from specific steps with checkpoint recovery
- Save/load plans to files
- Elapsed time tracking and ETA estimation
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from copex.ralph import RalphWiggum

# Default state file name
STATE_FILE_NAME = ".copex-state.json"


class StepStatus(Enum):
    """Status of a plan step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in a plan."""

    number: int
    description: str
    status: StepStatus = StepStatus.PENDING
    result: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    tokens_used: int = 0  # Track tokens for this step
    skip_condition: str | None = None  # Condition expression to skip this step
    depends_on: list[int] | None = None  # Step numbers this step depends on

    @property
    def duration_seconds(self) -> float | None:
        """Get step duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    async def should_skip(
        self,
        context: dict[str, Any] | None = None,
        evaluator: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> bool:
        """Evaluate whether this step should be skipped.

        Args:
            context: Dictionary of context variables for evaluation
            evaluator: Optional custom evaluator function(condition, context) -> bool

        Returns:
            True if the step should be skipped, False otherwise

        The skip_condition can be:
        - A simple expression like "step_3_completed"
        - A comparison like "error_count > 5"
        - A function call reference that the evaluator handles
        """
        if not self.skip_condition:
            return False

        ctx = context or {}

        # Use custom evaluator if provided
        if evaluator:
            try:
                return evaluator(self.skip_condition, ctx)
            except Exception:
                return False

        # Default simple evaluation
        condition = self.skip_condition.strip()

        # Check for simple boolean context key
        if condition in ctx:
            return bool(ctx[condition])

        # Check for "not X" pattern
        if condition.startswith("not "):
            key = condition[4:].strip()
            if key in ctx:
                return not bool(ctx[key])

        # Check for comparison patterns
        for op, func in [
            (">=", lambda a, b: a >= b),
            ("<=", lambda a, b: a <= b),
            ("!=", lambda a, b: a != b),
            ("==", lambda a, b: a == b),
            (">", lambda a, b: a > b),
            ("<", lambda a, b: a < b),
        ]:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    left_key = parts[0].strip()
                    right_val = parts[1].strip()
                    if left_key in ctx:
                        try:
                            # Try numeric comparison
                            left = float(ctx[left_key])
                            right = float(right_val)
                            return func(left, right)
                        except (ValueError, TypeError):
                            # Fall back to string comparison
                            return func(str(ctx[left_key]), right_val)
                break

        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "number": self.number,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "tokens_used": self.tokens_used,
            "skip_condition": self.skip_condition,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanStep:
        """Create step from dictionary."""
        return cls(
            number=data["number"],
            description=data["description"],
            status=StepStatus(data.get("status", "pending")),
            result=data.get("result"),
            error=data.get("error"),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            tokens_used=data.get("tokens_used", 0),
            skip_condition=data.get("skip_condition"),
            depends_on=data.get("depends_on"),
        )


@dataclass
class Plan:
    """A complete execution plan."""

    task: str
    steps: list[PlanStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    total_tokens: int = 0  # Total tokens used across all steps

    @property
    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(step.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) for step in self.steps)

    @property
    def current_step(self) -> PlanStep | None:
        """Get the next pending step."""
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                return step
        return None

    @property
    def completed_count(self) -> int:
        """Count of completed steps."""
        return sum(
            1 for step in self.steps if step.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
        )

    @property
    def failed_count(self) -> int:
        """Count of failed steps."""
        return sum(1 for step in self.steps if step.status == StepStatus.FAILED)

    @property
    def total_elapsed_seconds(self) -> float:
        """Total elapsed time for completed steps."""
        total = 0.0
        for step in self.steps:
            if step.duration_seconds is not None and step.status == StepStatus.COMPLETED:
                total += step.duration_seconds
        return total

    @property
    def avg_step_duration(self) -> float | None:
        """Average duration per completed step."""
        completed = [
            s for s in self.steps if s.status == StepStatus.COMPLETED and s.duration_seconds
        ]
        if not completed:
            return None
        return sum(s.duration_seconds for s in completed) / len(completed)

    def estimate_remaining_seconds(self) -> float | None:
        """Estimate remaining time based on average step duration."""
        avg = self.avg_step_duration
        if avg is None:
            return None
        remaining = len(self.steps) - self.completed_count
        return avg * remaining

    def to_dict(self) -> dict[str, Any]:
        """Convert plan to dictionary for serialization."""
        return {
            "task": self.task,
            "steps": [step.to_dict() for step in self.steps],
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        """Create plan from dictionary."""
        return cls(
            task=data["task"],
            steps=[PlanStep.from_dict(s) for s in data.get("steps", [])],
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            total_tokens=data.get("total_tokens", 0),
        )

    def to_json(self) -> str:
        """Serialize plan to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> Plan:
        """Create plan from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, path: Path) -> None:
        """Save plan to a file."""
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> Plan:
        """Load plan from a file."""
        return cls.from_json(path.read_text())

    def to_markdown(self) -> str:
        """Format plan as markdown."""
        lines = [f"# Plan: {self.task}", ""]
        for step in self.steps:
            status_icon = {
                StepStatus.PENDING: "⬜",
                StepStatus.RUNNING: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️",
            }.get(step.status, "⬜")
            lines.append(f"{status_icon} **Step {step.number}:** {step.description}")
            if step.result:
                lines.append(f"   - Result: {step.result[:100]}...")
            if step.error:
                lines.append(f"   - Error: {step.error}")
        return "\n".join(lines)


@dataclass
class PlanState:
    """
    Checkpoint state for resumable plan execution.

    Saved to .copex-state.json in working directory after each step completes.
    Enables resuming interrupted plans from the last completed step.
    """

    task: str
    plan: Plan
    completed: list[int] = field(default_factory=list)
    current_step: int = 1
    step_results: dict[str, str] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    total_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task": self.task,
            "plan": self.plan.to_dict(),
            "completed": self.completed,
            "current_step": self.current_step,
            "step_results": self.step_results,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanState":
        """Create from dictionary."""
        return cls(
            task=data["task"],
            plan=Plan.from_dict(data["plan"]),
            completed=data.get("completed", []),
            current_step=data.get("current_step", 1),
            step_results=data.get("step_results", {}),
            started_at=data.get("started_at", datetime.now().isoformat()),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
            total_tokens=data.get("total_tokens", 0),
        )

    def save(self, path: Path | None = None) -> Path:
        """
        Save state to file.

        Args:
            path: Optional path. Defaults to .copex-state.json in cwd.

        Returns:
            Path where state was saved.
        """
        if path is None:
            path = Path.cwd() / STATE_FILE_NAME
        self.last_updated = datetime.now().isoformat()
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "PlanState | None":
        """
        Load state from file.

        Args:
            path: Optional path. Defaults to .copex-state.json in cwd.

        Returns:
            PlanState if file exists, None otherwise.
        """
        if path is None:
            path = Path.cwd() / STATE_FILE_NAME
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    @classmethod
    def exists(cls, path: Path | None = None) -> bool:
        """Check if state file exists."""
        if path is None:
            path = Path.cwd() / STATE_FILE_NAME
        return path.exists()

    @classmethod
    def cleanup(cls, path: Path | None = None) -> bool:
        """
        Remove state file (called on successful completion).

        Returns:
            True if file was removed, False if it didn't exist.
        """
        if path is None:
            path = Path.cwd() / STATE_FILE_NAME
        if path.exists():
            path.unlink()
            return True
        return False

    def update_step(self, step: PlanStep) -> None:
        """Update state after a step completes."""
        self.current_step = step.number + 1
        if step.status == StepStatus.COMPLETED:
            if step.number not in self.completed:
                self.completed.append(step.number)
            if step.result:
                self.step_results[str(step.number)] = step.result[:500]  # Truncate long results
            self.total_tokens += step.tokens_used
        self.last_updated = datetime.now().isoformat()

    @classmethod
    def from_plan(cls, plan: Plan) -> "PlanState":
        """Create state from a plan."""
        completed = [s.number for s in plan.steps if s.status == StepStatus.COMPLETED]
        current = plan.current_step
        step_results = {
            str(s.number): s.result[:500] if s.result else ""
            for s in plan.steps
            if s.status == StepStatus.COMPLETED and s.result
        }
        return cls(
            task=plan.task,
            plan=plan,
            completed=completed,
            current_step=current.number if current else len(plan.steps) + 1,
            step_results=step_results,
            total_tokens=plan.total_tokens,
        )


PLAN_GENERATION_PROMPT = """You are a planning assistant. Generate a step-by-step plan for the following task.

TASK: {task}

Generate a numbered list of concrete, actionable steps. Each step should be:
1. Specific and executable
2. Self-contained (can be done independently or builds on previous steps)
3. Verifiable (you can check if it's done)

Format your response EXACTLY as:
STEP 1: [description]
STEP 2: [description]
...

Include 3-10 steps. Be concise but thorough."""


STEP_EXECUTION_PROMPT = """You are executing step {step_number} of a plan.

OVERALL TASK: {task}

COMPLETED STEPS:
{completed_steps}

CURRENT STEP: {current_step}

Execute this step now. When done, summarize what you accomplished."""


class PlanExecutor:
    """Executes plans step by step using a Copex client."""

    def __init__(self, client: Any, ralph: RalphWiggum | None = None):
        """
        Initialize executor with a Copex client.

        Args:
            client: A Copex client instance
            ralph: Optional RalphWiggum instance for iterative step execution
        """
        self.client = client
        self.ralph = ralph
        self._cancelled = False
        self.max_iterations_per_step: int = 10
        self._state: PlanState | None = None
        self._state_path: Path | None = None

    def cancel(self) -> None:
        """Cancel ongoing execution."""
        self._cancelled = True

    async def generate_plan(
        self,
        task: str,
        *,
        on_plan_generated: Callable[[Plan], None] | None = None,
    ) -> Plan:
        """Generate a plan for a task."""
        prompt = PLAN_GENERATION_PROMPT.format(task=task)
        response = await self.client.send(prompt)

        steps = self._parse_steps(response.content)
        plan = Plan(task=task, steps=steps)

        if on_plan_generated:
            on_plan_generated(plan)

        return plan

    def _parse_steps(self, content: str) -> list[PlanStep]:
        """Parse steps from AI response."""
        steps = []

        # Try line-by-line parsing first (most reliable)
        lines = content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match "STEP N: description" or "N. description" or "N: description"
            match = re.match(
                r"^(?:STEP\s*)?(\d+)[.:\)]\s*(.+)$",
                line,
                re.IGNORECASE,
            )
            if match:
                desc = match.group(2).strip()
                if desc:
                    steps.append(PlanStep(number=len(steps) + 1, description=desc))

        # Fallback: if line parsing failed, try multi-line pattern
        if not steps:
            pattern = r"(?:STEP\s*)?(\d+)[.:]\s*(.+?)(?=(?:\n\s*)?(?:STEP\s*)?\d+[.:]|\Z)"
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            for i, (num, desc) in enumerate(matches, 1):
                description = " ".join(desc.strip().split())
                if description:
                    steps.append(PlanStep(number=i, description=description))

        # Final fallback: split by lines and clean prefixes
        if not steps:
            for i, line in enumerate(lines, 1):
                clean = re.sub(r"^[\d]+[.:)]\s*", "", line.strip())
                clean = re.sub(r"^[-*]\s*", "", clean)
                if clean:
                    steps.append(PlanStep(number=i, description=clean))

        return steps

    async def execute_plan(
        self,
        plan: Plan,
        *,
        from_step: int = 1,
        on_step_start: Callable[[PlanStep], None] | None = None,
        on_step_complete: Callable[[PlanStep], None] | None = None,
        on_error: Callable[[PlanStep, Exception], bool] | None = None,
        save_checkpoints: bool = True,
        state_path: Path | None = None,
        skip_context: dict[str, Any] | None = None,
    ) -> Plan:
        """
        Execute a plan step by step.

        Args:
            plan: The plan to execute
            from_step: Start execution from this step number
            on_step_start: Called when a step starts
            on_step_complete: Called when a step completes
            on_error: Called on error, return True to continue, False to stop
            save_checkpoints: Whether to save state after each step (for resume)
            state_path: Path for state file (defaults to .copex-state.json in cwd)
            skip_context: Context dict for evaluating skip conditions

        Returns:
            The updated plan with execution results
        """
        self._cancelled = False
        self._state_path = state_path
        ctx = skip_context or {}

        # Initialize state for checkpointing
        if save_checkpoints:
            self._state = PlanState.from_plan(plan)
            self._state.save(self._state_path)

        for step in plan.steps:
            if self._cancelled:
                break

            if step.number < from_step:
                if step.status == StepStatus.PENDING:
                    step.status = StepStatus.SKIPPED
                continue

            if step.status in (StepStatus.COMPLETED, StepStatus.SKIPPED):
                continue

            # Check skip condition
            if await step.should_skip(context=ctx):
                step.status = StepStatus.SKIPPED
                step.completed_at = datetime.now()
                if save_checkpoints and self._state:
                    self._state.update_step(step)
                    self._state.plan = plan
                    self._state.save(self._state_path)
                if on_step_complete:
                    on_step_complete(step)
                continue

            step.status = StepStatus.RUNNING
            step.started_at = datetime.now()

            # Update checkpoint with running step
            if save_checkpoints and self._state:
                self._state.current_step = step.number
                self._state.save(self._state_path)

            if on_step_start:
                on_step_start(step)

            try:
                # Build context from completed steps
                completed_steps = (
                    "\n".join(
                        f"Step {s.number}: {s.description} - {s.result or 'Done'}"
                        for s in plan.steps
                        if s.status == StepStatus.COMPLETED and s.number < step.number
                    )
                    or "(none)"
                )

                prompt = STEP_EXECUTION_PROMPT.format(
                    step_number=step.number,
                    task=plan.task,
                    completed_steps=completed_steps,
                    current_step=step.description,
                )

                # Use Ralph loop if available, otherwise single call
                if self.ralph:
                    completion_promise = f"Step {step.number} complete"
                    ralph_state = await self.ralph.loop(
                        prompt,
                        max_iterations=self.max_iterations_per_step,
                        completion_promise=completion_promise,
                    )
                    # Use last response from Ralph loop as the result
                    step.result = (
                        ralph_state.history[-1] if ralph_state.history else "Step completed"
                    )
                else:
                    response = await self.client.send(prompt)
                    step.result = response.content

                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.now()

                # Update skip context with completion
                ctx[f"step_{step.number}_completed"] = True
                ctx[f"step_{step.number}_result"] = step.result

                # Save checkpoint after step completion
                if save_checkpoints and self._state:
                    self._state.update_step(step)
                    self._state.plan = plan  # Update plan in state
                    self._state.save(self._state_path)

                if on_step_complete:
                    on_step_complete(step)

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                step.completed_at = datetime.now()

                # Update skip context with failure
                ctx[f"step_{step.number}_failed"] = True
                ctx[f"step_{step.number}_error"] = str(e)

                # Save checkpoint on failure too
                if save_checkpoints and self._state:
                    self._state.current_step = step.number
                    self._state.plan = plan
                    self._state.save(self._state_path)

                if on_error:
                    should_continue = on_error(step, e)
                    if not should_continue:
                        break
                else:
                    break

        if plan.is_complete:
            plan.completed_at = datetime.now()
            plan.total_tokens = self._state.total_tokens if self._state else 0
            # Clean up state file on successful completion
            if save_checkpoints:
                PlanState.cleanup(self._state_path)

        return plan

    async def execute_step(
        self,
        plan: Plan,
        step_number: int,
    ) -> PlanStep:
        """Execute a single step from a plan."""
        step = next((s for s in plan.steps if s.number == step_number), None)
        if not step:
            raise ValueError(f"Step {step_number} not found in plan")

        await self.execute_plan(plan, from_step=step_number)
        return step


class PlanEditor:
    """Editor for modifying plan steps.

    Provides methods to edit, remove, insert, and reorder steps in a plan.

    Usage:
        editor = PlanEditor(plan)
        editor.edit_step(2, description="New description")
        editor.remove_step(3)
        editor.insert_step(2, "New step after step 2")
        editor.reorder([3, 1, 2])  # New order
    """

    def __init__(self, plan: Plan) -> None:
        self.plan = plan

    def edit_step(
        self,
        step_number: int,
        *,
        description: str | None = None,
        skip_condition: str | None = None,
        depends_on: list[int] | None = None,
    ) -> PlanStep:
        """Edit an existing step.

        Args:
            step_number: The step number to edit
            description: New description (if provided)
            skip_condition: New skip condition (if provided)
            depends_on: New dependencies (if provided)

        Returns:
            The edited step

        Raises:
            ValueError: If step not found
        """
        step = self._get_step(step_number)

        if description is not None:
            step.description = description
        if skip_condition is not None:
            step.skip_condition = skip_condition
        if depends_on is not None:
            step.depends_on = depends_on

        return step

    def remove_step(self, step_number: int) -> PlanStep:
        """Remove a step from the plan.

        Args:
            step_number: The step number to remove

        Returns:
            The removed step

        Raises:
            ValueError: If step not found
        """
        step = self._get_step(step_number)
        self.plan.steps.remove(step)

        # Renumber remaining steps
        for i, s in enumerate(self.plan.steps, 1):
            s.number = i

        # Update dependencies that referenced the removed step
        for s in self.plan.steps:
            if s.depends_on:
                s.depends_on = [
                    d - 1 if d > step_number else d
                    for d in s.depends_on
                    if d != step_number
                ]

        return step

    def insert_step(
        self,
        after_step: int,
        description: str,
        *,
        skip_condition: str | None = None,
        depends_on: list[int] | None = None,
    ) -> PlanStep:
        """Insert a new step after the specified step.

        Args:
            after_step: Insert after this step number (0 to insert at beginning)
            description: Description for the new step
            skip_condition: Optional skip condition
            depends_on: Optional dependencies

        Returns:
            The newly inserted step

        Raises:
            ValueError: If after_step is invalid
        """
        if after_step < 0 or after_step > len(self.plan.steps):
            raise ValueError(f"Invalid position: {after_step}")

        new_step = PlanStep(
            number=after_step + 1,
            description=description,
            skip_condition=skip_condition,
            depends_on=depends_on,
        )

        # Insert at the correct position
        self.plan.steps.insert(after_step, new_step)

        # Renumber steps after the insertion
        for i, s in enumerate(self.plan.steps, 1):
            s.number = i

        # Update dependencies to account for the new step
        for s in self.plan.steps:
            if s.depends_on and s != new_step:
                s.depends_on = [
                    d + 1 if d > after_step else d
                    for d in s.depends_on
                ]

        return new_step

    def reorder(self, new_order: list[int]) -> None:
        """Reorder steps according to the given order.

        Args:
            new_order: List of step numbers in desired order

        Raises:
            ValueError: If new_order doesn't contain all step numbers
        """
        current_numbers = {s.number for s in self.plan.steps}
        if set(new_order) != current_numbers:
            raise ValueError(
                f"new_order must contain exactly the current step numbers: {sorted(current_numbers)}"
            )

        # Build mapping from old to new position
        old_to_new = {old: new_pos + 1 for new_pos, old in enumerate(new_order)}

        # Sort steps by new order
        step_map = {s.number: s for s in self.plan.steps}
        self.plan.steps = [step_map[num] for num in new_order]

        # Update dependencies to reflect new numbering
        for step in self.plan.steps:
            if step.depends_on:
                step.depends_on = [old_to_new[d] for d in step.depends_on]

        # Renumber steps
        for i, s in enumerate(self.plan.steps, 1):
            s.number = i

    def duplicate_step(self, step_number: int) -> PlanStep:
        """Duplicate a step, inserting the copy after the original.

        Args:
            step_number: The step number to duplicate

        Returns:
            The new duplicated step
        """
        original = self._get_step(step_number)
        return self.insert_step(
            step_number,
            description=f"{original.description} (copy)",
            skip_condition=original.skip_condition,
            depends_on=original.depends_on.copy() if original.depends_on else None,
        )

    def clear_results(self) -> None:
        """Reset all steps to pending status, clearing results."""
        for step in self.plan.steps:
            step.status = StepStatus.PENDING
            step.result = None
            step.error = None
            step.started_at = None
            step.completed_at = None
            step.tokens_used = 0

    def _get_step(self, step_number: int) -> PlanStep:
        """Get a step by number."""
        for step in self.plan.steps:
            if step.number == step_number:
                return step
        raise ValueError(f"Step {step_number} not found in plan")


@dataclass
class PlanCheckpoint:
    """Checkpoint for step-level resume with context preservation.

    Extends PlanState with methods for checking resume capability
    and extracting step context for continuation.

    Usage:
        checkpoint = PlanCheckpoint.from_state(state)
        if checkpoint.can_resume_from(step_number):
            context = checkpoint.get_step_context(step_number)
            # Resume with context
    """

    state: PlanState
    step_contexts: dict[int, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: PlanState) -> "PlanCheckpoint":
        """Create a checkpoint from a PlanState."""
        # Build step contexts from completed steps
        step_contexts: dict[int, dict[str, Any]] = {}
        for step in state.plan.steps:
            if step.status == StepStatus.COMPLETED:
                step_contexts[step.number] = {
                    "description": step.description,
                    "result": step.result,
                    "duration": step.duration_seconds,
                    "tokens": step.tokens_used,
                }
        return cls(state=state, step_contexts=step_contexts)

    @classmethod
    def load(cls, path: Path | None = None) -> "PlanCheckpoint | None":
        """Load checkpoint from state file."""
        state = PlanState.load(path)
        if state is None:
            return None
        return cls.from_state(state)

    def save(self, path: Path | None = None) -> Path:
        """Save the checkpoint."""
        return self.state.save(path)

    def can_resume_from(self, step_number: int) -> bool:
        """Check if execution can resume from the given step.

        Args:
            step_number: The step to check

        Returns:
            True if:
            - All steps before step_number are completed or skipped
            - The step exists in the plan
            - The step is not already completed
        """
        plan = self.state.plan

        # Check step exists
        step = next((s for s in plan.steps if s.number == step_number), None)
        if step is None:
            return False

        # Already completed - no need to resume from here
        if step.status == StepStatus.COMPLETED:
            return False

        # Check all prior steps are done
        for s in plan.steps:
            if s.number < step_number:
                if s.status not in (StepStatus.COMPLETED, StepStatus.SKIPPED):
                    return False

        return True

    def get_step_context(self, step_number: int) -> dict[str, Any]:
        """Get context for resuming from a specific step.

        Args:
            step_number: The step to get context for

        Returns:
            Dictionary containing:
            - task: The overall plan task
            - prior_steps: List of completed step summaries
            - current_step: The step to resume
            - skip_context: Context dict for skip evaluation
        """
        plan = self.state.plan

        step = next((s for s in plan.steps if s.number == step_number), None)
        if step is None:
            raise ValueError(f"Step {step_number} not found")

        prior_steps = []
        skip_context: dict[str, Any] = {}

        for s in plan.steps:
            if s.number < step_number and s.status == StepStatus.COMPLETED:
                prior_steps.append({
                    "number": s.number,
                    "description": s.description,
                    "result_preview": (s.result or "")[:200],
                })
                skip_context[f"step_{s.number}_completed"] = True
                skip_context[f"step_{s.number}_result"] = s.result

        return {
            "task": plan.task,
            "prior_steps": prior_steps,
            "current_step": {
                "number": step.number,
                "description": step.description,
                "skip_condition": step.skip_condition,
                "depends_on": step.depends_on,
            },
            "skip_context": skip_context,
            "total_completed": len(self.state.completed),
            "total_steps": len(plan.steps),
        }

    def get_resume_prompt(self, step_number: int) -> str:
        """Generate a prompt for resuming from a step with context.

        Args:
            step_number: The step to resume from

        Returns:
            A formatted prompt string with context
        """
        ctx = self.get_step_context(step_number)

        prior_summary = ""
        if ctx["prior_steps"]:
            lines = []
            for ps in ctx["prior_steps"]:
                lines.append(f"Step {ps['number']}: {ps['description']}")
                if ps["result_preview"]:
                    lines.append(f"  → {ps['result_preview']}")
            prior_summary = "\n".join(lines)
        else:
            prior_summary = "(none)"

        return STEP_EXECUTION_PROMPT.format(
            step_number=ctx["current_step"]["number"],
            task=ctx["task"],
            completed_steps=prior_summary,
            current_step=ctx["current_step"]["description"],
        )
