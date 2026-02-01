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
import time
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

    @property
    def duration_seconds(self) -> float | None:
        """Get step duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

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
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            tokens_used=data.get("tokens_used", 0),
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
        return all(
            step.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
            for step in self.steps
        )

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
            1 for step in self.steps
            if step.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
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
        completed = [s for s in self.steps if s.status == StepStatus.COMPLETED and s.duration_seconds]
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
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
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

    def to_markdown(self, *, ascii_icons: bool = False) -> str:
        """Format plan as markdown."""
        if ascii_icons:
            icons = {
                StepStatus.PENDING: "-",
                StepStatus.RUNNING: "*",
                StepStatus.COMPLETED: "[OK]",
                StepStatus.FAILED: "[X]",
                StepStatus.SKIPPED: "->",
            }
        else:
            icons = {
                StepStatus.PENDING: "⬜",
                StepStatus.RUNNING: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️",
            }
        lines = [f"# Plan: {self.task}", ""]
        for step in self.steps:
            status_icon = icons.get(step.status, icons[StepStatus.PENDING])
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
            
            # Match "STEP N - description" or "N) description" variants
            match = re.match(
                r"^(?:STEP\s*)?(\d+)\s*[.:\)-]\s*(.+)$",
                line,
                re.IGNORECASE,
            )
            if match:
                desc = match.group(2).strip()
                if desc:
                    steps.append(PlanStep(number=len(steps) + 1, description=desc))
        
        # Fallback: if line parsing failed, try multi-line pattern
        if not steps:
            pattern = r"(?:STEP\s*)?(\d+)\s*[.:\)-]\s*(.+?)(?=(?:\n\s*)?(?:STEP\s*)?\d+\s*[.:\)-]|\Z)"
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

        if not steps:
            import logging
            logging.getLogger(__name__).warning("No plan steps parsed from response")
        
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
            
        Returns:
            The updated plan with execution results
        """
        self._cancelled = False
        self._state_path = state_path
        
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
                completed_steps = "\n".join(
                    f"Step {s.number}: {s.description} - {s.result or 'Done'}"
                    for s in plan.steps
                    if s.status == StepStatus.COMPLETED and s.number < step.number
                ) or "(none)"
                
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
                    step.result = ralph_state.history[-1] if ralph_state.history else "Step completed"
                else:
                    response = await self.client.send(prompt)
                    step.result = response.content
                    
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.now()
                
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
