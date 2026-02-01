"""
Plan Mode - Step-by-step task planning and execution for Copex.

Provides structured planning capabilities:
- Generate step-by-step plans from task descriptions
- Execute plans step by step with progress tracking
- Interactive review before execution
- Resume execution from specific steps
- Save/load plans to files
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable


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
        )


@dataclass
class Plan:
    """A complete execution plan."""
    
    task: str
    steps: list[PlanStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

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

    def to_dict(self) -> dict[str, Any]:
        """Convert plan to dictionary for serialization."""
        return {
            "task": self.task,
            "steps": [step.to_dict() for step in self.steps],
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        """Create plan from dictionary."""
        return cls(
            task=data["task"],
            steps=[PlanStep.from_dict(s) for s in data.get("steps", [])],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
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

    def __init__(self, client: Any):
        """Initialize executor with a Copex client."""
        self.client = client
        self._cancelled = False

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
    ) -> Plan:
        """
        Execute a plan step by step.
        
        Args:
            plan: The plan to execute
            from_step: Start execution from this step number
            on_step_start: Called when a step starts
            on_step_complete: Called when a step completes
            on_error: Called on error, return True to continue, False to stop
            
        Returns:
            The updated plan with execution results
        """
        self._cancelled = False
        
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
                
                response = await self.client.send(prompt)
                step.result = response.content
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.now()
                
                if on_step_complete:
                    on_step_complete(step)
                    
            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                step.completed_at = datetime.now()
                
                if on_error:
                    should_continue = on_error(step, e)
                    if not should_continue:
                        break
                else:
                    break
        
        if plan.is_complete:
            plan.completed_at = datetime.now()
        
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
