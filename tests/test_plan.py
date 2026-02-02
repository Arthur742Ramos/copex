"""Tests for plan mode."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from copex.plan import (
    Plan,
    PlanExecutor,
    PlanStep,
    StepStatus,
    PLAN_GENERATION_PROMPT,
    STEP_EXECUTION_PROMPT,
)


class TestPlanStep:
    """Tests for PlanStep dataclass."""

    def test_default_values(self):
        """Step should have sensible defaults."""
        step = PlanStep(number=1, description="Test step")
        
        assert step.number == 1
        assert step.description == "Test step"
        assert step.status == StepStatus.PENDING
        assert step.result is None
        assert step.error is None
        assert step.started_at is None
        assert step.completed_at is None

    def test_to_dict(self):
        """Should serialize to dictionary."""
        step = PlanStep(
            number=1,
            description="Test step",
            status=StepStatus.COMPLETED,
            result="Done",
        )
        
        data = step.to_dict()
        
        assert data["number"] == 1
        assert data["description"] == "Test step"
        assert data["status"] == "completed"
        assert data["result"] == "Done"

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "number": 2,
            "description": "Another step",
            "status": "failed",
            "error": "Something went wrong",
        }
        
        step = PlanStep.from_dict(data)
        
        assert step.number == 2
        assert step.description == "Another step"
        assert step.status == StepStatus.FAILED
        assert step.error == "Something went wrong"

    def test_from_dict_with_timestamps(self):
        """Should handle timestamp deserialization."""
        now = datetime.now()
        data = {
            "number": 1,
            "description": "Step",
            "started_at": now.isoformat(),
            "completed_at": now.isoformat(),
        }
        
        step = PlanStep.from_dict(data)
        
        assert step.started_at is not None
        assert step.completed_at is not None


class TestPlan:
    """Tests for Plan dataclass."""

    def test_empty_plan(self):
        """Empty plan should have no steps."""
        plan = Plan(task="Test task")
        
        assert plan.task == "Test task"
        assert plan.steps == []
        assert plan.is_complete
        assert plan.current_step is None

    def test_plan_with_steps(self):
        """Plan with pending steps should not be complete."""
        plan = Plan(
            task="Build app",
            steps=[
                PlanStep(number=1, description="Step 1"),
                PlanStep(number=2, description="Step 2"),
            ],
        )
        
        assert not plan.is_complete
        assert plan.current_step == plan.steps[0]
        assert plan.completed_count == 0

    def test_completed_plan(self):
        """Plan with all completed steps should be complete."""
        plan = Plan(
            task="Build app",
            steps=[
                PlanStep(number=1, description="Step 1", status=StepStatus.COMPLETED),
                PlanStep(number=2, description="Step 2", status=StepStatus.COMPLETED),
            ],
        )
        
        assert plan.is_complete
        assert plan.current_step is None
        assert plan.completed_count == 2

    def test_skipped_steps_count_as_complete(self):
        """Skipped steps should count towards completion."""
        plan = Plan(
            task="Build app",
            steps=[
                PlanStep(number=1, description="Step 1", status=StepStatus.SKIPPED),
                PlanStep(number=2, description="Step 2", status=StepStatus.COMPLETED),
            ],
        )
        
        assert plan.is_complete
        assert plan.completed_count == 2

    def test_failed_count(self):
        """Should count failed steps correctly."""
        plan = Plan(
            task="Build app",
            steps=[
                PlanStep(number=1, description="Step 1", status=StepStatus.COMPLETED),
                PlanStep(number=2, description="Step 2", status=StepStatus.FAILED),
                PlanStep(number=3, description="Step 3", status=StepStatus.FAILED),
            ],
        )
        
        assert plan.failed_count == 2
        assert plan.completed_count == 1

    def test_to_json_and_from_json(self):
        """Should serialize/deserialize to JSON."""
        plan = Plan(
            task="Build app",
            steps=[
                PlanStep(number=1, description="Step 1"),
                PlanStep(number=2, description="Step 2", status=StepStatus.COMPLETED),
            ],
        )
        
        json_str = plan.to_json()
        loaded = Plan.from_json(json_str)
        
        assert loaded.task == plan.task
        assert len(loaded.steps) == len(plan.steps)
        assert loaded.steps[0].description == "Step 1"
        assert loaded.steps[1].status == StepStatus.COMPLETED

    def test_save_and_load(self):
        """Should save and load from file."""
        plan = Plan(
            task="Test task",
            steps=[PlanStep(number=1, description="Step 1")],
        )
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        
        try:
            plan.save(path)
            loaded = Plan.load(path)
            
            assert loaded.task == plan.task
            assert len(loaded.steps) == 1
        finally:
            path.unlink()

    def test_to_markdown(self):
        """Should generate markdown representation."""
        plan = Plan(
            task="Build app",
            steps=[
                PlanStep(number=1, description="Step 1", status=StepStatus.COMPLETED),
                PlanStep(number=2, description="Step 2", status=StepStatus.PENDING),
                PlanStep(number=3, description="Step 3", status=StepStatus.FAILED, error="Error!"),
            ],
        )
        
        md = plan.to_markdown()
        
        assert "# Plan: Build app" in md
        assert "✅" in md  # Completed icon
        assert "⬜" in md  # Pending icon
        assert "❌" in md  # Failed icon
        assert "Error!" in md


class TestPlanExecutor:
    """Tests for PlanExecutor."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Copex client."""
        client = MagicMock()
        client.send = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_generate_plan_parses_steps(self, mock_client):
        """Should parse steps from AI response."""
        mock_client.send.return_value = MagicMock(
            content="""STEP 1: Create the project structure
STEP 2: Write the main module
STEP 3: Add tests
STEP 4: Document the code"""
        )
        
        executor = PlanExecutor(mock_client)
        plan = await executor.generate_plan("Build a Python package")
        
        assert len(plan.steps) == 4
        assert plan.steps[0].description == "Create the project structure"
        assert plan.steps[1].description == "Write the main module"
        assert plan.steps[2].description == "Add tests"
        assert plan.steps[3].description == "Document the code"

    @pytest.mark.asyncio
    async def test_generate_plan_handles_numbered_format(self, mock_client):
        """Should parse numbered list format."""
        mock_client.send.return_value = MagicMock(
            content="""1. First step
2. Second step
3. Third step"""
        )
        
        executor = PlanExecutor(mock_client)
        plan = await executor.generate_plan("Test task")
        
        assert len(plan.steps) == 3
        assert plan.steps[0].description == "First step"

    @pytest.mark.asyncio
    async def test_generate_plan_callback(self, mock_client):
        """Should call on_plan_generated callback."""
        mock_client.send.return_value = MagicMock(content="STEP 1: Do something")
        
        callback_called = []
        def on_plan_generated(plan):
            callback_called.append(plan)
        
        executor = PlanExecutor(mock_client)
        await executor.generate_plan("Test", on_plan_generated=on_plan_generated)
        
        assert len(callback_called) == 1
        assert callback_called[0].task == "Test"

    @pytest.mark.asyncio
    async def test_execute_plan_basic(self, mock_client):
        """Should execute all steps in order."""
        mock_client.send.return_value = MagicMock(content="Step completed")
        
        plan = Plan(
            task="Test task",
            steps=[
                PlanStep(number=1, description="Step 1"),
                PlanStep(number=2, description="Step 2"),
            ],
        )
        
        executor = PlanExecutor(mock_client)
        result = await executor.execute_plan(plan)
        
        assert result.is_complete
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[1].status == StepStatus.COMPLETED
        assert mock_client.send.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_plan_from_step(self, mock_client):
        """Should skip steps before from_step."""
        mock_client.send.return_value = MagicMock(content="Step completed")
        
        plan = Plan(
            task="Test task",
            steps=[
                PlanStep(number=1, description="Step 1"),
                PlanStep(number=2, description="Step 2"),
                PlanStep(number=3, description="Step 3"),
            ],
        )
        
        executor = PlanExecutor(mock_client)
        result = await executor.execute_plan(plan, from_step=2)
        
        assert result.steps[0].status == StepStatus.SKIPPED
        assert result.steps[1].status == StepStatus.COMPLETED
        assert result.steps[2].status == StepStatus.COMPLETED
        assert mock_client.send.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_plan_skips_completed_steps(self, mock_client):
        """Should not re-execute completed steps."""
        mock_client.send.return_value = MagicMock(content="Step completed")
        
        plan = Plan(
            task="Test task",
            steps=[
                PlanStep(number=1, description="Step 1", status=StepStatus.COMPLETED),
                PlanStep(number=2, description="Step 2"),
            ],
        )
        
        executor = PlanExecutor(mock_client)
        result = await executor.execute_plan(plan)
        
        assert mock_client.send.call_count == 1  # Only step 2

    @pytest.mark.asyncio
    async def test_execute_plan_callbacks(self, mock_client):
        """Should call step callbacks."""
        mock_client.send.return_value = MagicMock(content="Done")
        
        plan = Plan(
            task="Test",
            steps=[PlanStep(number=1, description="Step 1")],
        )
        
        started = []
        completed = []
        
        executor = PlanExecutor(mock_client)
        await executor.execute_plan(
            plan,
            on_step_start=lambda s: started.append(s.number),
            on_step_complete=lambda s: completed.append(s.number),
        )
        
        assert started == [1]
        assert completed == [1]

    @pytest.mark.asyncio
    async def test_execute_plan_error_stops_by_default(self, mock_client):
        """Should stop on error by default."""
        mock_client.send.side_effect = [
            MagicMock(content="Done"),
            Exception("Error!"),
            MagicMock(content="Done"),
        ]
        
        plan = Plan(
            task="Test",
            steps=[
                PlanStep(number=1, description="Step 1"),
                PlanStep(number=2, description="Step 2"),
                PlanStep(number=3, description="Step 3"),
            ],
        )
        
        executor = PlanExecutor(mock_client)
        result = await executor.execute_plan(plan)
        
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[1].status == StepStatus.FAILED
        assert result.steps[2].status == StepStatus.PENDING

    @pytest.mark.asyncio
    async def test_execute_plan_error_continues_if_callback_returns_true(self, mock_client):
        """Should continue on error if callback returns True."""
        mock_client.send.side_effect = [
            Exception("Error!"),
            MagicMock(content="Done"),
        ]
        
        plan = Plan(
            task="Test",
            steps=[
                PlanStep(number=1, description="Step 1"),
                PlanStep(number=2, description="Step 2"),
            ],
        )
        
        executor = PlanExecutor(mock_client)
        result = await executor.execute_plan(
            plan,
            on_error=lambda step, e: True,  # Continue
        )
        
        assert result.steps[0].status == StepStatus.FAILED
        assert result.steps[1].status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_plan_sets_timestamps(self, mock_client):
        """Should set started_at and completed_at."""
        mock_client.send.return_value = MagicMock(content="Done")
        
        plan = Plan(
            task="Test",
            steps=[PlanStep(number=1, description="Step 1")],
        )
        
        executor = PlanExecutor(mock_client)
        result = await executor.execute_plan(plan)
        
        assert result.steps[0].started_at is not None
        assert result.steps[0].completed_at is not None

    @pytest.mark.asyncio
    async def test_execute_step_single(self, mock_client):
        """Should execute a single step."""
        mock_client.send.return_value = MagicMock(content="Done")
        
        plan = Plan(
            task="Test",
            steps=[
                PlanStep(number=1, description="Step 1"),
                PlanStep(number=2, description="Step 2"),
            ],
        )
        
        executor = PlanExecutor(mock_client)
        step = await executor.execute_step(plan, 2)
        
        assert step.status == StepStatus.COMPLETED
        assert plan.steps[0].status == StepStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_execute_step_not_found(self, mock_client):
        """Should raise error if step not found."""
        plan = Plan(
            task="Test",
            steps=[PlanStep(number=1, description="Step 1")],
        )
        
        executor = PlanExecutor(mock_client)
        
        with pytest.raises(ValueError, match="Step 5 not found"):
            await executor.execute_step(plan, 5)

    @pytest.mark.asyncio
    async def test_cancel_execution(self, mock_client):
        """Should stop execution when cancelled."""
        call_count = 0
        
        async def slow_send(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return MagicMock(content="Done")
        
        mock_client.send = slow_send
        
        plan = Plan(
            task="Test",
            steps=[
                PlanStep(number=1, description="Step 1"),
                PlanStep(number=2, description="Step 2"),
            ],
        )
        
        executor = PlanExecutor(mock_client)
        
        def on_step_complete(step):
            executor.cancel()
        
        await executor.execute_plan(plan, on_step_complete=on_step_complete)
        
        # Should have stopped after first step
        assert call_count == 1
        assert plan.steps[0].status == StepStatus.COMPLETED
        assert plan.steps[1].status == StepStatus.PENDING

    def test_parse_steps_various_formats(self, mock_client):
        """Should handle various step formats."""
        executor = PlanExecutor(mock_client)
        
        # Test STEP X: format
        steps = executor._parse_steps("STEP 1: First\nSTEP 2: Second")
        assert len(steps) == 2
        
        # Test numbered list
        steps = executor._parse_steps("1. First\n2. Second\n3. Third")
        assert len(steps) == 3
        
        # Test with colons
        steps = executor._parse_steps("1: First\n2: Second")
        assert len(steps) == 2

    def test_parse_steps_with_parentheses(self, mock_client):
        """Should handle step format with parentheses."""
        executor = PlanExecutor(mock_client)
        
        steps = executor._parse_steps("1) First step\n2) Second step")
        assert len(steps) == 2
        assert steps[0].description == "First step"

    def test_parse_steps_empty_content(self, mock_client):
        """Should return empty list for empty content."""
        executor = PlanExecutor(mock_client)
        
        steps = executor._parse_steps("")
        assert steps == []
        
        steps = executor._parse_steps("   \n\n   ")
        assert steps == []

    def test_parse_steps_mixed_format(self, mock_client):
        """Should handle mixed formats."""
        executor = PlanExecutor(mock_client)
        
        content = """Here's my plan:
STEP 1: Do the first thing
STEP 2: Do the second thing
STEP 3: Finish up"""
        
        steps = executor._parse_steps(content)
        assert len(steps) == 3
        assert steps[0].description == "Do the first thing"


class TestPlanEdgeCases:
    """Edge case tests for Plan."""

    def test_plan_current_step_with_mixed_statuses(self):
        """Current step should skip completed and failed."""
        plan = Plan(
            task="Test",
            steps=[
                PlanStep(number=1, description="Step 1", status=StepStatus.COMPLETED),
                PlanStep(number=2, description="Step 2", status=StepStatus.FAILED),
                PlanStep(number=3, description="Step 3", status=StepStatus.PENDING),
            ],
        )
        
        assert plan.current_step == plan.steps[2]

    def test_plan_not_complete_with_running_step(self):
        """Plan with running step should not be complete."""
        plan = Plan(
            task="Test",
            steps=[
                PlanStep(number=1, description="Step 1", status=StepStatus.COMPLETED),
                PlanStep(number=2, description="Step 2", status=StepStatus.RUNNING),
            ],
        )
        
        assert not plan.is_complete

    def test_plan_json_roundtrip_preserves_all_fields(self):
        """JSON roundtrip should preserve all fields."""
        now = datetime.now()
        plan = Plan(
            task="Complex task",
            steps=[
                PlanStep(
                    number=1,
                    description="Step 1",
                    status=StepStatus.COMPLETED,
                    result="Success",
                    started_at=now,
                    completed_at=now,
                ),
                PlanStep(
                    number=2,
                    description="Step 2",
                    status=StepStatus.FAILED,
                    error="Something broke",
                ),
            ],
            created_at=now,
        )
        
        json_str = plan.to_json()
        loaded = Plan.from_json(json_str)
        
        assert loaded.task == plan.task
        assert loaded.steps[0].result == "Success"
        assert loaded.steps[1].error == "Something broke"


class TestStepStatus:
    """Tests for StepStatus enum."""

    def test_status_values(self):
        """All statuses should have correct values."""
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.RUNNING.value == "running"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_plan_generation_prompt_has_placeholders(self):
        """PLAN_GENERATION_PROMPT should have task placeholder."""
        assert "{task}" in PLAN_GENERATION_PROMPT
        # Should format without error
        result = PLAN_GENERATION_PROMPT.format(task="Build an app")
        assert "Build an app" in result

    def test_step_execution_prompt_has_placeholders(self):
        """STEP_EXECUTION_PROMPT should have all placeholders."""
        assert "{step_number}" in STEP_EXECUTION_PROMPT
        assert "{task}" in STEP_EXECUTION_PROMPT
        assert "{completed_steps}" in STEP_EXECUTION_PROMPT
        assert "{current_step}" in STEP_EXECUTION_PROMPT
        
        # Should format without error
        result = STEP_EXECUTION_PROMPT.format(
            step_number=1,
            task="Build app",
            completed_steps="None",
            current_step="Do something",
        )
        assert "Step 1" in result or "step 1" in result
