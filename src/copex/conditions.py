"""
Conditional Plan Steps - Dynamic plan flows based on conditions.

Allows plan steps to be conditionally executed based on:
- Prior step outputs
- Environment state
- Custom condition functions
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class Condition:
    """A condition that determines whether a step should execute.

    Conditions can reference prior step outputs using the syntax:
    - "${step.N.output}" - Raw output of step N
    - "${step.N.status}" - Status of step N (completed, failed, skipped)
    - "${env.VAR}" - Environment variable VAR
    """

    expression: str

    def evaluate(self, context: ConditionContext) -> bool:
        """Evaluate the condition against the given context.

        Args:
            context: The context containing step outputs and state

        Returns:
            True if the condition is met, False otherwise
        """
        # Parse and evaluate the expression
        try:
            return _evaluate_expression(self.expression, context)
        except Exception as e:
            # Log but don't crash - default to True (execute step)
            import logging

            logging.warning(f"Condition evaluation failed: {e}")
            return True

    @classmethod
    def always(cls) -> Condition:
        """Create a condition that always evaluates to True."""
        return cls("true")

    @classmethod
    def never(cls) -> Condition:
        """Create a condition that always evaluates to False."""
        return cls("false")

    @classmethod
    def step_completed(cls, step_index: int) -> Condition:
        """Create a condition that checks if a step completed successfully."""
        return cls(f"${{step.{step_index}.status}} == 'completed'")

    @classmethod
    def step_failed(cls, step_index: int) -> Condition:
        """Create a condition that checks if a step failed."""
        return cls(f"${{step.{step_index}.status}} == 'failed'")

    @classmethod
    def step_output_contains(cls, step_index: int, substring: str) -> Condition:
        """Create a condition that checks if a step's output contains a string."""
        return cls(f"'{substring}' in ${{step.{step_index}.output}}")

    @classmethod
    def env_equals(cls, var_name: str, value: str) -> Condition:
        """Create a condition that checks an environment variable."""
        return cls(f"${{env.{var_name}}} == '{value}'")

    @classmethod
    def custom(cls, func: Callable[[dict[str, Any]], bool]) -> CustomCondition:
        """Create a condition with a custom evaluation function."""
        return CustomCondition(func)


class CustomCondition(Condition):
    """A condition with a custom evaluation function."""

    def __init__(self, func: Callable[[dict[str, Any]], bool]) -> None:
        super().__init__("custom")
        self._func = func

    def evaluate(self, context: ConditionContext) -> bool:
        """Evaluate using the custom function."""
        return self._func(context.to_dict())


@dataclass
class ConditionContext:
    """Context for condition evaluation.

    Contains:
    - Step outputs and statuses from prior steps
    - Environment variables
    - Custom variables
    """

    step_outputs: dict[int, str]  # step_index -> output
    step_statuses: dict[int, str]  # step_index -> status
    env: Mapping[str, str]  # Environment variables
    variables: dict[str, Any]  # Custom variables

    def get_step_output(self, index: int) -> str:
        """Get the output of a step by index."""
        return self.step_outputs.get(index, "")

    def get_step_status(self, index: int) -> str:
        """Get the status of a step by index."""
        return self.step_statuses.get(index, "pending")

    def get_env(self, name: str) -> str:
        """Get an environment variable."""
        return self.env.get(name, "")

    def get_variable(self, name: str) -> Any:
        """Get a custom variable."""
        return self.variables.get(name)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary for custom functions."""
        return {
            "step_outputs": self.step_outputs,
            "step_statuses": self.step_statuses,
            "env": dict(self.env),
            "variables": self.variables,
        }

    @classmethod
    def empty(cls) -> ConditionContext:
        """Create an empty context."""
        import os

        return cls(
            step_outputs={},
            step_statuses={},
            env=os.environ,
            variables={},
        )


# Expression evaluation

# Reference pattern: ${step.N.field} or ${env.VAR} or ${var.NAME}
_REF_PATTERN = re.compile(r"\$\{(step|env|var)\.([^}]+)\}")

# Comparison operators
_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    " in ": lambda a, b: a in b,
    " not in ": lambda a, b: a not in b,
}


def _resolve_reference(ref_type: str, ref_path: str, context: ConditionContext) -> Any:
    """Resolve a variable reference to its value."""
    if ref_type == "step":
        # Parse step.N.field
        parts = ref_path.split(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid step reference: {ref_path}")
        try:
            step_index = int(parts[0])
        except ValueError:
            raise ValueError(f"Invalid step index: {parts[0]}") from None

        field = parts[1]
        if field == "output":
            return context.get_step_output(step_index)
        elif field == "status":
            return context.get_step_status(step_index)
        else:
            raise ValueError(f"Unknown step field: {field}")

    elif ref_type == "env":
        return context.get_env(ref_path)

    elif ref_type == "var":
        return context.get_variable(ref_path)

    else:
        raise ValueError(f"Unknown reference type: {ref_type}")


def _substitute_references(expr: str, context: ConditionContext) -> str:
    """Substitute all references in an expression with their values."""

    def replace(match: re.Match[str]) -> str:
        ref_type = match.group(1)
        ref_path = match.group(2)
        value = _resolve_reference(ref_type, ref_path, context)

        # Quote strings for proper comparison
        if isinstance(value, str):
            # Escape quotes in the value
            value = value.replace("'", "\\'")
            return f"'{value}'"
        return str(value)

    return _REF_PATTERN.sub(replace, expr)


def _evaluate_expression(expr: str, context: ConditionContext) -> bool:
    """Evaluate a condition expression.

    Supports:
    - Boolean literals: true, false
    - Comparisons: ==, !=, <, <=, >, >=
    - Membership: in, not in
    - Logical: and, or, not
    """
    expr = expr.strip()

    # Boolean literals
    if expr.lower() == "true":
        return True
    if expr.lower() == "false":
        return False

    # Substitute references
    resolved = _substitute_references(expr, context)

    # Try to find a comparison operator
    for op_str, op_func in _OPERATORS.items():
        if op_str in resolved:
            parts = resolved.split(op_str, 1)
            if len(parts) == 2:
                left = _parse_value(parts[0].strip())
                right = _parse_value(parts[1].strip())
                return op_func(left, right)

    # Try to evaluate as a simple truthy value
    value = _parse_value(resolved)
    return bool(value)


def _parse_value(val_str: str) -> Any:
    """Parse a value string into the appropriate type."""
    val_str = val_str.strip()

    # Quoted string
    if (val_str.startswith("'") and val_str.endswith("'")) or (
        val_str.startswith('"') and val_str.endswith('"')
    ):
        return val_str[1:-1]

    # Boolean
    if val_str.lower() == "true":
        return True
    if val_str.lower() == "false":
        return False

    # None
    if val_str.lower() in ("none", "null"):
        return None

    # Number
    try:
        if "." in val_str:
            return float(val_str)
        return int(val_str)
    except ValueError:
        pass

    # Default to string
    return val_str


# Convenience functions for building conditional steps


def when(condition: str | Condition) -> Condition:
    """Create a condition from a string expression or existing Condition.

    Args:
        condition: Expression string or Condition object

    Returns:
        A Condition object
    """
    if isinstance(condition, Condition):
        return condition
    return Condition(condition)


def all_of(*conditions: str | Condition) -> Condition:
    """Create a condition that requires all sub-conditions to be true.

    Args:
        *conditions: Conditions that must all be true

    Returns:
        A combined Condition
    """
    cond_objs = [when(c) for c in conditions]

    class AllOfCondition(Condition):
        def evaluate(self, context: ConditionContext) -> bool:
            return all(c.evaluate(context) for c in cond_objs)

    exprs = " and ".join(c.expression for c in cond_objs)
    return AllOfCondition(f"({exprs})")


def any_of(*conditions: str | Condition) -> Condition:
    """Create a condition that requires any sub-condition to be true.

    Args:
        *conditions: Conditions where at least one must be true

    Returns:
        A combined Condition
    """
    cond_objs = [when(c) for c in conditions]

    class AnyOfCondition(Condition):
        def evaluate(self, context: ConditionContext) -> bool:
            return any(c.evaluate(context) for c in cond_objs)

    exprs = " or ".join(c.expression for c in cond_objs)
    return AnyOfCondition(f"({exprs})")


def none_of(*conditions: str | Condition) -> Condition:
    """Create a condition that requires no sub-conditions to be true.

    Args:
        *conditions: Conditions that must all be false

    Returns:
        A combined Condition
    """
    cond_objs = [when(c) for c in conditions]

    class NoneOfCondition(Condition):
        def evaluate(self, context: ConditionContext) -> bool:
            return not any(c.evaluate(context) for c in cond_objs)

    exprs = " and ".join(f"not ({c.expression})" for c in cond_objs)
    return NoneOfCondition(f"({exprs})")
