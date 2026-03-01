"""Tests for copex.conditions — Condition evaluation and combinators."""

from __future__ import annotations

import pytest

from copex.conditions import (
    Condition,
    ConditionContext,
    CustomCondition,
    all_of,
    any_of,
    none_of,
    when,
    _parse_value,
    _resolve_reference,
    _substitute_references,
    _evaluate_expression,
)


@pytest.fixture
def ctx() -> ConditionContext:
    """Standard context for tests."""
    return ConditionContext(
        step_outputs={0: "all tests passed", 1: "build failed"},
        step_statuses={0: "completed", 1: "failed"},
        env={"NODE_ENV": "production", "DEBUG": "true"},
        variables={"version": "2.0", "count": 42},
    )


# ── Boolean literals ─────────────────────────────────────────────────

class TestBooleanLiterals:
    def test_true(self, ctx):
        assert Condition("true").evaluate(ctx) is True

    def test_false(self, ctx):
        assert Condition("false").evaluate(ctx) is False

    def test_true_case_insensitive(self, ctx):
        assert Condition("True").evaluate(ctx) is True
        assert Condition("TRUE").evaluate(ctx) is True

    def test_false_case_insensitive(self, ctx):
        assert Condition("False").evaluate(ctx) is False


# ── Factory methods ──────────────────────────────────────────────────

class TestFactoryMethods:
    def test_always(self, ctx):
        assert Condition.always().evaluate(ctx) is True

    def test_never(self, ctx):
        assert Condition.never().evaluate(ctx) is False

    def test_step_completed(self, ctx):
        assert Condition.step_completed(0).evaluate(ctx) is True
        assert Condition.step_completed(1).evaluate(ctx) is False

    def test_step_failed(self, ctx):
        assert Condition.step_failed(1).evaluate(ctx) is True
        assert Condition.step_failed(0).evaluate(ctx) is False

    def test_step_output_contains(self, ctx):
        assert Condition.step_output_contains(0, "tests passed").evaluate(ctx) is True
        assert Condition.step_output_contains(0, "xyz").evaluate(ctx) is False

    def test_env_equals(self, ctx):
        assert Condition.env_equals("NODE_ENV", "production").evaluate(ctx) is True
        assert Condition.env_equals("NODE_ENV", "dev").evaluate(ctx) is False


# ── Reference resolution ────────────────────────────────────────────

class TestResolveReference:
    def test_step_output(self, ctx):
        assert _resolve_reference("step", "0.output", ctx) == "all tests passed"

    def test_step_status(self, ctx):
        assert _resolve_reference("step", "1.status", ctx) == "failed"

    def test_step_missing_returns_default(self, ctx):
        assert _resolve_reference("step", "99.output", ctx) == ""
        assert _resolve_reference("step", "99.status", ctx) == "pending"

    def test_env(self, ctx):
        assert _resolve_reference("env", "NODE_ENV", ctx) == "production"

    def test_env_missing(self, ctx):
        assert _resolve_reference("env", "NONEXISTENT", ctx) == ""

    def test_var(self, ctx):
        assert _resolve_reference("var", "version", ctx) == "2.0"

    def test_invalid_step_reference(self, ctx):
        with pytest.raises(ValueError, match="Invalid step reference"):
            _resolve_reference("step", "invalid", ctx)

    def test_invalid_step_index(self, ctx):
        with pytest.raises(ValueError, match="Invalid step index"):
            _resolve_reference("step", "abc.output", ctx)

    def test_unknown_step_field(self, ctx):
        with pytest.raises(ValueError, match="Unknown step field"):
            _resolve_reference("step", "0.unknown", ctx)

    def test_unknown_ref_type(self, ctx):
        with pytest.raises(ValueError, match="Unknown reference type"):
            _resolve_reference("unknown", "x", ctx)


# ── Reference substitution ──────────────────────────────────────────

class TestSubstituteReferences:
    def test_step_reference(self, ctx):
        result = _substitute_references("${step.0.status}", ctx)
        assert result == "'completed'"

    def test_env_reference(self, ctx):
        result = _substitute_references("${env.NODE_ENV}", ctx)
        assert result == "'production'"

    def test_no_references(self, ctx):
        assert _substitute_references("hello", ctx) == "hello"


# ── Expression evaluation ───────────────────────────────────────────

class TestEvaluateExpression:
    def test_equality(self, ctx):
        assert _evaluate_expression("${step.0.status} == 'completed'", ctx) is True
        assert _evaluate_expression("${step.0.status} != 'failed'", ctx) is True

    def test_inequality(self, ctx):
        assert _evaluate_expression("${step.0.status} != 'completed'", ctx) is False

    def test_membership_in(self, ctx):
        assert _evaluate_expression("'tests' in ${step.0.output}", ctx) is True

    def test_membership_not_in(self, ctx):
        assert _evaluate_expression("'xyz' not in ${step.0.output}", ctx) is True

    def test_env_comparison(self, ctx):
        assert _evaluate_expression("${env.NODE_ENV} == 'production'", ctx) is True


# ── Value parsing ────────────────────────────────────────────────────

class TestParseValue:
    def test_quoted_string(self):
        assert _parse_value("'hello'") == "hello"
        assert _parse_value('"world"') == "world"

    def test_boolean(self):
        assert _parse_value("true") is True
        assert _parse_value("false") is False

    def test_none(self):
        assert _parse_value("none") is None
        assert _parse_value("null") is None

    def test_integer(self):
        assert _parse_value("42") == 42

    def test_float(self):
        assert _parse_value("3.14") == 3.14

    def test_unquoted_string(self):
        assert _parse_value("hello") == "hello"

    def test_whitespace_stripped(self):
        assert _parse_value("  42  ") == 42


# ── Custom conditions ───────────────────────────────────────────────

class TestCustomCondition:
    def test_custom_func(self, ctx):
        cond = Condition.custom(lambda d: d["variables"]["count"] == 42)
        assert cond.evaluate(ctx) is True

    def test_custom_func_false(self, ctx):
        cond = Condition.custom(lambda d: False)
        assert cond.evaluate(ctx) is False


# ── Combinators ──────────────────────────────────────────────────────

class TestCombinators:
    def test_all_of_true(self, ctx):
        cond = all_of("true", "true")
        assert cond.evaluate(ctx) is True

    def test_all_of_false(self, ctx):
        cond = all_of("true", "false")
        assert cond.evaluate(ctx) is False

    def test_any_of_true(self, ctx):
        cond = any_of("false", "true")
        assert cond.evaluate(ctx) is True

    def test_any_of_false(self, ctx):
        cond = any_of("false", "false")
        assert cond.evaluate(ctx) is False

    def test_none_of_true(self, ctx):
        cond = none_of("false", "false")
        assert cond.evaluate(ctx) is True

    def test_none_of_false(self, ctx):
        cond = none_of("true", "false")
        assert cond.evaluate(ctx) is False

    def test_when_string(self):
        cond = when("true")
        assert isinstance(cond, Condition)

    def test_when_condition(self):
        original = Condition("true")
        cond = when(original)
        assert cond is original


# ── Error handling ───────────────────────────────────────────────────

class TestErrorHandling:
    def test_invalid_expression_defaults_true(self, ctx):
        # Invalid expression should log warning and return True
        cond = Condition("${step.invalid.bad} == 'x'")
        assert cond.evaluate(ctx) is True


# ── ConditionContext ─────────────────────────────────────────────────

class TestConditionContext:
    def test_empty_context(self):
        ctx = ConditionContext.empty()
        assert ctx.step_outputs == {}
        assert ctx.step_statuses == {}

    def test_to_dict(self, ctx):
        d = ctx.to_dict()
        assert "step_outputs" in d
        assert "step_statuses" in d
        assert "env" in d
        assert "variables" in d
        assert d["variables"]["version"] == "2.0"

    def test_get_variable_missing(self, ctx):
        assert ctx.get_variable("nonexistent") is None
