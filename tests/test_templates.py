"""Tests for copex.templates — StepTemplate, TemplateRegistry, workflow builders."""

from __future__ import annotations

import pytest

from copex.templates import (
    BUILD_PROJECT,
    COMMIT_CHANGES,
    DEPLOY,
    HEALTH_CHECK,
    INSTALL_DEPENDENCIES,
    LINT_CODE,
    ROLLBACK,
    RUN_SPECIFIC_TEST,
    RUN_TESTS,
    SETUP_TEST_ENV,
    StepInstance,
    StepTemplate,
    TemplateRegistry,
    build_workflow,
    create_step,
    deploy_workflow,
    get_registry,
    test_workflow as make_test_workflow,
)


# ---------------------------------------------------------------------------
# StepTemplate
# ---------------------------------------------------------------------------


class TestStepTemplate:
    def test_instantiate(self) -> None:
        t = StepTemplate(
            name="{name}_step",
            description_template="Run {thing}",
            prompt_template="Do the {thing} now",
            tags=["test"],
        )
        inst = t.instantiate(name="my", thing="build")
        assert isinstance(inst, StepInstance)
        assert inst.name == "my_step"
        assert inst.description == "Run build"
        assert inst.prompt == "Do the build now"
        assert "test" in inst.tags
        assert inst.metadata["template"] == "{name}_step"

    def test_instantiate_copies_depends(self) -> None:
        t = StepTemplate(
            name="t",
            description_template="d",
            prompt_template="p",
            depends_on=["a", "b"],
        )
        inst = t.instantiate()
        inst.depends_on.append("c")
        assert "c" not in t.depends_on  # Original not mutated

    def test_missing_placeholder_raises(self) -> None:
        t = StepTemplate(
            name="t",
            description_template="{missing}",
            prompt_template="p",
        )
        with pytest.raises(KeyError):
            t.instantiate()


# ---------------------------------------------------------------------------
# TemplateRegistry
# ---------------------------------------------------------------------------


class TestTemplateRegistry:
    def test_builtins_registered(self) -> None:
        reg = TemplateRegistry()
        assert reg.get("run_tests") is not None
        assert reg.get("deploy") is not None
        assert reg.get("build_project") is not None

    def test_register_custom(self) -> None:
        reg = TemplateRegistry()
        custom = StepTemplate(name="custom", description_template="c", prompt_template="p")
        reg.register(custom)
        assert reg.get("custom") is custom

    def test_get_missing_returns_none(self) -> None:
        reg = TemplateRegistry()
        assert reg.get("nonexistent") is None

    def test_list_all(self) -> None:
        reg = TemplateRegistry()
        all_templates = reg.list()
        assert len(all_templates) >= 10  # At least the builtins

    def test_list_by_tag(self) -> None:
        reg = TemplateRegistry()
        testing = reg.list(tag="testing")
        assert all("testing" in t.tags for t in testing)
        assert len(testing) >= 2  # RUN_TESTS, RUN_SPECIFIC_TEST, SETUP_TEST_ENV

    def test_create_step(self) -> None:
        reg = TemplateRegistry()
        step = reg.create_step(
            "run_tests",
            test_framework="pytest",
            directory="tests/",
            options="-v",
            command="pytest tests/ -v",
        )
        assert step.name == "run_tests"
        assert "pytest" in step.prompt

    def test_create_step_missing_template(self) -> None:
        reg = TemplateRegistry()
        with pytest.raises(KeyError, match="Template not found"):
            reg.create_step("no_such_template")


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------


class TestGlobalRegistry:
    def test_get_registry(self) -> None:
        reg = get_registry()
        assert isinstance(reg, TemplateRegistry)
        assert reg.get("run_tests") is not None

    def test_create_step_convenience(self) -> None:
        step = create_step(
            "lint_code",
            linter="ruff",
            config_file="pyproject.toml",
            paths="src",
            lint_command="ruff check src",
        )
        assert step.name == "lint_code"


# ---------------------------------------------------------------------------
# Built-in templates sanity
# ---------------------------------------------------------------------------


class TestBuiltinTemplates:
    def test_run_tests(self) -> None:
        inst = RUN_TESTS.instantiate(
            test_framework="pytest", directory=".", options="-v", command="pytest . -v"
        )
        assert "pytest" in inst.prompt

    def test_deploy(self) -> None:
        inst = DEPLOY.instantiate(
            environment="prod",
            target="us-east-1",
            version="1.2.3",
            deploy_command="deploy --env prod",
        )
        assert "prod" in inst.description

    def test_rollback_has_condition(self) -> None:
        assert ROLLBACK.condition is not None

    def test_health_check(self) -> None:
        inst = HEALTH_CHECK.instantiate(
            service="api",
            health_endpoint="/healthz",
            expected_status="200",
            timeout="30s",
        )
        assert "/healthz" in inst.prompt


# ---------------------------------------------------------------------------
# Workflow builders
# ---------------------------------------------------------------------------


class TestWorkflowBuilders:
    def test_test_workflow(self) -> None:
        steps = make_test_workflow(framework="pytest", directory="tests/", options="-v")
        assert len(steps) == 2
        assert steps[0].name == "install_dependencies"
        assert steps[1].name == "run_tests"

    def test_build_workflow_full(self) -> None:
        steps = build_workflow(
            project_name="myapp",
            build_command="make build",
            lint=True,
            type_check=True,
        )
        names = [s.name for s in steps]
        assert "install_dependencies" in names
        assert "lint_code" in names
        assert "type_check" in names
        assert "build_project" in names

    def test_build_workflow_no_lint_no_typecheck(self) -> None:
        steps = build_workflow(
            project_name="myapp",
            build_command="make build",
            lint=False,
            type_check=False,
        )
        names = [s.name for s in steps]
        assert "lint_code" not in names
        assert "type_check" not in names

    def test_deploy_workflow_with_health(self) -> None:
        steps = deploy_workflow(
            environment="staging",
            target="us-west-2",
            version="2.0.0",
            health_endpoint="/health",
        )
        names = [s.name for s in steps]
        assert "deploy" in names
        assert "health_check" in names
        assert "rollback" in names

    def test_deploy_workflow_without_health(self) -> None:
        steps = deploy_workflow(
            environment="prod",
            target="eu-west-1",
            version="3.0.0",
        )
        names = [s.name for s in steps]
        assert "health_check" not in names
        assert "rollback" in names
