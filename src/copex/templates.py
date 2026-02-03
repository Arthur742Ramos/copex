"""
Step Templates - Reusable templates for common workflows.

Provides pre-built templates for:
- Testing workflows
- Build processes
- Deployment pipelines
- Code quality checks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from copex.conditions import Condition


@dataclass
class StepTemplate:
    """A reusable step template.

    Templates define the structure of a step with placeholders
    that can be filled in when instantiated.
    """

    name: str
    description_template: str  # Can contain {placeholders}
    prompt_template: str       # Main instruction template
    condition: Condition | None = None
    depends_on: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def instantiate(self, **kwargs: Any) -> "StepInstance":
        """Create a concrete step instance from this template.

        Args:
            **kwargs: Values for placeholders in the template

        Returns:
            A StepInstance with all placeholders filled
        """
        return StepInstance(
            name=self.name.format(**kwargs),
            description=self.description_template.format(**kwargs),
            prompt=self.prompt_template.format(**kwargs),
            condition=self.condition,
            depends_on=self.depends_on.copy(),
            tags=self.tags.copy(),
            metadata={**self.metadata, "template": self.name},
        )


@dataclass
class StepInstance:
    """A concrete step created from a template."""

    name: str
    description: str
    prompt: str
    condition: Condition | None = None
    depends_on: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Built-in Templates: Testing
# ============================================================================

RUN_TESTS = StepTemplate(
    name="run_tests",
    description_template="Run {test_framework} tests in {directory}",
    prompt_template="""Run the test suite using {test_framework}.

Directory: {directory}
Options: {options}

Execute the tests and report:
1. Total tests run
2. Passed/Failed/Skipped counts
3. Any failures with details
4. Coverage if available

Command: {command}""",
    tags=["testing", "verification"],
)

RUN_SPECIFIC_TEST = StepTemplate(
    name="run_specific_test",
    description_template="Run test: {test_name}",
    prompt_template="""Run a specific test or test file.

Test: {test_name}
Framework: {test_framework}
Options: {options}

Execute and report the result.""",
    tags=["testing"],
)

SETUP_TEST_ENV = StepTemplate(
    name="setup_test_env",
    description_template="Set up test environment",
    prompt_template="""Set up the testing environment.

Steps:
1. Install test dependencies: {dependencies}
2. Create test fixtures if needed
3. Set up test database/mocks: {setup_commands}
4. Verify environment is ready

Report any issues encountered.""",
    tags=["testing", "setup"],
)

# ============================================================================
# Built-in Templates: Building
# ============================================================================

BUILD_PROJECT = StepTemplate(
    name="build_project",
    description_template="Build {project_name}",
    prompt_template="""Build the project.

Project: {project_name}
Build command: {build_command}
Environment: {environment}

Execute the build and report:
1. Build success/failure
2. Any warnings or errors
3. Build artifacts created
4. Build time""",
    tags=["build"],
)

INSTALL_DEPENDENCIES = StepTemplate(
    name="install_dependencies",
    description_template="Install dependencies",
    prompt_template="""Install project dependencies.

Package manager: {package_manager}
Lockfile: {lockfile}
Options: {options}

Run: {install_command}

Report any issues with dependency resolution.""",
    tags=["build", "setup"],
)

LINT_CODE = StepTemplate(
    name="lint_code",
    description_template="Lint code with {linter}",
    prompt_template="""Run linting on the codebase.

Linter: {linter}
Config: {config_file}
Paths: {paths}

Command: {lint_command}

Report:
1. Number of issues found
2. Issues by severity
3. Auto-fixable issues""",
    tags=["build", "quality"],
)

FORMAT_CODE = StepTemplate(
    name="format_code",
    description_template="Format code with {formatter}",
    prompt_template="""Format the codebase.

Formatter: {formatter}
Config: {config_file}
Paths: {paths}

Command: {format_command}

Report files modified.""",
    tags=["build", "quality"],
)

TYPE_CHECK = StepTemplate(
    name="type_check",
    description_template="Type check with {type_checker}",
    prompt_template="""Run type checking on the codebase.

Type checker: {type_checker}
Config: {config_file}
Strictness: {strictness}

Command: {check_command}

Report type errors found.""",
    tags=["build", "quality"],
)

# ============================================================================
# Built-in Templates: Deployment
# ============================================================================

DEPLOY = StepTemplate(
    name="deploy",
    description_template="Deploy to {environment}",
    prompt_template="""Deploy the application.

Environment: {environment}
Target: {target}
Version: {version}

Steps:
1. Verify build artifacts exist
2. Run pre-deployment checks
3. Deploy: {deploy_command}
4. Verify deployment health

Report deployment status.""",
    tags=["deploy"],
)

ROLLBACK = StepTemplate(
    name="rollback",
    description_template="Rollback {environment} to {version}",
    prompt_template="""Rollback the deployment.

Environment: {environment}
Target version: {version}
Reason: {reason}

Execute rollback and verify system health.""",
    condition=Condition.step_failed(1),  # Only if previous step failed
    tags=["deploy", "recovery"],
)

HEALTH_CHECK = StepTemplate(
    name="health_check",
    description_template="Health check {service}",
    prompt_template="""Verify service health.

Service: {service}
Endpoint: {health_endpoint}
Expected status: {expected_status}
Timeout: {timeout}

Check and report health status.""",
    tags=["deploy", "verification"],
)

# ============================================================================
# Built-in Templates: Git/Version Control
# ============================================================================

COMMIT_CHANGES = StepTemplate(
    name="commit_changes",
    description_template="Commit: {message}",
    prompt_template="""Commit changes to git.

Message: {message}
Files: {files}
Options: {options}

Run:
1. git add {files}
2. git commit -m "{message}"

Report commit hash.""",
    tags=["git"],
)

CREATE_PR = StepTemplate(
    name="create_pr",
    description_template="Create PR: {title}",
    prompt_template="""Create a pull request.

Title: {title}
Base: {base_branch}
Head: {head_branch}
Body: {body}

Use: {pr_command}

Report PR URL.""",
    tags=["git", "collaboration"],
)

# ============================================================================
# Template Registry
# ============================================================================

class TemplateRegistry:
    """Registry of available step templates."""

    def __init__(self) -> None:
        self._templates: dict[str, StepTemplate] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in templates."""
        builtins = [
            # Testing
            RUN_TESTS,
            RUN_SPECIFIC_TEST,
            SETUP_TEST_ENV,
            # Building
            BUILD_PROJECT,
            INSTALL_DEPENDENCIES,
            LINT_CODE,
            FORMAT_CODE,
            TYPE_CHECK,
            # Deployment
            DEPLOY,
            ROLLBACK,
            HEALTH_CHECK,
            # Git
            COMMIT_CHANGES,
            CREATE_PR,
        ]
        for template in builtins:
            self.register(template)

    def register(self, template: StepTemplate) -> None:
        """Register a template."""
        self._templates[template.name] = template

    def get(self, name: str) -> StepTemplate | None:
        """Get a template by name."""
        return self._templates.get(name)

    def list(self, tag: str | None = None) -> list[StepTemplate]:
        """List templates, optionally filtered by tag."""
        templates = list(self._templates.values())
        if tag:
            templates = [t for t in templates if tag in t.tags]
        return templates

    def create_step(self, template_name: str, **kwargs: Any) -> StepInstance:
        """Create a step instance from a template.

        Args:
            template_name: Name of the template to use
            **kwargs: Values for template placeholders

        Returns:
            A StepInstance

        Raises:
            KeyError: If template not found
        """
        template = self._templates.get(template_name)
        if not template:
            raise KeyError(f"Template not found: {template_name}")
        return template.instantiate(**kwargs)


# Global registry
_registry = TemplateRegistry()


def get_registry() -> TemplateRegistry:
    """Get the global template registry."""
    return _registry


def create_step(template_name: str, **kwargs: Any) -> StepInstance:
    """Create a step from a template (convenience function)."""
    return _registry.create_step(template_name, **kwargs)


# ============================================================================
# Workflow Builders
# ============================================================================

def test_workflow(
    *,
    framework: str = "pytest",
    directory: str = ".",
    options: str = "-v",
) -> list[StepInstance]:
    """Create a standard test workflow.

    Args:
        framework: Test framework to use
        directory: Test directory
        options: Additional options

    Returns:
        List of step instances for the workflow
    """
    return [
        INSTALL_DEPENDENCIES.instantiate(
            package_manager="pip",
            lockfile="requirements.txt",
            options="",
            install_command="pip install -r requirements.txt",
        ),
        RUN_TESTS.instantiate(
            test_framework=framework,
            directory=directory,
            options=options,
            command=f"{framework} {directory} {options}",
        ),
    ]


def build_workflow(
    *,
    project_name: str,
    build_command: str,
    lint: bool = True,
    type_check: bool = True,
) -> list[StepInstance]:
    """Create a standard build workflow.

    Args:
        project_name: Name of the project
        build_command: Command to build the project
        lint: Whether to include linting
        type_check: Whether to include type checking

    Returns:
        List of step instances for the workflow
    """
    steps = [
        INSTALL_DEPENDENCIES.instantiate(
            package_manager="pip",
            lockfile="requirements.txt",
            options="",
            install_command="pip install -r requirements.txt",
        ),
    ]

    if lint:
        steps.append(LINT_CODE.instantiate(
            linter="ruff",
            config_file="pyproject.toml",
            paths="src",
            lint_command="ruff check src",
        ))

    if type_check:
        steps.append(TYPE_CHECK.instantiate(
            type_checker="mypy",
            config_file="pyproject.toml",
            strictness="strict",
            check_command="mypy src",
        ))

    steps.append(BUILD_PROJECT.instantiate(
        project_name=project_name,
        build_command=build_command,
        environment="production",
    ))

    return steps


def deploy_workflow(
    *,
    environment: str,
    target: str,
    version: str,
    health_endpoint: str | None = None,
) -> list[StepInstance]:
    """Create a standard deployment workflow.

    Args:
        environment: Target environment
        target: Deployment target
        version: Version to deploy
        health_endpoint: Health check endpoint

    Returns:
        List of step instances for the workflow
    """
    steps = [
        DEPLOY.instantiate(
            environment=environment,
            target=target,
            version=version,
            deploy_command=f"deploy --env {environment} --version {version}",
        ),
    ]

    if health_endpoint:
        steps.append(HEALTH_CHECK.instantiate(
            service=target,
            health_endpoint=health_endpoint,
            expected_status="200",
            timeout="30s",
        ))

    # Add rollback as conditional step
    steps.append(ROLLBACK.instantiate(
        environment=environment,
        version="previous",
        reason="Deployment failed",
    ))

    return steps
