"""
Ralph Wiggum - Iterative Development Loops for Copex.

Implements the Ralph Wiggum technique pioneered by Geoffrey Huntley:
https://ghuntley.com/ralph/

Core concept: The same prompt is fed to the AI repeatedly. The AI sees its
own previous work in the conversation history, allowing it to iteratively
improve until the task is complete.

"Me fail English? That's unpossible!" - Ralph Wiggum
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from copex.models import EventType


@dataclass
class RalphState:
    """State of a Ralph loop."""

    prompt: str
    iteration: int = 0
    max_iterations: int | None = None
    completion_promise: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    completion_reason: str | None = None
    history: list[str] = field(default_factory=list)


@dataclass
class RalphConfig:
    """Configuration for Ralph loops."""

    max_iterations: int | None = None  # None = unlimited
    completion_promise: str | None = None  # Text that signals completion
    delay_between_iterations: float = 1.0  # Seconds between iterations
    show_iteration_header: bool = True
    continue_on_error: bool = True
    max_consecutive_errors: int = 3


class RalphWiggum:
    """
    Ralph Wiggum - Iterative AI development loops.

    Usage:
        ralph = RalphWiggum(copex_client)
        result = await ralph.loop(
            prompt="Build a REST API with CRUD operations",
            completion_promise="API COMPLETE",
            max_iterations=30,
        )
    """

    def __init__(self, client: Any, config: RalphConfig | None = None):
        """
        Initialize Ralph.

        Args:
            client: A Copex client instance
            config: Optional configuration
        """
        self.client = client
        self.config = config or RalphConfig()
        self._state: RalphState | None = None
        self._cancelled = False

    @property
    def state(self) -> RalphState | None:
        """Get current loop state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if a loop is currently running."""
        return self._state is not None and not self._state.completed

    def cancel(self) -> None:
        """Cancel the current loop."""
        self._cancelled = True
        if self._state:
            self._state.completed = True
            self._state.completion_reason = "cancelled"

    async def loop(
        self,
        prompt: str,
        *,
        max_iterations: int | None = None,
        completion_promise: str | None = None,
        on_iteration: Callable[[int, str], None] | None = None,
        on_complete: Callable[[RalphState], None] | None = None,
    ) -> RalphState:
        """
        Run a Ralph loop.

        The same prompt is sent repeatedly. The AI sees the conversation
        history and can iterate on its previous work.

        Args:
            prompt: The task prompt to repeat
            max_iterations: Maximum iterations (None = unlimited)
            completion_promise: Text that signals completion (e.g., "TASK COMPLETE")
            on_iteration: Callback called after each iteration with (iteration, response)
            on_complete: Callback called when loop completes

        Returns:
            RalphState with final state and history
        """
        self._cancelled = False
        self._state = RalphState(
            prompt=prompt,
            max_iterations=max_iterations or self.config.max_iterations,
            completion_promise=completion_promise or self.config.completion_promise,
        )

        consecutive_errors = 0

        # Build the system message for Ralph
        ralph_instructions = self._build_ralph_instructions()

        try:
            while not self._cancelled:
                self._state.iteration += 1

                # Check max iterations
                if self._state.max_iterations and self._state.iteration > self._state.max_iterations:
                    self._state.completed = True
                    self._state.completion_reason = f"max_iterations ({self._state.max_iterations})"
                    break

                # Build iteration prompt
                if self.config.show_iteration_header:
                    iteration_prompt = f"{ralph_instructions}\n\n---\n\n{prompt}"
                else:
                    iteration_prompt = prompt

                try:
                    # Send to Copex
                    response = await self.client.send(iteration_prompt)
                    content = response.content
                    self._state.history.append(content)
                    consecutive_errors = 0

                    # Check for completion promise
                    if self._state.completion_promise:
                        if self._check_promise(content, self._state.completion_promise):
                            self._state.completed = True
                            self._state.completion_reason = f"promise: {self._state.completion_promise}"
                            break

                    # Callback
                    if on_iteration:
                        on_iteration(self._state.iteration, content)

                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors >= self.config.max_consecutive_errors:
                        self._state.completed = True
                        self._state.completion_reason = f"errors: {consecutive_errors} consecutive failures"
                        break

                    if not self.config.continue_on_error:
                        raise

                # Delay between iterations
                if self.config.delay_between_iterations > 0:
                    await asyncio.sleep(self.config.delay_between_iterations)

        finally:
            if not self._state.completed:
                self._state.completed = True
                self._state.completion_reason = "unknown"

            if on_complete:
                on_complete(self._state)

        return self._state

    def _build_ralph_instructions(self) -> str:
        """Build system instructions for the current iteration."""
        parts = [f"[Ralph Wiggum Loop - Iteration {self._state.iteration}]"]

        if self._state.max_iterations:
            parts.append(f"Max iterations: {self._state.max_iterations}")

        if self._state.completion_promise:
            parts.append(
                f"\nTo complete this loop, output: <promise>{self._state.completion_promise}</promise>\n"
                f"ONLY output this when the statement is genuinely TRUE."
            )
        else:
            parts.append("\nNo completion promise set - loop runs until max iterations or cancelled.")

        parts.append("\nYou can see your previous work in the conversation. Continue improving.")

        return "\n".join(parts)

    def _check_promise(self, content: str, promise: str) -> bool:
        """Check if content contains the completion promise."""
        # Look for <promise>TEXT</promise> pattern
        pattern = r"<promise>(.*?)</promise>"
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

        for match in matches:
            # Normalize whitespace
            normalized = " ".join(match.strip().split())
            if normalized.lower() == promise.lower():
                return True

        return False


async def ralph_loop(
    client: Any,
    prompt: str,
    *,
    max_iterations: int | None = None,
    completion_promise: str | None = None,
    on_iteration: Callable[[int, str], None] | None = None,
) -> RalphState:
    """
    Convenience function to run a Ralph loop.

    Example:
        async with Copex() as client:
            result = await ralph_loop(
                client,
                "Build a calculator app with tests",
                completion_promise="ALL TESTS PASSING",
                max_iterations=20,
            )
            print(f"Completed in {result.iteration} iterations")
    """
    ralph = RalphWiggum(client)
    return await ralph.loop(
        prompt,
        max_iterations=max_iterations,
        completion_promise=completion_promise,
        on_iteration=on_iteration,
    )

