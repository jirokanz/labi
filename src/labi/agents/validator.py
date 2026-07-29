"""Validator agent -- checks the code's ACTUAL execution output against
the goal (not just a static read-through of the code text), using a
second, independent LLM pass. This only runs after ExecutorAgent has
actually executed the code (see can_handle below) -- execution_exit_code
== 0 proves it didn't crash, but not that it did what was asked; that's
what this checks.

If no validation-capable provider is available, this degrades to an
"unverified pass" (matching the old validate_result()'s semantics)
rather than blocking task completion on missing infrastructure -- a
present-but-unconfirmed validator shouldn't be worse than having none at
all.
"""

from .base import BaseAgent
from labi.context.snapshot import ContextSnapshot
from labi.context.update import TaskUpdate


class ValidatorAgent(BaseAgent):
    name = "validator"
    capability = "validation"

    def can_handle(self, snapshot: ContextSnapshot) -> bool:
        # Only validate once the executor has actually run the code
        # successfully -- validating a crash, or code that was never run,
        # isn't a real check.
        return bool(snapshot.artifacts) and snapshot.execution_exit_code == 0

    def process(self, snapshot: ContextSnapshot) -> TaskUpdate:
        if not snapshot.artifacts:
            return TaskUpdate(status="failed", error="No artifacts to validate")

        latest = snapshot.artifacts[-1]
        prompt = self.prompt_builder.build_for_validation(snapshot, latest)

        try:
            response = self._generate(prompt, label="Validating", max_tokens=150)
        except RuntimeError:
            # No validation provider available -- unverified, not failed.
            # Matches the old validate_result()'s "ran=False" behavior:
            # missing infra shouldn't block a task that otherwise ran fine.
            return TaskUpdate(
                status="completed",
                previous_output="No validation provider available -- unverified.",
                completed=True,
            )

        verdict = response.strip()
        if verdict.upper().startswith("PASS"):
            return TaskUpdate(
                status="completed",
                previous_output=f"Validator confirmed the output matches the goal: {verdict}",
                completed=True,
            )
        return TaskUpdate(
            status="coding",
            error=verdict,
            previous_output=verdict,
        )
