"""Validator agent -- checks code quality and correctness."""

from .base import BaseAgent
from labi.context.snapshot import ContextSnapshot
from labi.context.update import TaskUpdate


class ValidatorAgent(BaseAgent):
    name = "validator"
    capability = "validation"

    def can_handle(self, snapshot: ContextSnapshot) -> bool:
        return bool(snapshot.artifacts)

    def process(self, snapshot: ContextSnapshot) -> TaskUpdate:
        if not snapshot.artifacts:
            return TaskUpdate(status="failed", error="No artifacts to validate")

        latest = snapshot.artifacts[-1]
        prompt = self.prompt_builder.build_for_validation(snapshot, latest)

        try:
            response = self._generate(prompt, label="Validating")
        except RuntimeError as e:
            return TaskUpdate(status="failed", error=str(e))

        if self._is_valid(response):
            return TaskUpdate(
                status="completed",
                previous_output=f"Validation passed: {response}",
                completed=True,
            )
        return TaskUpdate(
            status="coding",
            error=f"Validation failed: {response}",
            previous_output=response,
        )

    def _is_valid(self, text: str) -> bool:
        stripped = text.strip().upper()
        if stripped == "PASS" or stripped.startswith("PASS"):
            return True
        text_lower = text.lower()
        problem_keywords = (
            "fail", "error", "issue", "bug", "exception", "incorrect",
            "vulnerab", "broken", "missing", "unsafe", "security risk",
        )
        if any(kw in text_lower for kw in problem_keywords):
            return False
        return True  # default pass -- absence of a flagged problem is treated as passing
