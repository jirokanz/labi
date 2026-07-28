"""Planner agent -- breaks goal into steps."""

from typing import List

from .base import BaseAgent
from labi.context.snapshot import ContextSnapshot
from labi.context.update import TaskUpdate


class PlannerAgent(BaseAgent):
    name = "planner"
    capability = "planning"

    def can_handle(self, snapshot: ContextSnapshot) -> bool:
        # Only plan once -- on a retry after a coding/validation failure,
        # snapshot.plan is already set, so skip straight to the executor
        # instead of burning a planning call (and the workflow's retry
        # budget) re-deriving the same steps.
        return snapshot.plan is None

    def process(self, snapshot: ContextSnapshot) -> TaskUpdate:
        prompt = self.prompt_builder.build_for_planning(snapshot)
        try:
            response = self._generate(prompt, label="Planning")
        except RuntimeError as e:
            return TaskUpdate(status="failed", error=str(e))

        plan = self._parse_plan(response)
        if not plan:
            return TaskUpdate(status="failed", error="Planner produced no usable steps",
                               previous_output=response)

        return TaskUpdate(
            plan=plan,
            status="planning",
            current_step=plan[0],
            previous_output=response,
        )

    def _parse_plan(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        steps = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line[0] in "-*\u2022"):
                clean = line.split(".", 1)[-1].strip().lstrip("-*\u2022 ").strip()
                if clean:
                    steps.append(clean)
        if not steps and text.strip():
            steps = [text.strip()]
        return steps
