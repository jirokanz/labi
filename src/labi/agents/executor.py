"""Executor agent -- generates code/artifacts."""

import re

from .base import BaseAgent
from labi.context.snapshot import ContextSnapshot
from labi.context.update import TaskUpdate
from labi.context.artifact import ArtifactType

_CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


class ExecutorAgent(BaseAgent):
    name = "executor"
    capability = "coding"

    def can_handle(self, snapshot: ContextSnapshot) -> bool:
        return bool(snapshot.plan)

    def process(self, snapshot: ContextSnapshot) -> TaskUpdate:
        if not snapshot.plan:
            return TaskUpdate(status="failed", error="No plan available")

        current_step = snapshot.current_step or snapshot.plan[0]
        prompt = self.prompt_builder.build_for_coding(snapshot, current_step)

        try:
            response = self._generate(prompt, label="Coding", render="code")
        except RuntimeError as e:
            return TaskUpdate(status="failed", error=str(e))

        code = self._extract_code(response)
        artifact_id = self.prompt_builder.artifact_store.store_artifact(
            name=f"{snapshot.task_id}_output.py",
            content=code,
            artifact_type=ArtifactType.CODE,
            task_id=snapshot.task_id,
            created_by=self.name,
        )

        return TaskUpdate(
            previous_output=response,
            artifact_ids=[artifact_id],
            status="coding",
            next_step=self._get_next_step(snapshot),
        )

    def _extract_code(self, text: str) -> str:
        matches = _CODE_FENCE_RE.findall(text)
        if matches:
            return "\n".join(matches)
        return text

    def _get_next_step(self, snapshot: ContextSnapshot) -> str:
        if not snapshot.plan:
            return "complete"
        current_idx = snapshot.plan.index(snapshot.current_step) if snapshot.current_step in snapshot.plan else -1
        if current_idx >= 0 and current_idx + 1 < len(snapshot.plan):
            return snapshot.plan[current_idx + 1]
        return "validate"
