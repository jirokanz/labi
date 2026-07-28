"""Centralised prompt construction, so the exact wording each agent sends
a provider lives in one place instead of being duplicated inside every
agents/*.py process() method."""

from .snapshot import ContextSnapshot
from .artifact import Artifact, ArtifactStore


class PromptBuilder:
    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store

    def build_for_planning(self, snapshot: ContextSnapshot) -> str:
        parts = [
            "You are a planning agent. Break down the goal into clear, numbered steps.",
            f"Goal: {snapshot.goal}",
            "Output a numbered list of steps and nothing else.",
        ]
        if snapshot.facts:
            parts.append(f"Known facts: {snapshot.facts}")
        return "\n\n".join(parts)

    def build_for_coding(self, snapshot: ContextSnapshot, step: str) -> str:
        parts = [
            "You are a coding agent. Write clean, production-ready Python code "
            "in a single ```python fenced block.",
            f"Goal: {snapshot.goal}",
            f"Current step: {step}",
        ]
        if snapshot.plan:
            parts.append("Full plan:\n" + "\n".join(snapshot.plan))
        if snapshot.artifacts:
            latest = snapshot.artifacts[-1]
            parts.append(f"Previous artifact ({latest.name}):\n```\n{latest.content[:500]}\n```")
        return "\n\n".join(parts)

    def build_for_validation(self, snapshot: ContextSnapshot, artifact: Artifact) -> str:
        parts = [
            "You are a validator. Review the code below for errors, edge cases, and security issues.",
            f"Code:\n```python\n{artifact.content}\n```",
            "Reply with exactly 'PASS' if the code is correct, or describe the issues found.",
        ]
        return "\n\n".join(parts)
