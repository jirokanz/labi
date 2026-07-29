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
            parts.append(f"Previous attempt ({latest.name}):\n```\n{latest.content[:500]}\n```")
        # Feed the concrete reason the last attempt didn't work back into
        # the prompt -- without this, a retry just regenerates blindly
        # from the same original prompt and is likely to repeat the same
        # mistake. execution_stderr covers a crash; snapshot.error covers
        # either a crash summary or a validator's goal-mismatch reason
        # (see ValidatorAgent), whichever ran last.
        if snapshot.execution_exit_code not in (None, 0) and snapshot.execution_stderr:
            parts.append(f"The previous attempt failed when run, with this error:\n{snapshot.execution_stderr[:500]}\n"
                          "Fix the code so it runs without this error.")
        elif snapshot.error:
            parts.append(f"The previous attempt was rejected for this reason:\n{snapshot.error[:500]}\n"
                          "Address this in the new version.")
        return "\n\n".join(parts)

    def build_for_validation(self, snapshot: ContextSnapshot, artifact: Artifact) -> str:
        """Checks the code's ACTUAL output (snapshot.execution_stdout, set
        by ExecutorAgent after a real sandboxed run) against the goal --
        not just a static read-through of the code text, which can't
        catch "runs fine but produces the wrong thing"."""
        stdout_display = (snapshot.execution_stdout or "")[:1500]
        parts = [
            f"Goal: {snapshot.goal}",
            f"Code that was run:\n{artifact.content}",
            f"Output produced:\n{stdout_display}",
            "Does this output actually accomplish the stated goal? This is a real "
            "check, not a rubber stamp -- look for wrong values, missing parts of "
            "the request, or output that runs without error but doesn't answer "
            "what was asked.",
            "Respond with exactly one line: 'PASS' or 'FAIL: <one-sentence reason>'.",
        ]
        if snapshot.execution_stderr:
            parts.insert(3, f"Stderr when run:\n{snapshot.execution_stderr[:500]}")
        return "\n\n".join(parts)
