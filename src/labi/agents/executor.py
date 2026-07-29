"""Executor agent -- generates code, then actually runs it in the sandbox
(tools/python/sandbox.py) rather than only asking an LLM to review the
text. A code-review-only "validator" can't catch "runs fine but produces
the wrong thing" or "crashes on line 12" -- only running it can.

Preserves the one safety gate that matters most from the old interactive
CLI loop: a goal the heuristic risk classifier flags HIGH RISK (delete,
filesystem, database, payment keywords -- see intelligence/classifier.py)
still requires explicit confirmation before its code is ever executed,
even though the rest of the old per-attempt "run/edit/feedback/skip"
interactive menu is gone in favor of automatic execute+validate+retry
(that's what a "workflow" means -- see workflows/software_dev.py). The
confirmation function is injectable so tests (and any non-interactive
caller) don't have to block on real stdin.
"""

from .base import BaseAgent
from labi.context.snapshot import ContextSnapshot
from labi.context.update import TaskUpdate
from labi.context.artifact import ArtifactType
from labi.intelligence.classifier import TaskClassifier
from labi.intelligence.types import RiskLevel
from labi.providers.generation import _c, format_code_block
from labi.tools.python.sandbox import extract_code, execute_code


def _matched_risk_keywords(classifier, goal):
    """The words that actually triggered the risk flag -- TaskProfile.keywords
    is complexity keywords (a different list), not risk ones, so using it
    for this message would show the wrong words."""
    text = (goal or "").lower()
    return [kw for kw in classifier.RISK_KEYWORDS if kw in text]


def _default_confirm_high_risk(goal, keywords):
    kw = ", ".join(keywords) if keywords else "goal wording"
    print(_c(f"   Risk assessment: HIGH (flagged on: {kw})", "yellow"))
    answer = input(_c(
        "   This goal was flagged HIGH RISK. Type 'yes' to run the generated code anyway: ",
        "yellow")).strip().lower()
    return answer == "yes"


class ExecutorAgent(BaseAgent):
    name = "executor"
    capability = "coding"

    def __init__(self, registry, prompt_builder, stats_store=None, cost_tracker=None,
                 confirm_high_risk_fn=None):
        super().__init__(registry, prompt_builder, stats_store=stats_store, cost_tracker=cost_tracker)
        # Injectable so agents can run non-interactively in tests (or any
        # future non-CLI caller) without blocking on real stdin.
        self.confirm_high_risk_fn = confirm_high_risk_fn or _default_confirm_high_risk

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

        code = extract_code(response)
        print(format_code_block(code, title=(snapshot.goal or "")[:60]))

        risk_profile = TaskClassifier().classify(snapshot.goal)
        risk_keywords = _matched_risk_keywords(TaskClassifier, snapshot.goal)
        if risk_profile.risk == RiskLevel.HIGH:
            if not self.confirm_high_risk_fn(snapshot.goal, risk_keywords):
                return TaskUpdate(
                    status="failed",
                    error="Declined to run HIGH RISK code -- not executed.",
                )
        elif risk_profile.risk == RiskLevel.MEDIUM:
            kw = ", ".join(risk_keywords) if risk_keywords else "goal wording"
            print(_c(f"   Risk assessment: MEDIUM (flagged on: {kw})", "yellow"))

        artifact_id = self.prompt_builder.artifact_store.store_artifact(
            name=f"{snapshot.task_id}_output.py",
            content=code,
            artifact_type=ArtifactType.CODE,
            task_id=snapshot.task_id,
            created_by=self.name,
        )

        print(_c("   Executing (sandboxed)...", "cyan"))
        stdout, stderr, exit_code = execute_code(code)

        if exit_code != 0:
            print(_c(f"   Execution failed (exit code {exit_code}).", "red"))
            return TaskUpdate(
                previous_output=response,
                artifact_ids=[artifact_id],
                status="coding",
                error=f"Execution failed (exit code {exit_code}): {stderr[:300]}",
                execution_stdout=stdout,
                execution_stderr=stderr,
                execution_exit_code=exit_code,
            )

        print(_c("   Execution succeeded.", "green"))
        return TaskUpdate(
            previous_output=stdout,
            artifact_ids=[artifact_id],
            status="coding",
            next_step=self._get_next_step(snapshot),
            execution_stdout=stdout,
            execution_stderr=stderr,
            execution_exit_code=exit_code,
        )

    def _get_next_step(self, snapshot: ContextSnapshot) -> str:
        if not snapshot.plan:
            return "complete"
        current_idx = snapshot.plan.index(snapshot.current_step) if snapshot.current_step in snapshot.plan else -1
        if current_idx >= 0 and current_idx + 1 < len(snapshot.plan):
            return snapshot.plan[current_idx + 1]
        return "validate"
