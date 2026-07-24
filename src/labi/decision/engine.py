"""DecisionEngine - Maps profile + decision to execution plan.""" 

from labi.intelligence.types import TaskProfile, ReuseDecision, RecommendedStrategy, RiskLevel
from labi.decision.types import ExecutionAction, ExecutionPlan

class DecisionEngine:
    def decide(self, profile: TaskProfile, reuse_decision: ReuseDecision) -> ExecutionPlan:
        if reuse_decision.action == RecommendedStrategy.GENERATE_VERIFY or reuse_decision.confidence < 0.2:
            return ExecutionPlan(
                action=ExecutionAction.FULL_GENERATION,
                provider=reuse_decision.provider_hint,
                memory_id=None,
                requires_planning=True,
                requires_coding=True,
                requires_validation=True,
                reason="No trusted memory. Full generation pipeline required."
            )
        if reuse_decision.action == RecommendedStrategy.ADAPT:
            return ExecutionPlan(
                action=ExecutionAction.ADAPT_MEMORY,
                provider=reuse_decision.provider_hint,
                memory_id=reuse_decision.similar_memory_id,
                requires_planning=False,
                requires_coding=True,
                requires_validation=True,
                reason="Adapting existing memory."
            )
        if reuse_decision.action == RecommendedStrategy.REUSE:
            if profile.risk in (RiskLevel.HIGH, RiskLevel.MEDIUM):
                return ExecutionPlan(
                    action=ExecutionAction.EXECUTE_MEMORY,
                    provider=None,
                    memory_id=reuse_decision.similar_memory_id,
                    requires_planning=False,
                    requires_coding=False,
                    requires_validation=True,
                    reason="High/medium risk triggers validation."
                )
            return ExecutionPlan(
                action=ExecutionAction.EXECUTE_MEMORY,
                provider=None,
                memory_id=reuse_decision.similar_memory_id,
                requires_planning=False,
                requires_coding=False,
                requires_validation=False,
                reason="Direct memory reuse (low risk)."
            )
        return ExecutionPlan(
            action=ExecutionAction.FULL_GENERATION,
            provider=reuse_decision.provider_hint,
            memory_id=None,
            requires_planning=True,
            requires_coding=True,
            requires_validation=True,
            reason="Fallback: unhandled decision state."
        )
