"""DecisionEngine test harness.""" 

import pytest
from labi.decision.engine import DecisionEngine
from labi.decision.types import ExecutionAction, ExecutionPlan
from labi.decision.validator import validate_plan, ExecutionPlanError
from labi.intelligence.types import TaskProfile, ReuseDecision, TaskCategory, RiskLevel, RecommendedStrategy
from labi.intelligence.trace import DecisionTrace

engine = DecisionEngine()

def make_profile(risk=RiskLevel.LOW):
    return TaskProfile(category=TaskCategory.UTILITY, complexity=0.3, risk=risk, required_role="answer", recommended_strategy=RecommendedStrategy.REUSE, estimated_tokens=200)

def make_decision(action, confidence=0.9, memory_id="memory_001"):
    return ReuseDecision(action=action, confidence=confidence, reasoning="test", similar_memory_id=memory_id, provider_hint="deepseek:test")

def test_low_risk_reuse():
    plan = engine.decide(make_profile(RiskLevel.LOW), make_decision(RecommendedStrategy.REUSE))
    assert plan.action == ExecutionAction.EXECUTE_MEMORY and plan.requires_validation is False

def test_high_risk_reuse_validates():
    plan = engine.decide(make_profile(RiskLevel.HIGH), make_decision(RecommendedStrategy.REUSE))
    assert plan.requires_validation is True

def test_adapt():
    plan = engine.decide(make_profile(), make_decision(RecommendedStrategy.ADAPT))
    assert plan.action == ExecutionAction.ADAPT_MEMORY and plan.requires_coding is True

def test_generate_no_memory():
    plan = engine.decide(make_profile(), make_decision(RecommendedStrategy.GENERATE_VERIFY, confidence=0.0))
    assert plan.action == ExecutionAction.FULL_GENERATION

def test_low_confidence_fallback():
    plan = engine.decide(make_profile(), make_decision(RecommendedStrategy.REUSE, confidence=0.1))
    assert plan.action == ExecutionAction.FULL_GENERATION

def test_unknown_action_fallback():
    class Fake: pass
    fake_action = Fake()
    decision = ReuseDecision(action=fake_action, confidence=0.9, reasoning="x", similar_memory_id="mem")
    plan = engine.decide(make_profile(), decision)
    assert plan.action == ExecutionAction.FULL_GENERATION

def test_trace_survives():
    trace = DecisionTrace(stage="test", notes=["original"])
    decision = make_decision(RecommendedStrategy.REUSE)
    decision.trace = trace
    engine.decide(make_profile(), decision)
    assert decision.trace.stage == "test"

def test_validator_catches_bad_execute():
    with pytest.raises(ExecutionPlanError):
        validate_plan(ExecutionPlan(ExecutionAction.EXECUTE_MEMORY, None, None, False, False, False, "x"))

def test_validator_catches_bad_adapt():
    with pytest.raises(ExecutionPlanError):
        validate_plan(ExecutionPlan(ExecutionAction.ADAPT_MEMORY, None, "mem", False, False, False, "x"))

def test_validator_catches_bad_gen():
    with pytest.raises(ExecutionPlanError):
        validate_plan(ExecutionPlan(ExecutionAction.FULL_GENERATION, None, "mem", True, True, True, "x"))
