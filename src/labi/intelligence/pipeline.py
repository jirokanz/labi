"""IntelligencePipeline - Orchestrates the full decision chain.""" 

from dataclasses import dataclass

from labi.intelligence.classifier import TaskClassifier
from labi.intelligence.types import TaskProfile, ReuseDecision
from labi.memory.router import MemoryRouter
from labi.decision.engine import DecisionEngine
from labi.decision.types import ExecutionPlan

@dataclass
class IntelligenceResult:
    goal: str
    profile: TaskProfile
    decision: ReuseDecision
    plan: ExecutionPlan

class IntelligencePipeline:
    def __init__(self, classifier: TaskClassifier, router: MemoryRouter, engine: DecisionEngine):
        self.classifier = classifier
        self.router = router
        self.engine = engine

    def process(self, goal: str) -> IntelligenceResult:
        profile = self.classifier.classify(goal)
        decision = self.router.decide(goal, profile)
        plan = self.engine.decide(profile, decision)
        return IntelligenceResult(goal=goal, profile=profile, decision=decision, plan=plan)
