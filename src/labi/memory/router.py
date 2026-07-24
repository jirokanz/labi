
from labi.memory.database import MemoryDatabase
from labi.memory.confidence import ConfidenceEvaluator
from labi.memory.policy import MemoryDecisionPolicy
from labi.memory.guard import ReplayGuard

class MemoryRouter:
    def __init__(self, db, evaluator=None, policy=None, guard=None):
        self.db = db
        self.evaluator = evaluator or ConfidenceEvaluator()
        self.policy = policy or MemoryDecisionPolicy()
        self.guard = guard or ReplayGuard()

    def route(self, goal, current_task_id=None):
        candidates = self.db.find_similar_goals(goal, limit=3)
        if current_task_id:
            candidates = [c for c in candidates if c["id"] != current_task_id]
        if not candidates:
            return {"decision": "regenerate", "confidence": 0.0, "candidate": None}

        best = None
        best_score = -1.0
        best_metrics = {}
        for cand in candidates:
            metrics = self.evaluator.evaluate(goal, cand)
            score = metrics["confidence_score"]
            if score > best_score:
                best_score = score
                best = cand
                best_metrics = metrics

        decision = self.policy.decide(best_score)
        guard_result = self.guard.check(best)
        if decision == "reuse" and not guard_result["allowed"]:
            decision = "adapt"

        return {
            "decision": decision,
            "confidence": best_score,
            "candidate": best,
            "goal_similarity": best_metrics.get("goal_similarity", 0.0),
            "guard_blocked": not guard_result["allowed"],
            "guard_reason": guard_result["reason"]
        }
