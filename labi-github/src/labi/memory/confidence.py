
import re
from labi.core.config import Config

class ConfidenceEvaluator:
    def __init__(self, config=None):
        self.config = config or Config()
        self.sim_w = self.config.get("memory.confidence.similarity_weight", 0.4)
        self.succ_w = self.config.get("memory.confidence.success_weight", 0.3)
        self.rep_w = self.config.get("memory.confidence.replay_weight", 0.3)

    @staticmethod
    def tokenize(text):
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def goal_similarity(self, a, b):
        wa, wb = self.tokenize(a), self.tokenize(b)
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    def calculate_replay_rate(self, task):
        total = task.get("replay_count", 0)
        if total == 0:
            return 0.5
        return task.get("replay_success_count", 0) / total

    def evaluate(self, goal, candidate):
        sim = self.goal_similarity(goal, candidate.get("goal", ""))
        hist = 1.0 if candidate.get("success") else 0.0
        replay = self.calculate_replay_rate(candidate)
        confidence = sim*self.sim_w + hist*self.succ_w + replay*self.rep_w
        return {
            "goal_similarity": sim,
            "historical_success": hist,
            "replay_success_rate": replay,
            "confidence_score": confidence
        }
