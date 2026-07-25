
from labi.core.config import Config

class MemoryDecisionPolicy:
    def __init__(self, config=None):
        self.config = config or Config()
        self.reuse = self.config.get("memory.thresholds.reuse", 0.85)
        self.adapt = self.config.get("memory.thresholds.adapt", 0.50)

    def decide(self, confidence):
        if confidence >= self.reuse:
            return "reuse"
        elif confidence >= self.adapt:
            return "adapt"
        else:
            return "regenerate"
