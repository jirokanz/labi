"""TaskClassifier - Heuristic-based, zero-LLM cost.""" 

from labi.intelligence.types import TaskProfile, TaskCategory, RiskLevel, RecommendedStrategy

class TaskClassifier:
    COMPLEXITY_KEYWORDS = {
        "distributed": 0.25, "architecture": 0.20, "scalable": 0.20,
        "websocket": 0.15, "database": 0.10, "security": 0.20,
        "async": 0.10, "arbitrage": 0.20, "realtime": 0.15, "parallel": 0.15
    }
    RISK_KEYWORDS = {
        "delete": RiskLevel.HIGH, "filesystem": RiskLevel.HIGH,
        "database": RiskLevel.HIGH, "payment": RiskLevel.HIGH,
        "drop": RiskLevel.HIGH, "api": RiskLevel.MEDIUM,
        "network": RiskLevel.MEDIUM, "scrape": RiskLevel.MEDIUM
    }
    CATEGORY_KEYWORDS = {
        TaskCategory.CODING: ["code", "python", "function", "script", "algorithm"],
        TaskCategory.AUTOMATION: ["scrape", "automate", "bot", "monitor", "cron"],
        TaskCategory.RESEARCH: ["research", "analyse", "compare", "search", "find"],
        TaskCategory.UTILITY: ["convert", "format", "extract", "merge", "split"],
    }
    ROLE_KEYWORDS = {
        "coder": ["code", "build", "create", "implement", "function"],
        "planner": ["plan", "design", "architecture", "strategy", "approach"],
    }

    def classify(self, goal: str) -> TaskProfile:
        text = goal.lower()
        return TaskProfile(
            category=self._detect_category(text),
            complexity=self._compute_complexity(text),
            risk=self._detect_risk(text),
            required_role=self._detect_role(text),
            recommended_strategy=self._infer_strategy(text),
            estimated_tokens=int(len(text.split()) * 1.5) + 100,
            keywords=[kw for kw in self.COMPLEXITY_KEYWORDS if kw in text]
        )

    def _compute_complexity(self, text: str) -> float:
        score = sum(w for kw, w in self.COMPLEXITY_KEYWORDS.items() if kw in text)
        return round(min(score + (len(text) / 2000), 1.0), 2)

    def _detect_category(self, text: str) -> TaskCategory:
        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return cat
        return TaskCategory.CONVERSATION

    def _detect_risk(self, text: str) -> RiskLevel:
        for kw, risk in self.RISK_KEYWORDS.items():
            if kw in text:
                return risk
        return RiskLevel.LOW

    def _detect_role(self, text: str) -> str:
        if any(kw in text for kw in self.ROLE_KEYWORDS["coder"]):
            return "coder"
        if any(kw in text for kw in self.ROLE_KEYWORDS["planner"]):
            return "planner"
        return "answer"

    def _infer_strategy(self, text: str) -> RecommendedStrategy:
        comp = self._compute_complexity(text)
        risk = self._detect_risk(text)
        if risk == RiskLevel.HIGH or comp > 0.7:
            return RecommendedStrategy.GENERATE_VERIFY
        if comp > 0.4:
            return RecommendedStrategy.ADAPT
        return RecommendedStrategy.REUSE
