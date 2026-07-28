"""TaskClassifier - Heuristic-based, zero-LLM cost."""

import re

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

    # Signal family 1: explicit freshness keywords -- a goal that names
    # its own time-sensitivity ("latest", "current", "today"...).
    LIVE_INFO_KEYWORDS = [
        "latest", "current", "currently", "today", "right now", "as of",
        "recent", "recently", "this week", "this month", "this year",
        "news", "update", "updated", "release", "released", "price", "stock",
        "score",
    ]

    # Signal family 2: "who is the <role>" role-holder questions. These
    # have NO freshness keyword at all ("who is the CEO of OpenAI?") but
    # are exactly as time-sensitive as "who is the current CEO" -- asking
    # about a role implicitly means "whoever holds it now", which any
    # model's frozen training data can only answer as of its cutoff. This
    # is the gap a plain keyword list misses.
    CURRENT_ROLE_TITLES = [
        "ceo", "president", "prime minister", "chairman", "chairwoman",
        "chair", "director", "chancellor", "governor", "mayor", "pope",
        "monarch", "king", "queen", "head coach", "commissioner",
        "secretary general", "secretary of state",
    ]
    CURRENT_ROLE_PATTERN = re.compile(
        r"\bwho\s+(?:is|are)\s+(?:the\s+)?(?:current\s+|new\s+)?(" +
        "|".join(re.escape(t) for t in CURRENT_ROLE_TITLES) + r")\b"
    )
    # Explicit historical framing overrides the role pattern -- "who was
    # the first president" or "who is the historical founder" is asking
    # about a fixed, settled fact, not the current officeholder.
    HISTORICAL_MARKERS = [
        "first", "original", "founder", "founding", "died", "history of",
        "used to be", "former", "was the", "back in", "historically",
    ]

    def classify(self, goal: str) -> TaskProfile:
        text = goal.lower()
        requires_web, web_confidence = self._detect_requires_web(text)
        return TaskProfile(
            category=self._detect_category(text),
            complexity=self._compute_complexity(text),
            risk=self._detect_risk(text),
            required_role=self._detect_role(text),
            recommended_strategy=self._infer_strategy(text),
            estimated_tokens=int(len(text.split()) * 1.5) + 100,
            keywords=[kw for kw in self.COMPLEXITY_KEYWORDS if kw in text],
            requires_web=requires_web,
            web_confidence=web_confidence,
        )

    def _detect_requires_web(self, text: str):
        """Returns (requires_web, confidence). The keyword match is the
        stronger signal (the goal names its own time-sensitivity
        directly); the role-holder pattern is a bit weaker since it's
        inferred rather than stated, and is suppressed entirely by an
        explicit historical marker."""
        if any(kw in text for kw in self.LIVE_INFO_KEYWORDS):
            return True, 0.9
        if self.CURRENT_ROLE_PATTERN.search(text) and not any(m in text for m in self.HISTORICAL_MARKERS):
            return True, 0.75
        return False, 0.0

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
