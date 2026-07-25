"""MemoryRouter - Evaluates trust using MemoryDB + ProviderRegistry.""" 

from datetime import datetime
from typing import List, Optional

from labi.intelligence.types import TaskProfile, ReuseDecision, RecommendedStrategy
from labi.intelligence.trace import DecisionTrace
from labi.memory.database import MemoryDB
from labi.memory.models import MemoryCandidate
from labi.providers.registry import ProviderRegistry

class MemoryRouter:
    def __init__(self, memory_db: MemoryDB, provider_registry: ProviderRegistry):
        self.memory_db = memory_db
        self.registry = provider_registry

    def decide(self, goal: str, profile: TaskProfile, limit: int = 5) -> ReuseDecision:
        candidates = self.memory_db.find_similar(goal, limit=limit)
        if not candidates:
            return ReuseDecision(
                action=RecommendedStrategy.GENERATE_VERIFY,
                confidence=0.0,
                reasoning="No compatible memory found.",
                provider_hint=self._best_provider(profile),
                trace=DecisionTrace(stage="memory_router", notes=["Database returned 0 candidates"])
            )

        scored = []
        for cand in candidates:
            score, reason, trace = self._score_memory(cand, profile)
            scored.append((score, reason, trace, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top_reason, top_trace, top_cand = scored[0]

        if top_score > 0.75:
            action = RecommendedStrategy.REUSE
        elif top_score > 0.45:
            action = RecommendedStrategy.ADAPT
        else:
            action = RecommendedStrategy.GENERATE_VERIFY

        if action in (RecommendedStrategy.REUSE, RecommendedStrategy.ADAPT):
            self.memory_db.mark_used(top_cand.record.id)

        top_trace.add_signal("final_score", top_score)
        top_trace.add_note(f"Action: {action.value}")
        top_trace.add_meta("similarity_method", top_cand.similarity_method)

        return ReuseDecision(
            action=action,
            confidence=top_score,
            reasoning=top_reason,
            similar_memory_id=top_cand.record.id if top_score > 0.3 else None,
            provider_hint=self._best_provider(profile),
            trace=top_trace
        )

    def _score_memory(self, cand: MemoryCandidate, profile: TaskProfile):
        mem = cand.record
        trace = DecisionTrace(stage="memory_router")
        notes = []

        similarity = cand.similarity
        trace.add_signal("similarity", similarity)
        notes.append(f"Sim={similarity:.2f}")

        prev_success = 1.0 if mem.success else 0.0
        trace.add_signal("prev_success", prev_success)

        status = self.registry.get_provider_status(mem.provider_key)
        provider_score = status.routing_score if status.available else 0.0
        trace.add_signal("provider_health", provider_score)
        notes.append(f"Provider={provider_score:.2f}")

        age_days = (datetime.utcnow() - mem.created_at).total_seconds() / 86400
        age_factor = max(0.0, 1.0 - (age_days / 30))
        if mem.last_used_at and (datetime.utcnow() - mem.last_used_at).total_seconds() / 86400 < 7:
            age_factor = min(1.0, age_factor + 0.1)
        trace.add_signal("age_factor", age_factor)
        notes.append(f"Age={age_factor:.2f}")

        complexity_match = 1.0 - abs(profile.complexity - mem.task_complexity)
        trace.add_signal("complexity_match", complexity_match)

        total = (similarity * 0.35) + (prev_success * 0.25) + (provider_score * 0.20) + (age_factor * 0.10) + (complexity_match * 0.10)
        trace.add_signal("weighted_score", total)
        for note in notes:
            trace.add_note(note)
        return round(total, 3), " | ".join(notes), trace

    def _best_provider(self, profile: TaskProfile) -> Optional[str]:
        return self.registry.select_best_provider(role=profile.required_role, complexity=profile.complexity)
