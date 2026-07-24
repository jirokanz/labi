"""ProviderRegistry - tracks provider availability and routing scores.

Referenced by intelligence/router.py but never existed in this repo --
that was the actual ImportError blocking the whole test suite from even
collecting (not the MemoryDatabase/MemoryDB rename, though that was a
real, separate bug too).

Kept intentionally simple (in-memory, adaptive score with no persistence
or real provider calls) since execution/executors.py is still mock-only
for this phase per its own docstring ("Mock executors for Phase 2.4.7
integration testing") -- this exists to make the decision layer
(classifier -> router -> engine) testable end-to-end without depending on
real provider integration yet. update_score() is the hook for wiring in
real outcomes once a non-mock generation executor exists.
"""

from dataclasses import dataclass


@dataclass
class ProviderStatus:
    name: str
    available: bool = True
    routing_score: float = 0.5  # 0-1, higher = more trusted/preferred


class ProviderRegistry:
    def __init__(self):
        self._providers = {}  # provider_key ("name:role") -> ProviderStatus

    def register(self, name, role, available=True, routing_score=0.5):
        key = f"{name}:{role}"
        self._providers[key] = ProviderStatus(name=name, available=available, routing_score=routing_score)
        return key

    def get_provider_status(self, provider_key):
        if provider_key in self._providers:
            return self._providers[provider_key]
        # Unknown key (e.g. a memory record referencing a provider that's
        # since been removed/renamed) -- degrade to a neutral default
        # rather than raising, since the router treats unavailable as a
        # score of 0 and this shouldn't hard-fail a memory lookup.
        return ProviderStatus(name=provider_key, available=True, routing_score=0.5)

    def select_best_provider(self, role, complexity=0.5):
        candidates = [
            (key, status) for key, status in self._providers.items()
            if key.endswith(f":{role}") and status.available
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda kv: kv[1].routing_score, reverse=True)
        return candidates[0][0]

    def update_score(self, provider_key, success, decay=0.9):
        """Exponential moving average update -- call this with the real
        outcome of a provider call once execution is no longer mocked."""
        status = self._providers.get(provider_key)
        if not status:
            return
        target = 1.0 if success else 0.0
        status.routing_score = round((status.routing_score * decay) + (target * (1 - decay)), 4)
