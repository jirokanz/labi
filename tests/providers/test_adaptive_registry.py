import os
import tempfile

import pytest

from labi.providers.adaptive_registry import AdaptiveProviderRegistry, BaseProvider
from labi.providers.stats import ProviderStatsStore


@pytest.fixture
def stats_store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = ProviderStatsStore(path=path)
    yield store
    store.conn.close()
    os.remove(path)


def _seed(stats_store, provider_name, capability, calls, successes, latency_ms_each, cost_each):
    for _ in range(calls):
        stats_store.record_provider_call(
            provider_name, capability,
            success=True if _ < successes else False,
            latency_ms=latency_ms_each,
            cost_usd=cost_each,
        )


def test_free_and_paid_providers_tie_without_cost_data(stats_store):
    """With no recorded stats (cold start), cost never enters the score --
    ranking falls back to static priority alone, same as before this change."""
    registry = AdaptiveProviderRegistry()
    cheap = BaseProvider("cheap", "m", "", "", ["coding"], priority=10)
    pricey = BaseProvider("pricey", "m", "", "", ["coding"], priority=10)
    registry.register(cheap)
    registry.register(pricey)

    best = registry.get_best("coding", stats_store=stats_store)
    # Tied static priority + no stats -- order falls back to insertion order
    # (stable sort), not an assertion this test needs to over-specify.
    assert best is not None

    assert registry._score(cheap, "coding", stats_store) == registry._score(pricey, "coding", stats_store)


def test_cheaper_provider_outranks_pricier_one_at_equal_quality(stats_store):
    """Once there's enough data (>= MIN_SAMPLES), a provider that's
    otherwise equally successful/fast but cheaper per call should score
    higher -- this is the actual cost-aware routing behavior."""
    registry = AdaptiveProviderRegistry()
    cheap = BaseProvider("cheap", "m", "", "", ["coding"], priority=10)
    pricey = BaseProvider("pricey", "m", "", "", ["coding"], priority=10)
    registry.register(cheap)
    registry.register(pricey)

    # Same success rate and latency, different cost per call.
    _seed(stats_store, "cheap", "coding", calls=10, successes=10, latency_ms_each=500, cost_each=0.0)
    _seed(stats_store, "pricey", "coding", calls=10, successes=10, latency_ms_each=500, cost_each=0.01)

    best = registry.get_best("coding", stats_store=stats_store)
    assert best.name == "cheap"

    cheap_score = registry._score(cheap, "coding", stats_store)
    pricey_score = registry._score(pricey, "coding", stats_store)
    assert cheap_score > pricey_score
    # $0.01/call * COST_WEIGHTS["coding"] == the gap expected when
    # everything else about the two providers is equal.
    expected_gap = 0.01 * AdaptiveProviderRegistry.COST_WEIGHTS["coding"]
    assert cheap_score - pricey_score == pytest.approx(expected_gap)


def test_cost_weight_is_lower_for_coding_than_answering():
    """Coding is quality-sensitive (a wrong answer triggers autofix retries),
    so it should weigh cost less heavily than a capability like answering
    where a merely-good-enough free response is acceptable."""
    assert AdaptiveProviderRegistry.COST_WEIGHTS["coding"] < AdaptiveProviderRegistry.COST_WEIGHTS["answering"]


def test_unlisted_capability_falls_back_to_default_weight(stats_store):
    """A capability with no explicit entry in COST_WEIGHTS (e.g. a future
    web_search) still gets a cost penalty, via DEFAULT_COST_WEIGHT, rather
    than silently skipping cost-awareness."""
    registry = AdaptiveProviderRegistry()
    cheap = BaseProvider("cheap", "m", "", "", ["web_search"], priority=10)
    pricey = BaseProvider("pricey", "m", "", "", ["web_search"], priority=10)
    registry.register(cheap)
    registry.register(pricey)

    _seed(stats_store, "cheap", "web_search", calls=10, successes=10, latency_ms_each=500, cost_each=0.0)
    _seed(stats_store, "pricey", "web_search", calls=10, successes=10, latency_ms_each=500, cost_each=0.01)

    cheap_score = registry._score(cheap, "web_search", stats_store)
    pricey_score = registry._score(pricey, "web_search", stats_store)
    expected_gap = 0.01 * AdaptiveProviderRegistry.DEFAULT_COST_WEIGHT
    assert cheap_score - pricey_score == pytest.approx(expected_gap)


def test_high_cost_can_be_outweighed_by_higher_success_rate(stats_store):
    """Cost is a tie-breaking penalty, not an override -- a meaningfully
    more successful paid provider can still beat a cheap flaky one."""
    registry = AdaptiveProviderRegistry()
    flaky_free = BaseProvider("flaky_free", "m", "", "", ["coding"], priority=10)
    reliable_paid = BaseProvider("reliable_paid", "m", "", "", ["coding"], priority=10)
    registry.register(flaky_free)
    registry.register(reliable_paid)

    _seed(stats_store, "flaky_free", "coding", calls=10, successes=3, latency_ms_each=500, cost_each=0.0)
    _seed(stats_store, "reliable_paid", "coding", calls=10, successes=10, latency_ms_each=500, cost_each=0.01)

    best = registry.get_best("coding", stats_store=stats_store)
    assert best.name == "reliable_paid"
