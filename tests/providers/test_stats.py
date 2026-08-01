from labi.providers.stats import ProviderStatsStore


def test_record_and_get_recent_failures(tmp_path):
    store = ProviderStatsStore(str(tmp_path / "stats.db"))
    store.record_failure("groq", "coding", "timeout", "Request timed out after 30s")
    store.record_failure("groq", "coding", "quota_exceeded", "429 Too Many Requests")

    recent = store.get_recent_failures()
    assert len(recent) == 2
    # Most recent first.
    assert recent[0]["reason"] == "quota_exceeded"
    assert recent[1]["reason"] == "timeout"


def test_get_recent_failures_filters_by_provider_and_capability(tmp_path):
    store = ProviderStatsStore(str(tmp_path / "stats.db"))
    store.record_failure("groq", "coding", "timeout", "msg1")
    store.record_failure("gemini", "answering", "not_found", "msg2")

    groq_only = store.get_recent_failures(provider="groq")
    assert len(groq_only) == 1
    assert groq_only[0]["provider"] == "groq"

    answering_only = store.get_recent_failures(capability="answering")
    assert len(answering_only) == 1
    assert answering_only[0]["capability"] == "answering"


def test_get_recent_failures_respects_limit(tmp_path):
    store = ProviderStatsStore(str(tmp_path / "stats.db"))
    for i in range(10):
        store.record_failure("groq", "coding", "timeout", f"attempt {i}")
    assert len(store.get_recent_failures(limit=3)) == 3


def test_failure_message_gets_truncated(tmp_path):
    store = ProviderStatsStore(str(tmp_path / "stats.db"))
    long_message = "x" * 1000
    store.record_failure("groq", "coding", "other", long_message)
    recorded = store.get_recent_failures()[0]["message"]
    assert len(recorded) <= 500


def test_get_failure_summary_counts_by_provider_and_reason(tmp_path):
    store = ProviderStatsStore(str(tmp_path / "stats.db"))
    store.record_failure("groq", "coding", "timeout", "msg")
    store.record_failure("groq", "coding", "timeout", "msg")
    store.record_failure("groq", "coding", "quota_exceeded", "msg")
    store.record_failure("gemini", "answering", "not_found", "msg")

    summary = store.get_failure_summary()
    # Sorted by count descending -- groq/timeout (2) should lead.
    assert summary[0]["provider"] == "groq"
    assert summary[0]["reason"] == "timeout"
    assert summary[0]["count"] == 2

    by_key = {(r["provider"], r["reason"]): r["count"] for r in summary}
    assert by_key[("groq", "quota_exceeded")] == 1
    assert by_key[("gemini", "not_found")] == 1


def test_get_failure_summary_empty_when_no_failures(tmp_path):
    store = ProviderStatsStore(str(tmp_path / "stats.db"))
    assert store.get_failure_summary() == []
