from labi.providers.generation import _c, format_code_block, _highlight_line


def test_format_code_block_does_not_raise():
    """Regression test: an earlier extraction of stream_generate/_c into
    this module left _highlight_line (and therefore format_code_block)
    referencing _USE_COLOR from the old module-level scope, which no
    longer existed there -- a NameError on every call. Untested CLI-
    display code, so it slipped through. format_code_block is the only
    public entry point that exercises _highlight_line, so calling it
    here is what would have caught this."""
    result = format_code_block("def f():\n    return 1", title="test")
    assert "def f()" in result
    assert "return 1" in result


def test_highlight_line_does_not_raise_on_various_lines():
    for line in ["", "# a comment", "x = 'a string'", "def f(a, b):", "    return a + b"]:
        _highlight_line(line)  # should not raise


def test_c_returns_plain_text_when_color_disabled(monkeypatch):
    import labi.providers.generation as gen
    monkeypatch.setattr(gen, "_USE_COLOR", False)
    assert _c("hello", "red") == "hello"


# ---- Model fallback chains with failure reasons ----

def test_classify_failure_reason_timeout():
    from labi.providers.generation import classify_failure_reason
    assert classify_failure_reason("Request timed out after 30s") == "timeout"


def test_classify_failure_reason_quota():
    from labi.providers.generation import classify_failure_reason
    assert classify_failure_reason("litellm.RateLimitError: 429 Too Many Requests") == "quota_exceeded"


def test_classify_failure_reason_auth():
    from labi.providers.generation import classify_failure_reason
    assert classify_failure_reason("AuthenticationError: 401 invalid api key") == "auth_error"


def test_classify_failure_reason_not_found():
    # The exact shape of the real Gemini failure hit live this session.
    from labi.providers.generation import classify_failure_reason
    assert classify_failure_reason("litellm.NotFoundError: GeminiException - 404") == "not_found"


def test_classify_failure_reason_api_error():
    from labi.providers.generation import classify_failure_reason
    assert classify_failure_reason("APIError: 503 Service Unavailable") == "api_error"


def test_classify_failure_reason_falls_back_to_other():
    from labi.providers.generation import classify_failure_reason
    assert classify_failure_reason("something totally unrecognized happened") == "other"


def test_classify_failure_reason_handles_empty_message():
    from labi.providers.generation import classify_failure_reason
    assert classify_failure_reason("") == "other"
    assert classify_failure_reason(None) == "other"


def test_stream_generate_raises_provider_call_error_with_classified_reason(tmp_path):
    from labi.providers.generation import stream_generate, ProviderCallError
    from labi.providers.stats import ProviderStatsStore
    from tests.fakes import FakeProvider

    stats = ProviderStatsStore(str(tmp_path / "stats.db"))
    failing = FakeProvider("flaky", ["answering"], "unused", should_fail=True)

    try:
        stream_generate(failing, "hello", stats_store=stats, capability="answering")
        assert False, "expected ProviderCallError"
    except ProviderCallError as e:
        assert e.provider_name == "flaky"
        assert e.reason == "other"  # FakeProvider raises a generic RuntimeError

    # The failure should have been recorded, not just raised.
    recent = stats.get_recent_failures(provider="flaky")
    assert len(recent) == 1
    assert recent[0]["reason"] == "other"
    assert "flaky is down" in recent[0]["message"]
