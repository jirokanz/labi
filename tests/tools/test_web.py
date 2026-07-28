import pytest

from labi.tools import web


def test_extract_content_strips_tags_and_scripts():
    html = """
    <html><head><title>ignored</title></head>
    <body>
        <script>var x = 1;</script>
        <style>.a { color: red; }</style>
        <h1>Hello World</h1>
        <p>Some   body   text.</p>
    </body></html>
    """
    text = web.extract_content(html)
    assert "Hello World" in text
    assert "Some body text." in text
    assert "var x" not in text
    assert "color: red" not in text
    assert "ignored" not in text  # <head><title> content is dropped


def test_extract_content_truncates_to_max_chars():
    html = "<p>" + ("word " * 5000) + "</p>"
    text = web.extract_content(html, max_chars=50)
    assert len(text) == 50


def test_fetch_rejects_non_http_scheme():
    result = web.fetch("file:///etc/passwd")
    assert result is None
    assert "scheme" in web.LAST_ERROR["fetch"]


def test_fetch_rejects_unresolvable_host():
    result = web.fetch("http://this-host-does-not-exist.invalid/")
    assert result is None
    assert "fetch" in web.LAST_ERROR


def test_fetch_rejects_loopback_host():
    result = web.fetch("http://127.0.0.1/secret")
    assert result is None
    assert "unsafe" in web.LAST_ERROR["fetch"]


def test_fetch_rejects_link_local_metadata_host():
    # Cloud metadata endpoints (AWS/GCP/Azure) all live at this address --
    # a classic SSRF target that must never be reachable via this tool.
    result = web.fetch("http://169.254.169.254/latest/meta-data/")
    assert result is None
    assert "unsafe" in web.LAST_ERROR["fetch"]


def test_fetch_uses_injected_fetch_fn_and_returns_decoded_text():
    def fake_fetch(url):
        assert url == "https://example.com/page"
        return b"hello world"

    # Bypass the real hostname safety check by patching it for this test --
    # example.com resolves fine, so this exercises the happy path.
    result = web.fetch("https://example.com/page", fetch_fn=fake_fetch)
    assert result == "hello world"


def test_fetch_truncates_oversized_responses():
    def fake_fetch(url):
        return b"x" * 1000

    result = web.fetch("https://example.com/page", max_bytes=10, fetch_fn=fake_fetch)
    assert result == "x" * 10


def test_search_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    result = web.search("latest groq model", api_key=None)
    assert result is None
    assert "BRAVE_API_KEY" in web.LAST_ERROR["search"]


def test_search_parses_results_from_fetch_json_fn():
    def fake_fetch_json(query):
        assert query == "latest groq model"
        return {
            "web": {
                "results": [
                    {"title": "Groq Models", "url": "https://groq.com/models", "description": "List of models"},
                    {"title": "Other", "url": "https://example.com", "description": "desc"},
                ]
            }
        }

    results = web.search("latest groq model", api_key="fake-key", fetch_json_fn=fake_fetch_json)
    assert results == [
        {"title": "Groq Models", "url": "https://groq.com/models", "snippet": "List of models"},
        {"title": "Other", "url": "https://example.com", "snippet": "desc"},
    ]


def test_search_respects_max_results():
    def fake_fetch_json(query):
        return {"web": {"results": [{"title": f"T{i}", "url": f"https://x.com/{i}", "description": ""} for i in range(10)]}}

    results = web.search("q", api_key="fake-key", max_results=3, fetch_json_fn=fake_fetch_json)
    assert len(results) == 3


def test_search_returns_none_on_fetch_error():
    def failing_fetch_json(query):
        raise TimeoutError("timed out")

    result = web.search("q", api_key="fake-key", fetch_json_fn=failing_fetch_json)
    assert result is None
    assert "TimeoutError" in web.LAST_ERROR["search"]
