"""
Integration tests for the full web pipeline:

    classifier (requires_web) -> web.search() -> AdaptiveProviderRegistry
        -> summarizer provider -> MemoryDB + SourceStore -> citation output

Unit tests elsewhere already prove each piece works in isolation
(tests/tools/test_web.py, tests/tools/test_sources.py,
tests/intelligence/test_classifier.py, tests/providers/test_adaptive_registry.py).
What those can't catch is a wiring regression between the pieces -- e.g.
try_web_search_answer() calling the wrong capability name, or silently
swallowing a real result. These tests exercise agent.needs_live_info()
and agent.try_web_search_answer() together, with a fake LLM provider
(no real network/litellm calls) standing in for the summarizer.
"""

import os
import tempfile

import pytest

from labi.agent import (
    MemoryDB,
    WorkspaceManager,
    SessionContext,
    needs_live_info,
    try_web_search_answer,
)
from labi.providers.adaptive_registry import AdaptiveProviderRegistry
from labi.providers.stats import ProviderStatsStore
from labi.tools import web as web_search
from labi.tools.sources import SourceStore


class FakeProvider:
    """Duck-types BaseProvider's interface without touching litellm/the
    network -- generate_stream()/generate() just return a canned
    response, optionally after raising once to exercise the fallback
    chain (see AdaptiveProviderRegistry.get_all in agent.py's
    try_web_search_answer)."""

    def __init__(self, name, capabilities, response_text, priority=10, should_fail=False):
        self.name = name
        self.capabilities = capabilities
        self.priority = priority
        self.capability_priority = {}
        self.context_window = None
        self.model = "gpt-3.5-turbo"  # a model litellm has offline cost data for
        self._response_text = response_text
        self._should_fail = should_fail

    def priority_for(self, capability):
        return self.capability_priority.get(capability, self.priority)

    def generate_stream(self, prompt, system_prompt=None, max_tokens=768, history=None):
        if self._should_fail:
            raise RuntimeError(f"{self.name} is down")
        yield self._response_text

    def generate(self, prompt, system_prompt=None, max_tokens=768, history=None):
        if self._should_fail:
            raise RuntimeError(f"{self.name} is down")
        return {
            "content": self._response_text, "model": self.model,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "provider": self.name,
        }


@pytest.fixture
def temp_paths(tmp_path):
    return {
        "memory_db": str(tmp_path / "memory.db"),
        "stats_db": str(tmp_path / "stats.db"),
        "sources_db": str(tmp_path / "sources.db"),
        "workspace": str(tmp_path / "workspace"),
    }


@pytest.fixture
def pipeline(temp_paths):
    return {
        "registry": AdaptiveProviderRegistry(),
        "stats_store": ProviderStatsStore(temp_paths["stats_db"]),
        "source_store": SourceStore(temp_paths["sources_db"]),
        "memory_db": MemoryDB(temp_paths["memory_db"]),
        "workspace": WorkspaceManager(temp_paths["workspace"]),
        "session": SessionContext(),
    }


def _fake_search_results():
    return [
        {"title": "OpenAI Leadership", "url": "https://openai.com/about", "snippet": "Sam Altman is CEO of OpenAI."},
        {"title": "OpenAI Newsroom", "url": "https://openai.com/news", "snippet": "Leadership updates."},
    ]


def test_full_pipeline_classifier_to_citation(monkeypatch, pipeline):
    """The end-to-end happy path the doc's example asks for: a role-holder
    question with no freshness keyword is classified as needing the web,
    a search runs, a provider summarizes with citations, and the result
    lands in both MemoryDB and SourceStore."""
    goal = "Who is the current CEO of OpenAI?"
    assert needs_live_info(goal) is True

    monkeypatch.setattr(web_search, "search", lambda q, **kw: _fake_search_results())

    pipeline["registry"].register(FakeProvider(
        "fake-summarizer", ["web_search"],
        "Sam Altman is the CEO of OpenAI [1]. See also [2] for leadership updates.",
    ))

    task_id = "task_integration_1"
    handled = try_web_search_answer(
        goal, task_id,
        pipeline["registry"], pipeline["stats_store"], pipeline["source_store"],
        pipeline["memory_db"], pipeline["workspace"], pipeline["session"],
    )

    assert handled is True

    # Citation made it into the stored answer.
    stored = pipeline["memory_db"].conn.execute(
        "SELECT answer, provider FROM tasks WHERE id=?", (task_id,)
    ).fetchone()
    assert stored is not None
    answer, provider_name = stored
    assert "[1]" in answer
    assert provider_name == "fake-summarizer"

    # Sources were persisted and are retrievable by task_id.
    sources = pipeline["source_store"].get_sources(task_id)
    assert len(sources) == 2
    assert {s["url"] for s in sources} == {"https://openai.com/about", "https://openai.com/news"}

    # Session history was updated so a follow-up question has context.
    assert pipeline["session"].turns
    assert pipeline["session"].turns[-1]["goal"] == goal


def test_pipeline_falls_back_to_next_provider_on_failure(monkeypatch, pipeline):
    """The fallback-chain behavior (get_all + try/continue) must survive
    the web_search wiring specifically, not just the plain answering path
    it was originally built for."""
    monkeypatch.setattr(web_search, "search", lambda q, **kw: _fake_search_results())

    pipeline["registry"].register(FakeProvider("flaky", ["web_search"], "", priority=5, should_fail=True))
    pipeline["registry"].register(FakeProvider("reliable", ["web_search"], "Sam Altman is CEO [1].", priority=20))

    handled = try_web_search_answer(
        "Who is the current CEO of OpenAI?", "task_integration_2",
        pipeline["registry"], pipeline["stats_store"], pipeline["source_store"],
        pipeline["memory_db"], pipeline["workspace"], pipeline["session"],
    )

    assert handled is True
    stored = pipeline["memory_db"].conn.execute(
        "SELECT provider FROM tasks WHERE id=?", ("task_integration_2",)
    ).fetchone()
    assert stored[0] == "reliable"


def test_pipeline_returns_false_when_search_unavailable(monkeypatch, pipeline):
    """No API key / search failure -- caller (agent.py's main loop) should
    fall through to the normal answering pipeline, not crash or silently
    do nothing."""
    monkeypatch.setattr(web_search, "search", lambda q, **kw: None)
    pipeline["registry"].register(FakeProvider("fake-summarizer", ["web_search"], "answer"))

    handled = try_web_search_answer(
        "Who is the current CEO of OpenAI?", "task_integration_3",
        pipeline["registry"], pipeline["stats_store"], pipeline["source_store"],
        pipeline["memory_db"], pipeline["workspace"], pipeline["session"],
    )
    assert handled is False
    assert pipeline["memory_db"].conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE id=?", ("task_integration_3",)
    ).fetchone()[0] == 0


def test_pipeline_returns_false_when_no_results_found(monkeypatch, pipeline):
    monkeypatch.setattr(web_search, "search", lambda q, **kw: [])
    pipeline["registry"].register(FakeProvider("fake-summarizer", ["web_search"], "answer"))

    handled = try_web_search_answer(
        "Who is the current CEO of OpenAI?", "task_integration_4",
        pipeline["registry"], pipeline["stats_store"], pipeline["source_store"],
        pipeline["memory_db"], pipeline["workspace"], pipeline["session"],
    )
    assert handled is False


def test_pipeline_returns_false_when_no_web_search_provider_registered(monkeypatch, pipeline):
    """A registry with only e.g. a coding provider (no web_search
    capability anywhere) must degrade gracefully rather than error."""
    monkeypatch.setattr(web_search, "search", lambda q, **kw: _fake_search_results())
    pipeline["registry"].register(FakeProvider("coder-only", ["coding"], "irrelevant"))

    handled = try_web_search_answer(
        "Who is the current CEO of OpenAI?", "task_integration_5",
        pipeline["registry"], pipeline["stats_store"], pipeline["source_store"],
        pipeline["memory_db"], pipeline["workspace"], pipeline["session"],
    )
    assert handled is False


def test_pipeline_returns_false_when_all_providers_fail(monkeypatch, pipeline):
    monkeypatch.setattr(web_search, "search", lambda q, **kw: _fake_search_results())
    pipeline["registry"].register(FakeProvider("a", ["web_search"], "", should_fail=True))
    pipeline["registry"].register(FakeProvider("b", ["web_search"], "", should_fail=True))

    handled = try_web_search_answer(
        "Who is the current CEO of OpenAI?", "task_integration_6",
        pipeline["registry"], pipeline["stats_store"], pipeline["source_store"],
        pipeline["memory_db"], pipeline["workspace"], pipeline["session"],
    )
    assert handled is False


def test_timeless_question_never_triggers_web_pipeline(monkeypatch, pipeline):
    """Negative-path integration check: a goal that shouldn't trigger web
    search never touches web.search() at all -- the classifier gate,
    not just the tool itself, is what's under test here."""
    goal = "Explain how a hash table works"
    assert needs_live_info(goal) is False

    called = {"count": 0}

    def _tracking_search(q, **kw):
        called["count"] += 1
        return _fake_search_results()

    monkeypatch.setattr(web_search, "search", _tracking_search)

    # Mirrors agent.py's main loop: only call try_web_search_answer at all
    # when needs_live_info() says to.
    if needs_live_info(goal):
        try_web_search_answer(
            goal, "task_integration_7",
            pipeline["registry"], pipeline["stats_store"], pipeline["source_store"],
            pipeline["memory_db"], pipeline["workspace"], pipeline["session"],
        )

    assert called["count"] == 0
