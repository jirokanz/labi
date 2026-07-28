import os
import tempfile

import pytest

from labi.tools.sources import SourceStore


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = SourceStore(path=path)
    yield s
    s.conn.close()
    os.remove(path)


def test_record_and_get_sources(store):
    store.record("task_1", "https://example.com/a", "Example A", "some content")
    store.record("task_1", "https://example.com/b", "Example B", "other content")

    sources = store.get_sources("task_1")
    urls = {s["url"] for s in sources}
    assert urls == {"https://example.com/a", "https://example.com/b"}
    assert all(s["content_hash"] for s in sources)


def test_get_sources_empty_for_unknown_task(store):
    assert store.get_sources("nonexistent") == []


def test_unchanged_since_true_for_identical_recent_content(store):
    store.record("task_1", "https://example.com/a", "Title", "same content")
    assert store.unchanged_since("https://example.com/a", "same content", within_hours=24) is True


def test_unchanged_since_false_when_content_differs(store):
    store.record("task_1", "https://example.com/a", "Title", "old content")
    assert store.unchanged_since("https://example.com/a", "new content", within_hours=24) is False


def test_unchanged_since_false_for_unknown_url(store):
    assert store.unchanged_since("https://never-seen.com", "content") is False


def test_record_upserts_on_same_task_and_url(store):
    store.record("task_1", "https://example.com/a", "Old Title", "old content")
    store.record("task_1", "https://example.com/a", "New Title", "new content")

    sources = store.get_sources("task_1")
    assert len(sources) == 1
    assert sources[0]["title"] == "New Title"
