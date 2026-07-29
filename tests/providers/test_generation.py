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
