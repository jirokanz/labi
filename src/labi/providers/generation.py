"""
Shared provider-call machinery: stream_generate() plus the tiny color
helper it uses for CLI output.

Pulled out of agent.py so it can be reused by src/labi/agents/*.py (the
Planner/Executor/Validator agents) without agent.py and agents/* importing
each other -- this module depends on nothing from either, so both can
depend on it. This is the same call path the original CLI loop in
agent.py already used (stats recording, cost tracking, fallback to a
non-streaming call on error); the agents package reuses it rather than
re-implementing provider calls, so there's exactly one place that knows
how to talk to a provider and record what happened.
"""

import re
import sys
import time

import litellm

_ANSI = {
    "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
    "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m",
    "magenta": "\033[35m", "red": "\033[31m", "grey": "\033[90m",
}
_USE_COLOR = sys.stdout.isatty()


def estimate_tokens(text):
    """Rough chars/4 estimate -- deliberately not calling litellm.token_counter
    here (that's a heavier, model-specific call). This only needs to be
    right enough to skip a provider whose context window is clearly too
    small, not exact."""
    return len(text) // 4


def _c(text, color):
    if not _USE_COLOR:
        return text
    return f"{_ANSI[color]}{text}{_ANSI['reset']}"


_PY_KEYWORDS = {
    "def", "class", "return", "if", "elif", "else", "for", "while", "in",
    "import", "from", "as", "with", "try", "except", "finally", "raise",
    "pass", "break", "continue", "yield", "lambda", "None", "True", "False",
    "and", "or", "not", "is", "global", "nonlocal", "assert", "async", "await",
}

_STRING_RE = re.compile(r"""('[^'\\]*(?:\\.[^'\\]*)*'|"[^"\\]*(?:\\.[^"\\]*)*")""")


def _highlight_line(line):
    if not _USE_COLOR:
        return line
    stripped = line.strip()
    if stripped.startswith("#"):
        return _c(line, "grey")

    # Pull out string literals first so the word-splitter below doesn't
    # tear them apart before a color can be applied.
    parts = _STRING_RE.split(line)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # captured string literal
            out.append(_c(part, "green"))
            continue
        for tok in re.split(r"(\W+)", part):
            out.append(_c(tok, "magenta") if tok in _PY_KEYWORDS else tok)
    return "".join(out)


def format_code_block(code, title="code"):
    """Boxed, line-numbered, lightly syntax-highlighted code display --
    replaces a bare `print(code)` wall of text."""
    lines = code.splitlines() or [""]
    width = max((len(l) for l in lines), default=0)
    width = min(max(width, len(title)), 100)
    bar = "─" * (width + 6)
    out = [f"\n{_c('┌' + bar + '┐', 'dim')}", f"{_c('│', 'dim')} {_c(title, 'bold')}"]
    out.append(_c("├" + bar + "┤", "dim"))
    gutter_width = len(str(len(lines)))
    for i, line in enumerate(lines, 1):
        num = str(i).rjust(gutter_width)
        out.append(f"{_c(num, 'grey')} {_c('│', 'dim')} {_highlight_line(line)}")
    out.append(_c("└" + bar + "┘", "dim"))
    return "\n".join(out)


def stream_generate(provider, prompt, system_prompt=None, max_tokens=1024, history=None,
                     label="Generating", render="text", stats_store=None, capability=None,
                     cost_tracker=None):
    """Streams from the provider as it's produced (vibe-coding feel).
    render='text'  -> print raw chunks live (good for prose: plans, answers)
    render='code'  -> print a lightweight progress indicator instead of raw
                       chunks, and let the caller show a formatted code box
                       once generation is complete (avoids dumping raw,
                       unhighlighted, possibly-mid-fence text to the screen
                       and then immediately re-printing the same code).
    If stats_store (a ProviderStatsStore) + capability are given, records
    success/latency/cost so AdaptiveProviderRegistry can rank providers by
    measured performance instead of only the static hardcoded priority.
    If cost_tracker (a CostTracker) is given, the estimated cost of this
    call is added to its running total for the current task."""
    print(f"{_c(f'   [{provider.name}] {label}...', 'cyan')}")
    start = time.monotonic()
    success = False
    output_text = ""
    try:
        chunks = []
        dots = 0
        for chunk in provider.generate_stream(prompt, system_prompt, max_tokens, history):
            chunks.append(chunk)
            if render == "text":
                print(chunk, end="", flush=True)
            else:
                dots += 1
                if dots % 20 == 0:
                    print(".", end="", flush=True)
        if render == "text":
            print("\n")
        else:
            print(" done\n")
        success = True
        output_text = "".join(chunks)
        return output_text
    except Exception:
        try:
            result = provider.generate(prompt, system_prompt, max_tokens, history)
            if render == "text":
                print(result["content"] + "\n")
            success = True
            output_text = result["content"]
            return output_text
        except Exception:
            raise
    finally:
        latency_ms = (time.monotonic() - start) * 1000
        cost = 0.0
        total_tokens = 0
        if success:
            full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt
            try:
                # completion_cost estimates tokens from the raw text when no
                # usage object is available (true for our manual streaming
                # accumulation). Models litellm has no pricing data for
                # raise/return 0 here -- that means "unknown", not "free".
                cost = litellm.completion_cost(model=provider.model, prompt=full_prompt, completion=output_text) or 0.0
            except Exception:
                cost = 0.0
            try:
                total_tokens = (litellm.token_counter(model=provider.model, text=full_prompt)
                                 + litellm.token_counter(model=provider.model, text=output_text))
            except Exception:
                total_tokens = 0
        if cost_tracker is not None:
            cost_tracker.add(cost)
        if stats_store is not None and capability is not None:
            try:
                stats_store.record_provider_call(provider.name, capability, success, latency_ms, cost_usd=cost)
                stats_store.record_daily_usage(provider.name, tokens=total_tokens, requests=1)
            except Exception:
                pass  # stats tracking should never break the actual task
