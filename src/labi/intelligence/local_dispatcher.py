"""
Local-first request dispatcher.

The LLM should be the last resort, not the first. This module intercepts
requests that can be answered deterministically -- with zero API calls --
before they ever reach provider selection.

Deliberately scoped to three categories only: arithmetic, UUID generation,
timestamps. These share a property the rest of the "no API needed" list
(JSON formatting, regex extraction, file hashing) doesn't: they're fully
determined by the goal text alone, with no separate input data required,
and a wrong match just falls through safely to the normal pipeline rather
than silently giving a bad answer with no review step. "Extract emails"
or "hash this file" need actual input attached to be unambiguous -- those
are left to the existing sandboxed code-gen path, which already handles
them correctly.

"Repeat question" is intentionally not handled here either -- that's
already MemoryRouter's job (find_similar + replay), which runs before
generation and offers a review step, unlike this module which returns an
answer with no confirmation at all. Duplicating it here would just create
two different answers to "have I seen this before?".
"""

import ast
import operator
import re
import time
import uuid as _uuid
from datetime import datetime, timezone


# ---- Safe arithmetic evaluation ----
#
# Two independent layers, matching the defense-in-depth pattern already
# used in tools/python/security.py: a character whitelist regex gate
# (so no letters/names/underscores can ever reach the parser -- rules out
# `__import__`-style tricks before ast.parse even runs), and a restricted
# AST walk that only permits numeric literals and basic arithmetic ops.

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MATH_CHARSET_RE = re.compile(r'^[\d\s.\+\-\*/%\(\)]+$')
_MATH_WRAPPER_RE = re.compile(
    r"^\s*(?:what(?:'s| is)|calculate|compute|evaluate|solve)?\s*(.*?)\s*[\?=]*\s*$",
    re.IGNORECASE,
)


def _safe_eval_node(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("non-numeric constant")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")  # guard against hangs/memory blowup
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval_node(node.operand))
    raise ValueError(f"disallowed expression node: {type(node).__name__}")


def _format_number(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def try_math_expression(goal):
    """Only matches when the *entire* cleaned goal is a math expression --
    not when one is embedded in unrelated text. This trades recall for
    precision: a missed match just falls through to the normal (reviewed)
    pipeline, but a bad partial-extraction match would hand back a wrong
    answer with no review step at all."""
    normalized = goal.replace("\u00d7", "*").replace("\u00f7", "/")
    m = _MATH_WRAPPER_RE.match(normalized)
    candidate = (m.group(1) if m else normalized).strip()
    if not candidate or not _MATH_CHARSET_RE.match(candidate):
        return None
    try:
        tree = ast.parse(candidate, mode="eval")
        result = _safe_eval_node(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError, OverflowError, RecursionError):
        return None
    return _format_number(result)


# ---- UUID generation ----

def try_uuid_request(goal):
    lowered = goal.lower()
    if "uuid" in lowered or "guid" in lowered:
        return str(_uuid.uuid4())
    return None


# ---- Timestamps ----

def try_timestamp_request(goal):
    lowered = goal.lower()
    if "unix timestamp" in lowered or "unix time" in lowered or "epoch time" in lowered:
        return str(int(time.time()))
    if "current time" in lowered or "current timestamp" in lowered or "what time is it" in lowered:
        return datetime.now(timezone.utc).isoformat()
    return None


def dispatch_locally(goal):
    """Try each local handler in order. Returns {"handler": str, "result": str}
    on a match, or None if nothing applies -- in which case the caller
    should proceed with the normal (LLM-backed) pipeline unchanged."""
    for handler_name, handler in (
        ("math", try_math_expression),
        ("uuid", try_uuid_request),
        ("timestamp", try_timestamp_request),
    ):
        result = handler(goal)
        if result is not None:
            return {"handler": handler_name, "result": result}
    return None
