#!/usr/bin/env python3
"""
Troubleshoot labi's provider/model setup end to end.

Shows, in order:
  1. The full live model catalog per provider (not just what got picked)
  2. What labi's discovery would actually select right now, or why it failed
  3. Quota status: known daily limits vs. today's tracked usage
  4. Token/credit balance: live, direct from each provider's own API where
     one is exposed (OpenRouter has a real balance endpoint; Groq/Cerebras
     expose it via rate-limit response headers on a minimal chat call;
     Gemini/Mistral don't expose this at all, and are reported as such)
  5. Measured performance from past labi runs (success rate, latency, cost)
  6. The actual current routing per task type -- which provider+model
     handles coding/planning/answering/validation right now, in priority
     order, using the exact same registry/scoring logic as `labi run`

Run from the repo root with the venv active:
    python3 troubleshoot_providers.py
"""
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from labi.agent import build_registry, STATS_DB_PATH
from labi.providers.adaptive_registry import (
    KNOWN_QUOTAS, compute_quota_status, AVOID_MODELS,
    pick_cerebras_model, pick_groq_model, pick_gemini_model, pick_openrouter_model,
    _fetch_json, LAST_DISCOVERY_ERROR,
)
from labi.providers.stats import ProviderStatsStore

CAPABILITIES = ["coding", "planning", "answering", "validation"]

RAW_FETCHERS = {
    "cerebras": ("CEREBRAS_API_KEY", lambda k: _fetch_json(
        "https://api.cerebras.ai/v1/models", headers={"Authorization": f"Bearer {k}"})),
    "groq": ("GROQ_API_KEY", lambda k: _fetch_json(
        "https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {k}"})),
    "gemini": ("GEMINI_API_KEY", lambda k: _fetch_json(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={k}")),
    "openrouter": ("OPENROUTER_API_KEY", lambda k: _fetch_json(
        "https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {k}"})),
    "mistral": ("MISTRAL_API_KEY", lambda k: _fetch_json(
        "https://api.mistral.ai/v1/models", headers={"Authorization": f"Bearer {k}"})),
}


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def show_raw_catalogs():
    section("1. Raw live model catalogs")
    for name, (env_var, fetch) in RAW_FETCHERS.items():
        key = os.environ.get(env_var, "")
        print(f"\n--- {name} ({env_var}) ---")
        if not key:
            print("  no key set -- skipped")
            continue
        try:
            data = fetch(key)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        if name == "gemini":
            ids = [m["name"].split("/", 1)[-1] for m in data.get("models", [])
                   if "generateContent" in m.get("supportedGenerationMethods", [])]
        else:
            ids = [m["id"] for m in data.get("data", [])]
        if not ids:
            print("  (zero models returned)")
            continue
        avoid = AVOID_MODELS.get(name, set())
        for mid in ids:
            flag = "  [DENYLISTED -- deprecated/paid, will never be auto-picked]" if mid in avoid else ""
            print(f"  - {mid}{flag}")


def show_selected_models():
    section("2. What labi's live discovery picks right now")
    pickers = {"cerebras": pick_cerebras_model, "groq": pick_groq_model,
               "gemini": pick_gemini_model, "openrouter": pick_openrouter_model}
    for name, picker in pickers.items():
        key = os.environ.get(f"{name.upper()}_API_KEY", "")
        if not key:
            print(f"  {name}: no key set")
            continue
        model = picker(key)
        if model:
            print(f"  {name}: {model}")
        else:
            reason = LAST_DISCOVERY_ERROR.get(name, "unknown")
            print(f"  {name}: FAILED to select a model -- {reason}")
    mistral_key = os.environ.get("MISTRAL_API_KEY", "")
    note = "" if mistral_key else "  (no key set)"
    print(f"  mistral: mistral-small-latest (rolling alias, no live discovery needed){note}")


def show_quota():
    section("3. Quota status (known daily limits vs. today's tracked usage)")
    stats_store = ProviderStatsStore(STATS_DB_PATH)
    usage = stats_store.get_all_daily_usage()
    for name, quota in KNOWN_QUOTAS.items():
        today = usage.get(name, {"requests": 0, "tokens": 0})
        status = compute_quota_status(name, today)
        if status is None:
            print(f"  {name}: no published daily limit tracked -- {today['requests']} requests today (untracked ceiling)")
        else:
            print(f"  {name}: {status['used']}/{status['limit']} requests today "
                  f"({status['pct_used']}%), {status['remaining']} remaining")


def _post_json(url, payload, headers=None, timeout=10):
    """POST helper for the minimal chat-completion probe below -- separate
    from _fetch_json (GET-only, imported from adaptive_registry) since
    nothing in the real labi package needs a POST helper, only this
    troubleshooting probe does. Rate-limit headers on OpenAI-compatible
    APIs are sometimes only sent on error responses too, so this reads
    headers off both the success and HTTPError paths, not just 200."""
    merged_headers = {"User-Agent": "curl/8.0", "Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=merged_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()), dict(resp.headers)
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
        except Exception:
            body = {}
        return body, dict(e.headers or {})


def _rate_limit_headers(headers):
    """Pull out whichever of the common x-ratelimit-* token headers a
    provider actually sent -- names aren't fully standardized, so this
    checks a few known variants rather than assuming one exact set."""
    lowered = {k.lower(): v for k, v in (headers or {}).items()}
    out = {}
    for key in ("x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens",
                "x-ratelimit-reset-tokens", "x-ratelimit-limit-requests",
                "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests"):
        if key in lowered:
            out[key] = lowered[key]
    return out


def fetch_token_balance(name, key, model_id=None):
    """Returns a dict describing token/credit balance, or None if this
    provider doesn't expose one at all. Never raises -- failures are
    folded into the returned dict as an 'error' key."""
    if name == "openrouter":
        try:
            data = _fetch_json("https://openrouter.ai/api/v1/auth/key",
                                headers={"Authorization": f"Bearer {key}"})
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        info = data.get("data", {})
        limit = info.get("limit")
        usage = info.get("usage")
        return {
            "kind": "credit_usd",
            "limit": limit,  # None means no hard limit set on this key
            "used": usage,
            "remaining": (limit - usage) if (limit is not None and usage is not None) else None,
            "is_free_tier": info.get("is_free_tier"),
        }

    if name in ("groq", "cerebras"):
        # The /models list endpoint does NOT return rate-limit headers on
        # either provider -- only chat completion responses do (standard
        # for OpenAI-compatible APIs). This costs a trivial sliver of real
        # quota (max_tokens=1) to check, unlike every read-only call above.
        if not model_id:
            return {"kind": "rate_limit_headers",
                    "error": "no model available to probe with (discovery failed above)"}
        url = ("https://api.groq.com/openai/v1/chat/completions" if name == "groq"
               else "https://api.cerebras.ai/v1/chat/completions")
        payload = {"model": model_id, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
        try:
            _, headers = _post_json(url, payload, headers={"Authorization": f"Bearer {key}"})
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        rl = _rate_limit_headers(headers)
        if not rl:
            return {"kind": "rate_limit_headers",
                     "error": "chat-completion response returned no rate-limit headers either"}
        return {"kind": "rate_limit_headers", **rl}

    # gemini, mistral: no per-key balance or quota exposed via the API at all
    return None


def show_token_balance():
    section("4. Token / credit balance (live, direct from each provider)")
    print("  (groq/cerebras checks below spend 1 real token each via a minimal chat call;")
    print("   openrouter checks are free/read-only; gemini/mistral don't expose this)")
    pickers = {"groq": pick_groq_model, "cerebras": pick_cerebras_model}
    for name in ("cerebras", "groq", "gemini", "openrouter", "mistral"):
        key = os.environ.get(f"{name.upper()}_API_KEY", "")
        if not key:
            print(f"  {name}: no key set -- skipped")
            continue
        model_id = pickers[name](key) if name in pickers else None
        balance = fetch_token_balance(name, key, model_id=model_id)
        if balance is None:
            print(f"  {name}: not available -- this provider doesn't expose a per-key "
                  f"balance or quota through its API")
        elif "error" in balance and balance.get("kind") != "rate_limit_headers":
            print(f"  {name}: FAILED -- {balance['error']}")
        elif balance.get("kind") == "credit_usd":
            tier = " (free tier key)" if balance.get("is_free_tier") else ""
            if balance["limit"] is None:
                print(f"  {name}: ${balance['used'] or 0:.4f} used so far, no hard limit set on this key{tier}")
            else:
                print(f"  {name}: ${balance['used'] or 0:.4f} used / ${balance['limit']:.4f} limit "
                      f"(${balance['remaining']:.4f} remaining){tier}")
        elif balance.get("kind") == "rate_limit_headers":
            if "error" in balance:
                print(f"  {name}: {balance['error']}")
            else:
                remaining_t = balance.get("x-ratelimit-remaining-tokens")
                limit_t = balance.get("x-ratelimit-limit-tokens")
                reset_t = balance.get("x-ratelimit-reset-tokens")
                if remaining_t is not None or limit_t is not None:
                    print(f"  {name}: {remaining_t or '?'}/{limit_t or '?'} tokens remaining"
                          + (f" (resets in {reset_t})" if reset_t else ""))
                else:
                    print(f"  {name}: no token-specific headers, only request-count headers: {balance}")


def show_measured_performance():
    section("5. Measured performance (from past labi runs, if any)")
    stats_store = ProviderStatsStore(STATS_DB_PATH)
    all_stats = stats_store.get_all_provider_stats()
    if not all_stats:
        print("  No calls recorded yet at this state path -- rankings below are still on")
        print("  static priority (cold start, fewer than MIN_SAMPLES=4 calls tracked).")
        return
    for row in sorted(all_stats, key=lambda r: (r["provider"], r["capability"])):
        calls = row["calls"]
        success_rate = (row["successes"] / calls * 100) if calls else 0
        avg_latency = (row["total_latency_ms"] / calls / 1000) if calls else 0
        print(f"  {row['provider']} / {row['capability']}: {calls} calls, "
              f"{success_rate:.0f}% success, {avg_latency:.2f}s avg latency, "
              f"${row['total_cost_usd']:.4f} total cost")


def show_task_routing():
    section("6. Current routing per task type (same logic as `labi run`)")
    registry = build_registry()
    stats_store = ProviderStatsStore(STATS_DB_PATH)
    for cap in CAPABILITIES:
        ranked = registry.get_all(cap, stats_store=stats_store)
        if not ranked:
            print(f"\n  {cap}: NO PROVIDER REGISTERED -- falls straight to the offline mock")
            continue
        print(f"\n  {cap}:")
        for i, p in enumerate(ranked):
            marker = "  <- would be used" if i == 0 else ""
            print(f"    {i + 1}. {p.name} ({p.model}){marker}")


if __name__ == "__main__":
    show_raw_catalogs()
    show_selected_models()
    show_quota()
    show_token_balance()
    show_measured_performance()
    show_task_routing()
    print("\n" + "=" * 72)
    print("Done. Section 6 is the actual decision -- which provider+model handles")
    print("each task type right now, in priority order.")
    print("=" * 72)
