#!/usr/bin/env python3
"""
Troubleshoot labi's provider/model setup end to end.

Shows, in order:
  1. The full live model catalog per provider (not just what got picked)
  2. What labi's discovery would actually select right now, or why it failed
  3. Quota status: known daily limits vs. today's tracked usage
  4. Measured performance from past labi runs (success rate, latency, cost)
  5. The actual current routing per task type -- which provider+model
     handles coding/planning/answering/validation right now, in priority
     order, using the exact same registry/scoring logic as `labi run`

Run from the repo root with the venv active:
    python3 troubleshoot_providers.py
"""
import os
import sys

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


def show_measured_performance():
    section("4. Measured performance (from past labi runs, if any)")
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
    section("5. Current routing per task type (same logic as `labi run`)")
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
    show_measured_performance()
    show_task_routing()
    print("\n" + "=" * 72)
    print("Done. Section 5 is the actual decision -- which provider+model handles")
    print("each task type right now, in priority order.")
    print("=" * 72)
