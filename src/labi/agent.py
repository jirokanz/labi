#!/usr/bin/env python3
"""
Labi -- Interactive Autonomous Agent.

Loop: plan -> generate code -> show it -> you approve / edit / give feedback
-> execute (sandboxed) -> auto-fix on failure -> save to memory for replay.
"""

import difflib
import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Must be set before litellm/huggingface_hub get imported, or the progress
# bar for the one-off tokenizer download (litellm falls back to an HF
# tokenizer to count tokens for models it doesn't recognize, e.g. some
# groq/llama routes) prints a raw tqdm bar straight into the terminal.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from dotenv import load_dotenv
import litellm

litellm.suppress_debug_info = True
litellm.set_verbose = False

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from labi.intelligence.classifier import TaskClassifier
from labi.intelligence.local_dispatcher import dispatch_locally
from labi.providers.adaptive_registry import (
    BaseProvider,
    AdaptiveProviderRegistry as ProviderRegistry,
    pick_cerebras_model,
    pick_groq_model,
    pick_gemini_model,
    pick_openrouter_model,
    KNOWN_QUOTAS,
    compute_quota_status,
    LAST_DISCOVERY_ERROR,
)
from labi.providers.stats import ProviderStatsStore
from labi.core.task import Task
from labi.context.manager import ContextManager
from labi.context.prompt_builder import PromptBuilder
from labi.context.artifact import ArtifactStore
from labi.agents.planner import PlannerAgent
from labi.agents.executor import ExecutorAgent
from labi.agents.validator import ValidatorAgent
from labi.workflows.software_dev import SoftwareDevelopmentWorkflow
from labi.providers.cost import CostTracker
from labi.tools import web as web_search
from labi.tools.sources import SourceStore

load_dotenv()

# ---------- Configuration ----------
# Anchored to a fixed location under $HOME (matching WORKSPACE_ROOT below),
# not the current working directory -- otherwise running 'labi' from
# different directories (e.g. via the global /usr/local/bin/labi wrapper,
# which doesn't cd anywhere) silently fragments provider stats and memory
# across whichever cwd you happened to be in each time.
_LABI_STATE_DIR = Path.home() / "labi" / "state"
_LABI_STATE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = os.getenv("LABI_DB_PATH", str(_LABI_STATE_DIR / "memory.db"))
STATS_DB_PATH = os.getenv("LABI_STATS_DB_PATH", str(_LABI_STATE_DIR / "provider_stats.db"))
SOURCES_DB_PATH = os.getenv("LABI_SOURCES_DB_PATH", str(_LABI_STATE_DIR / "sources.db"))
WORKSPACE_ROOT = Path.home() / "labi" / "workspace"
MAX_FIX_ATTEMPTS = 3
EXECUTION_TIMEOUT = 20
MAX_OUTPUT_BYTES = 1_048_576

# ---------- Terminal formatting ----------
from labi.providers.generation import (  # noqa: E402  (see providers/generation.py docstring)
    _c, stream_generate, format_code_block, _highlight_line, _PY_KEYWORDS, _STRING_RE, ProviderCallError,
)


# ---------- Code extraction ----------
from labi.providers.generation import estimate_tokens  # noqa: E402  (see providers/generation.py docstring)

from labi.tools.python.sandbox import extract_code, execute_code  # noqa: E402  (see tools/python/sandbox.py docstring)


# ---------- Memory ----------
class MemoryDB:
    def __init__(self, path):
        import sqlite3
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                goal TEXT,
                plan TEXT,
                code TEXT,
                answer TEXT,
                success BOOLEAN,
                provider TEXT,
                workspace_path TEXT,
                replay_depth INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        cur = self.conn.execute("PRAGMA table_info(tasks)")
        existing = [row[1] for row in cur.fetchall()]
        required = {
            "plan": "TEXT", "code": "TEXT", "answer": "TEXT", "success": "BOOLEAN",
            "provider": "TEXT", "workspace_path": "TEXT", "replay_depth": "INTEGER DEFAULT 0",
            "cost_usd": "REAL DEFAULT 0",
        }
        for col, col_type in required.items():
            if col not in existing:
                self.conn.execute(f"ALTER TABLE tasks ADD COLUMN {col} {col_type}")
                self.conn.commit()

    def find_similar(self, goal):
        words = set(goal.lower().split())
        cursor = self.conn.execute(
            "SELECT id, goal, plan, code, answer, provider, workspace_path, replay_depth "
            "FROM tasks WHERE success=1"
        )
        best = None
        best_sim = 0.0
        for row in cursor.fetchall():
            task_words = set(row[1].lower().split())
            similarity = len(words & task_words) / len(words | task_words) if words else 0
            if similarity > 0.5 and similarity > best_sim:
                best_sim = similarity
                best = {
                    "id": row[0], "goal": row[1], "plan": row[2], "code": row[3],
                    "answer": row[4], "provider": row[5], "workspace_path": row[6],
                    "replay_depth": row[7] or 0, "similarity": similarity,
                }
        return best

    def save_task(self, task_id, goal, plan, code, answer, provider,
                  workspace_path=None, success=True, replay_depth=0, cost_usd=0.0):
        self.conn.execute(
            "INSERT OR REPLACE INTO tasks "
            "(id, goal, plan, code, answer, success, provider, workspace_path, replay_depth, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, goal, plan, code, answer, success, provider, workspace_path, replay_depth, cost_usd),
        )
        self.conn.commit()


# ---------- Workspace ----------
class WorkspaceManager:
    def __init__(self, root=WORKSPACE_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_workspace(self, task_id):
        path = self.root / f"task_{task_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_code(self, workspace_path, code):
        code_path = workspace_path / "code.py"
        code_path.write_text(code)
        return code_path

    def save_execution(self, workspace_path, stdout, stderr, exit_code):
        exec_path = workspace_path / "execution.json"
        exec_path.write_text(json.dumps({
            "stdout": stdout, "stderr": stderr, "exit_code": exit_code,
            "timestamp": datetime.now().isoformat(),
        }, indent=2))
        return exec_path

    def save_plan(self, workspace_path, plan):
        plan_path = workspace_path / "plan.txt"
        plan_path.write_text(plan)
        return plan_path

    def clone_workspace(self, source_task_id, new_task_id):
        source_path = self.root / f"task_{source_task_id}"
        if not source_path.exists():
            return None
        new_path = self.root / f"task_{new_task_id}"
        import shutil
        shutil.copytree(source_path, new_path, dirs_exist_ok=True)
        exec_file = new_path / "execution.json"
        if exec_file.exists():
            exec_file.unlink()
        return new_path


# ---------- Replay ----------
class ReplayManager:
    MAX_REPLAY_DEPTH = 3

    def __init__(self, memory_db, workspace_manager):
        self.memory_db = memory_db
        self.workspace = workspace_manager

    def replay_task(self, cached, new_goal):
        depth = (cached.get("replay_depth") or 0) + 1
        if depth > self.MAX_REPLAY_DEPTH:
            print(f"   Replay depth exceeded ({depth} > {self.MAX_REPLAY_DEPTH}); regenerating fresh.")
            return None
        new_task_id = f"replay_{uuid.uuid4().hex[:8]}"
        new_workspace = self.workspace.clone_workspace(cached["id"], new_task_id)
        if not new_workspace:
            return None
        code = cached.get("code")
        if code:
            self.workspace.save_code(new_workspace, code)
        self.memory_db.save_task(
            task_id=new_task_id, goal=new_goal, plan=cached.get("plan"), code=code,
            answer=None, provider=cached.get("provider"), workspace_path=str(new_workspace),
            success=False, replay_depth=depth,
        )
        return {"task_id": new_task_id, "workspace_path": str(new_workspace), "code": code, "replay_depth": depth}


class MemoryRouter:
    def __init__(self, memory_db):
        self.db = memory_db

    def route(self, goal):
        cached = self.db.find_similar(goal)
        if cached:
            return {"decision": "reuse", "candidate": cached}
        return {"decision": "regenerate", "candidate": None}


# ---------- Sandboxed execution ----------
# execute_code() is imported from labi.tools.python.sandbox above.


# ---------- Provider registry ----------
def print_provider_rankings(registry, stats_store):
    print(f"\n{_c('Provider rankings (measured performance, falls back to static priority until ' + str(ProviderRegistry.MIN_SAMPLES) + '+ calls):', 'bold')}")
    capabilities = sorted({cap for p in registry.providers for cap in p.capabilities})
    for cap in capabilities:
        candidates = registry.get_all(cap, stats_store=stats_store)
        if not candidates:
            continue
        print(f"\n  {_c(cap, 'cyan')}:")
        for rank, p in enumerate(candidates, 1):
            stats = stats_store.get_provider_stats(p.name, cap)
            if stats and stats["calls"] >= ProviderRegistry.MIN_SAMPLES:
                rate = stats["successes"] / stats["calls"] * 100
                avg_ms = stats["total_latency_ms"] / stats["calls"]
                detail = f"{rate:.0f}% success, {avg_ms:.0f}ms avg over {stats['calls']} calls"
            elif stats:
                detail = f"only {stats['calls']} calls so far -- using static priority ({p.priority_for(cap)})"
            else:
                detail = f"no data yet -- using static priority ({p.priority_for(cap)})"
            ctx = f", {p.context_window // 1000}K context" if p.context_window else ""
            quota_factor = registry._quota_factor(p, stats_store)
            quota_note = f", quota-dampened {quota_factor:.0%}" if quota_factor < 1.0 else ""
            print(f"    {rank}. {p.name:12s} {_c(detail + ctx + quota_note, 'grey')}")


def print_cost_summary(stats_store):
    total = stats_store.get_total_cost()
    print(f"\n{_c('Lifetime estimated cost: $' + f'{total:.6f}', 'bold')}")
    stats = stats_store.get_all_provider_stats()
    if not stats:
        print("  (no calls recorded yet)")
        return
    stats = [s for s in stats if s["calls"] > 0]
    stats.sort(key=lambda s: s["total_cost_usd"], reverse=True)
    print(f"\n  {'provider':12s} {'capability':12s} {'calls':>6s} {'cost':>12s}")
    for s in stats:
        cost_str = f"${s['total_cost_usd']:.6f}"
        print(f"  {s['provider']:12s} {s['capability']:12s} {s['calls']:>6d} {cost_str:>12s}")
    print(_c("\n  Note: $0.000000 can mean genuinely free OR that litellm has no "
             "pricing data for that model -- not a guarantee of zero cost.", "grey"))


def print_quota_summary(stats_store):
    from datetime import datetime as _dt, timezone as _tz
    usage_today = stats_store.get_all_daily_usage()
    now = _dt.now(_tz.utc)
    hours_to_reset = 24 - now.hour - (now.minute / 60)
    print(f"\n{_c('Today’s usage (resets ~' + f'{hours_to_reset:.1f}h at UTC midnight):', 'bold')}")
    if not usage_today:
        print("  (no calls recorded today)")
        return
    for provider, usage in sorted(usage_today.items()):
        status = compute_quota_status(provider, usage)
        if status:
            bar_len = 20
            filled = int(bar_len * status["pct_used"] / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            color = "red" if status["pct_used"] >= 90 else ("yellow" if status["pct_used"] >= 70 else "green")
            print(f"  {provider:12s} {_c(bar, color)} {status['used']}/{status['limit']} requests "
                  f"({status['pct_used']}%) -- ~{usage['tokens']:,} tokens today")
        else:
            print(f"  {provider:12s} {usage['requests']} requests, ~{usage['tokens']:,} tokens today "
                  f"{_c('(no published daily limit tracked)', 'grey')}")
    print(_c("\n  Note: limits are last-known-good, not live-verified -- see KNOWN_QUOTAS in agent.py.", "grey"))


def print_failure_summary(stats_store):
    summary = stats_store.get_failure_summary()
    print(f"\n{_c('Failure breakdown (by provider, reason):', 'bold')}")
    if not summary:
        print("  (no failures recorded)")
        return
    for row in summary:
        print(f"  {row['provider']:12s} {row['reason']:16s} {row['count']:>4d}x")

    recent = stats_store.get_recent_failures(limit=5)
    print(f"\n{_c('Most recent failures:', 'bold')}")
    for f in recent:
        msg = f["message"][:80] + ("..." if len(f["message"]) > 80 else "")
        print(f"  {_c(f['timestamp'], 'grey')} {f['provider']:10s} [{f['capability']}] "
              f"{_c(f['reason'], 'yellow')}: {msg}")


def build_registry():
    registry = ProviderRegistry()

    def register_provider(name, model, api_base, key_env, capabilities, priority, capability_priority=None):
        key = os.getenv(key_env)
        if key:
            registry.register(BaseProvider(name, model, api_base, key, capabilities, priority, capability_priority))
            return True
        return False

    def register_discovered(name, model_prefix, api_base, key_env, capabilities, priority, picker,
                             capability_priority=None, context_window=None):
        """Same as register_provider, but the model id is looked up live
        via `picker` (see pick_*_model in providers/adaptive_registry.py)
        instead of being hardcoded -- so a provider retiring/renaming a
        model doesn't silently break this file. Skips registration (with
        a warning) if discovery fails or no key is set."""
        key = os.getenv(key_env)
        if not key:
            return False
        model_id = picker(key)
        if not model_id:
            reason = LAST_DISCOVERY_ERROR.get(name, "unknown reason")
            print(_c(f"   Warning: could not discover a working {name} model "
                      f"({reason}) -- skipping {name} this session.", "yellow"))
            return False
        registry.register(BaseProvider(name, f"{model_prefix}/{model_id}", api_base, key,
                                        capabilities, priority, capability_priority, context_window))
        return True

    # Groq: fast general-purpose model, primary for both planning and
    # coding for now. llama-3.3-70b-versatile (the old hardcoded model)
    # is deprecated by Groq, shutdown 08/16/2026 -- discovered live now
    # (see pick_groq_model) instead of hardcoding its replacement.
    # context_window: every model in PREFERRED_GROQ_MODELS (gpt-oss-120b,
    # qwen3.6-27b, llama-3.3) is independently verified at 128K+ on Groq,
    # so 128000 is a safe floor regardless of which one discovery lands on.
    register_discovered("groq", "groq", "https://api.groq.com/openai/v1",
                         "GROQ_API_KEY", ["planning", "coding"], 10, pick_groq_model,
                         context_window=128000)
    # DeepSeek removed for now -- not actually free (per-token paid API),
    # so it shouldn't be a default in a "free/cheap providers" pool.
    # Re-add with capability_priority={"coding": 5} on groq's coding entry
    # dropped to e.g. 25 if you want it back as the specialized coder later:
    # register_provider("deepseek", "deepseek/deepseek-chat", "https://api.deepseek.com/v1",
    #                    "DEEPSEEK_API_KEY", ["coding", "planning"], priority=20,
    #                    capability_priority={"coding": 5})
    # OpenRouter: model id is discovered live and restricted to ids
    # OpenRouter prices at $0 (see pick_openrouter_model) -- the old
    # hardcoded id (meta-llama/llama-3.1-70b-instruct, no ':free' suffix)
    # was actually the paid route, which contradicts this being a
    # free-tier pool.
    # Also given "coding" and "validation" here as a fallback: previously
    # coding had only Groq and validation had only Gemini, so either
    # one's key being bad or its discovery failing dropped that whole
    # capability straight to the offline mock with no live LLM at all.
    # capability_priority pins OpenRouter behind validation's primary
    # (Gemini, 40) -- without this override OpenRouter's general priority
    # (30) would rank AHEAD of Gemini for validation, which is backwards;
    # coding needs no override since OpenRouter's 30 is already behind
    # Groq's 10.
    # context_window intentionally left unset (None) -- which specific
    # free model OpenRouter serves varies too much to guess a safe number.
    # web_search is added to OpenRouter and Gemini (not Groq, which is
    # pinned to planning/coding above) so a dedicated web_search-quality
    # provider isn't required for this capability to resolve -- it just
    # reuses the same general-purpose "answering" models to summarize
    # search results, same as the plain answering path.
    register_discovered("openrouter", "openrouter", "https://openrouter.ai/api/v1",
                         "OPENROUTER_API_KEY", ["answering", "planning", "coding", "validation", "web_search"], 30,
                         pick_openrouter_model, capability_priority={"validation": 45})
    # Gemini: gemini-2.5-flash (the old hardcoded model) is deprecated,
    # shutdown 10/16/2026, and some accounts report it already failing
    # ahead of that date -- discovered live now (see pick_gemini_model).
    # context_window: every model in PREFERRED_GEMINI_MODELS (the 3.x
    # Flash family) is documented at ~1M tokens.
    # model_prefix is "openai" (NOT "gemini") on purpose: litellm treats
    # a "gemini/..." model string as its own NATIVE Gemini integration,
    # which validates against litellm's own internal model list and
    # largely ignores the custom api_base below -- so a freshly-discovered
    # model newer than that internal list (e.g. gemini-3.5-flash) fails
    # with NotFoundError before our api_base is ever reached. "openai/..."
    # makes litellm do a plain OpenAI-format passthrough to whatever
    # api_base we give it, with no internal model-name validation --
    # which is what we actually want, since we're pointing this at
    # Google's own OpenAI-compatible endpoint below, not litellm's
    # hardcoded Gemini route.
    register_discovered("gemini", "openai", "https://generativelanguage.googleapis.com/v1beta/openai/",
                         "GEMINI_API_KEY", ["answering", "validation", "web_search"], 40, pick_gemini_model,
                         context_window=1000000, capability_priority={"answering": 25})
    # NVIDIA NIM removed for now -- its free access is a "hosted evaluation
    # endpoint" rather than a confirmed indefinite production free tier.
    # Re-add if you verify your own account's limits are workable:
    # register_provider("nvidia", "nvidia/llama-3.1-70b-instruct", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", ["planning", "coding"], 50)
    # Upgraded from llama3.1-8b -- an 8B model was too weak to be a
    # meaningful answering fallback. Model name is discovered live
    # (see pick_cerebras_model) rather than hardcoded, since Cerebras's
    # free catalog is known to change without notice.
    register_discovered("cerebras", "cerebras", "https://api.cerebras.ai/v1",
                         "CEREBRAS_API_KEY", ["answering"], 60, pick_cerebras_model)
    # Mistral's "-latest" alias is a rolling pointer Mistral itself keeps
    # pointed at its newest Small model, so unlike the others it doesn't
    # need live discovery to stay valid -- it can still silently change
    # behavior/pricing underneath you, just won't 404.
    register_provider("mistral", "mistral/mistral-small-latest", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", ["answering"], 70)
    # Cohere removed for now -- same "free evaluation limits" trial flavor
    # as NVIDIA above, not a clearly indefinite free tier. Re-add if
    # verified workable on your account:
    # register_provider("cohere", "cohere/command-r", "https://api.cohere.ai/v1", "COHERE_API_KEY", ["answering"], 80)

    registry.register(BaseProvider("mock", "mock", "", "", ["text_generation"], 999))
    return registry


class SessionContext:
    """Rolling window of recent goal/response exchanges for the whole
    session -- separate from the per-task 'history' used inside the code
    review loop. Without this, every new goal is a blank slate: a
    follow-up like 'in terms of X' has no idea what X is a follow-up to."""

    def __init__(self, max_turns=3):
        self.max_turns = max_turns
        self.turns = []  # list of {"goal": str, "kind": str, "summary": str}

    def add(self, goal, kind, summary):
        summary = (summary or "").strip()
        if len(summary) > 300:
            summary = summary[:300] + "..."
        self.turns.append({"goal": goal, "kind": kind, "summary": summary})
        self.turns = self.turns[-self.max_turns:]

    def reset(self):
        self.turns = []

    def as_context(self):
        if not self.turns:
            return ""
        lines = ["Recent conversation (for context on follow-up questions):"]
        for t in self.turns:
            lines.append(f"- User asked: {t['goal']}")
            if t["summary"]:
                lines.append(f"  Response summary: {t['summary']}")
        return "\n".join(lines) + "\n\n"


# Continuation phrases that signal "this is a follow-up to what I just
# asked", not a fresh task -- the plain keyword list below missed these
# (e.g. "in terms of ai token provider?" doesn't start with what/how/etc,
# so it was falling through to the coding path instead of answering).
CONTINUATION_PREFIXES = [
    "in terms of", "in term of", "what about", "how about", "and ",
    "also ", "regarding", "about ", "for ", "with respect to", "so ",
    "but ", "ok so", "okay so",
]


def needs_live_info(goal, classifier=None):
    """Delegates to TaskClassifier.classify()'s requires_web signal
    instead of a standalone keyword list. That signal covers two
    families: goals that name their own time-sensitivity directly
    ("latest", "current", "today"...), plus "who is the <role>"
    role-holder questions (e.g. "who is the CEO of OpenAI?"), which have
    no freshness keyword at all but are just as time-sensitive -- asking
    about a role implicitly means "whoever holds it now". See
    classifier.py's _detect_requires_web for the confidence scoring."""
    classifier = classifier or TaskClassifier()
    return classifier.classify(goal).requires_web


def try_web_search_answer(goal, task_id, registry, stats_store, source_store, memory_db, workspace, session):
    """The observe -> summarize half of the agent loop: search the web,
    then have a provider answer using only those results with inline
    citations. Returns True if it produced and saved an answer (caller
    should treat the goal as handled this turn), False if search or
    summarization couldn't happen for any reason (caller should fall
    back to the normal plan/code/answer pipeline instead)."""
    print(_c("   [web_search] Looking this up...", "cyan"))
    results = web_search.search(goal)
    if results is None:
        reason = web_search.LAST_ERROR.get("search", "unavailable")
        print(_c(f"   Web search unavailable ({reason}) -- "
                  f"answering from the model's own knowledge instead.", "yellow"))
        return False
    if not results:
        print(_c("   No web results found -- answering from the model's own knowledge instead.", "yellow"))
        return False

    candidates = registry.get_all("web_search", stats_store=stats_store)
    if not candidates:
        print("No provider configured for web_search -- answering from the model's own knowledge instead.")
        return False

    sources_context = "\n\n".join(
        f"[{i + 1}] {r['title']} ({r['url']})\n{r['snippet']}"
        for i, r in enumerate(results)
    )
    prompt = (
        f"Using ONLY the sources below, answer this question: {goal}\n\n"
        f"Sources:\n{sources_context}\n\n"
        "Cite sources inline as [1], [2], etc. If the sources don't actually "
        "answer the question, say so rather than guessing."
    )

    cost_tracker = CostTracker()
    for provider in candidates:
        try:
            answer = stream_generate(
                provider, prompt,
                system_prompt="You are a research assistant. Be concise and cite sources inline.",
                max_tokens=512, label="Reading sources",
                stats_store=stats_store, capability="web_search", cost_tracker=cost_tracker,
            )
        except ProviderCallError as e:
            print(_c(f"   [{provider.name}] failed ({e.reason}) -- trying next provider...", "yellow"))
            continue
        except Exception as e:
            print(f"Provider {provider.name} failed: {e}")
            continue

        memory_db.save_task(task_id=task_id, goal=goal, plan=None, code=None,
                             answer=answer, provider=provider.name,
                             workspace_path=str(workspace.create_workspace(task_id)),
                             success=True, cost_usd=cost_tracker.total)
        for r in results:
            source_store.record(task_id, r["url"], r["title"], r["snippet"])
        session.add(goal, "answer", answer)
        if cost_tracker.total > 0:
            print(_c(f"   (cost: ${cost_tracker.total:.6f})", "grey"))
        print(_c("\nSources:", "grey"))
        for i, r in enumerate(results, 1):
            print(_c(f"  [{i}] {r['title']} -- {r['url']}", "grey"))
        return True

    print("All web_search providers failed -- answering from the model's own knowledge instead.")
    return False


def is_question_or_followup(goal, session):
    question_keywords = ["what", "how", "why", "when", "where", "who", "is", "are",
                          "can", "do", "does", "will", "would", "could", "should"]
    lowered = goal.lower()
    if any(lowered.startswith(kw) for kw in question_keywords) and len(goal.split()) > 2:
        return True
    if any(lowered.startswith(p) for p in CONTINUATION_PREFIXES):
        return True
    # A short fragment with no action verb, right after a prior turn, is
    # almost always a continuation rather than a brand-new coding task.
    action_verbs = ["write", "make", "build", "create", "generate", "script",
                    "code", "check", "fix", "add", "implement", "run"]
    if session.turns and len(goal.split()) <= 6 and not any(v in lowered for v in action_verbs):
        return True
    return False


def main():
    print("\nLabi -- Interactive Agent\n")

    registry = build_registry()
    print(f"Registered {len(registry.providers)} providers:")
    for p in registry.providers:
        print(f"   - {p.name} (caps: {', '.join(p.capabilities)}, priority {p.priority})")

    memory_db = MemoryDB(DB_PATH)
    stats_store = ProviderStatsStore(STATS_DB_PATH)
    source_store = SourceStore(SOURCES_DB_PATH)
    workspace = WorkspaceManager()
    router = MemoryRouter(memory_db)
    replay_manager = ReplayManager(memory_db, workspace)
    session = SessionContext()

    while True:
        goal = input("\nYour goal/question (or 'exit', 'providers', 'reset', 'cost', 'quota', 'failures'): ").strip()
        if not goal:
            continue
        if goal.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if goal.lower() == "providers":
            print_provider_rankings(registry, stats_store)
            continue
        if goal.lower() == "reset":
            session.reset()
            print("Conversation context cleared.")
            continue
        if goal.lower() == "cost":
            print_cost_summary(stats_store)
            continue
        if goal.lower() == "quota":
            print_quota_summary(stats_store)
            continue
        if goal.lower() == "failures":
            print_failure_summary(stats_store)
            continue

        local = dispatch_locally(goal)
        if local is not None:
            print(_c(f"   [local:{local['handler']}] No API call needed.", "cyan"))
            print(local["result"])
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            memory_db.save_task(task_id=task_id, goal=goal, plan=None, code=None,
                                 answer=local["result"], provider=f"local:{local['handler']}",
                                 workspace_path=None, success=True, cost_usd=0.0)
            session.add(goal, "answer", local["result"])
            continue

        task_id = f"task_{uuid.uuid4().hex[:8]}"

        if needs_live_info(goal):
            handled = try_web_search_answer(
                goal, task_id, registry, stats_store, source_store, memory_db, workspace, session)
            if handled:
                continue
            # No key configured / no results / all summarizer providers
            # failed -- fall through to the normal pipeline below so the
            # model still attempts an answer from its own knowledge
            # rather than the goal going completely unhandled.

        decision = router.route(goal)
        if decision["decision"] == "reuse" and decision["candidate"]:
            cached = decision["candidate"]
            print(f"\nFound similar memory (similarity {cached['similarity']:.2f}): '{cached['goal']}'")
            use_it = input("Reuse it? [y/N/f=regenerate with feedback] ").strip().lower()
            if use_it == "y":
                replay_result = replay_manager.replay_task(cached, goal)
                if replay_result:
                    code = replay_result["code"]
                    print("Replaying previous solution...")
                    stdout, stderr, exit_code = execute_code(code)
                    if exit_code == 0:
                        print(f"Replay succeeded.\nOutput:\n{stdout}")
                        memory_db.save_task(
                            task_id=replay_result["task_id"], goal=goal, plan=cached.get("plan"),
                            code=code, answer=stdout, provider=cached.get("provider"),
                            workspace_path=replay_result["workspace_path"], success=True,
                            replay_depth=replay_result["replay_depth"],
                        )
                        continue
                    else:
                        print("Replay failed, falling back to fresh generation.")
            # 'n' or replay failed or depth exceeded -> fall through to fresh generation

        is_question = is_question_or_followup(goal, session)

        if is_question:
            context = session.as_context()
            prompt = f"{context}New message: {goal}" if context else goal
            candidates = registry.get_all("answering", stats_store=stats_store,
                                           min_context=estimate_tokens(prompt) + 200)
            if not candidates:
                print("No provider configured for answering (or none with enough context for this conversation -- try 'reset').")
                continue
            cost_tracker = CostTracker()
            for provider in candidates:
                try:
                    answer = stream_generate(
                        provider, prompt,
                        system_prompt="You are a helpful assistant. Answer concisely and accurately. "
                                       "If recent conversation context is given, treat the new message as a "
                                       "follow-up to it unless it clearly changes topic.",
                        max_tokens=512, label="Thinking",
                        stats_store=stats_store, capability="answering", cost_tracker=cost_tracker,
                    )
                    memory_db.save_task(task_id=task_id, goal=goal, plan=None, code=None,
                                         answer=answer, provider=provider.name,
                                         workspace_path=str(workspace.create_workspace(task_id)), success=True,
                                         cost_usd=cost_tracker.total)
                    session.add(goal, "answer", answer)
                    if cost_tracker.total > 0:
                        print(_c(f"   (cost: ${cost_tracker.total:.6f})", "grey"))
                    break
                except ProviderCallError as e:
                    print(_c(f"   [{provider.name}] failed ({e.reason}) -- trying next provider...", "yellow"))
                    continue
                except Exception as e:
                    print(f"Provider {provider.name} failed: {e}")
                    continue
            continue

        task_context_manager = ContextManager(artifact_store=ArtifactStore())
        prompt_builder = PromptBuilder(task_context_manager.artifact_store)
        cost_tracker = CostTracker()

        planner_agent = PlannerAgent(registry, prompt_builder, stats_store=stats_store, cost_tracker=cost_tracker)
        executor_agent = ExecutorAgent(registry, prompt_builder, stats_store=stats_store, cost_tracker=cost_tracker)
        validator_agent = ValidatorAgent(registry, prompt_builder, stats_store=stats_store, cost_tracker=cost_tracker)

        workflow = SoftwareDevelopmentWorkflow(task_context_manager)
        workflow.add_agent(planner_agent).add_agent(executor_agent).add_agent(validator_agent)

        conv_id = task_context_manager.create_conversation()
        task_context_manager.add_message(conv_id, "user", session.as_context())
        task = Task(id=task_id, goal=goal, context_id=conv_id)
        result = workflow.execute(task)

        ws_path = workspace.create_workspace(task_id)
        final_snapshot = task_context_manager.snapshot(task_id)
        plan_text = "\n".join(final_snapshot.plan) if final_snapshot.plan else ""
        workspace.save_plan(ws_path, plan_text)
        artifacts = result.get("artifacts") or task_context_manager.get_task_artifacts(task_id)
        code = artifacts[-1].content if artifacts else ""
        provider_name = executor_agent.last_provider or "unknown"
        if code:
            workspace.save_code(ws_path, code)
        if final_snapshot.execution_stdout is not None:
            workspace.save_execution(ws_path, final_snapshot.execution_stdout or "",
                                      final_snapshot.execution_stderr or "",
                                      final_snapshot.execution_exit_code)

        if cost_tracker.total > 0:
            print(_c(f"Task cost so far: ${cost_tracker.total:.6f} across {cost_tracker.calls} calls", "grey"))

        if result["status"] == "completed":
            stdout = final_snapshot.execution_stdout or ""
            print("\nTask completed and saved to memory.")
            memory_db.save_task(task_id=task_id, goal=goal, plan=plan_text, code=code, answer=stdout,
                                 provider=provider_name, workspace_path=str(ws_path), success=True,
                                 cost_usd=cost_tracker.total)
            session.add(goal, "code", f"Wrote and ran code for: {goal}. Output: {stdout[:150]}")
        else:
            print(f"\nTask not completed: {result.get('error', 'unknown error')}")
            memory_db.save_task(task_id=task_id, goal=goal, plan=plan_text, code=code,
                                 answer=f"INCOMPLETE: {result.get('error', '')}", provider=provider_name,
                                 workspace_path=str(ws_path), success=False, cost_usd=cost_tracker.total)


if __name__ == "__main__":
    main()
