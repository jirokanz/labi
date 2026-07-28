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
import subprocess
import sys
import tempfile
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

from labi.tools.python.security import validate_code as static_validate_code
from labi.tools.python.limits import make_preexec_fn, enforce_output_limit
from labi.intelligence.classifier import TaskClassifier
from labi.intelligence.local_dispatcher import dispatch_locally
from labi.intelligence.types import RiskLevel
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
from labi.providers.generation import _c, stream_generate  # noqa: E402  (see providers/generation.py docstring)

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
    replaces the old bare `print(code)` wall of text."""
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


# ---------- Code extraction ----------
def estimate_tokens(text):
    """Rough chars/4 estimate -- deliberately not calling litellm.token_counter
    here (that's a heavier, model-specific call). This only needs to be
    right enough to skip a provider whose context window is clearly too
    small, not exact."""
    return len(text) // 4


def extract_code(text):
    match = re.search(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()



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
def execute_code(code, timeout=EXECUTION_TIMEOUT):
    """Run generated code with: static validation, a subprocess boundary,
    and (on POSIX) CPU/memory rlimits. This was previously a bare
    subprocess.run with no checks at all."""
    code = extract_code(code)

    violations = static_validate_code(code)
    if violations:
        return "", "Blocked before execution:\n- " + "\n- ".join(violations), -2

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout,
            preexec_fn=make_preexec_fn(cpu_seconds=timeout, memory_mb=256),
        )
        stdout, stderr, exit_code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, exit_code = "", "Execution timed out", -1
    finally:
        os.unlink(fname)

    stdout = enforce_output_limit(stdout, MAX_OUTPUT_BYTES)
    stderr = enforce_output_limit(stderr, MAX_OUTPUT_BYTES)
    return stdout, stderr, exit_code


def show_diff(old_code, new_code):
    if not old_code:
        return
    diff = list(difflib.unified_diff(
        old_code.splitlines(), new_code.splitlines(),
        lineterm="", fromfile="previous", tofile="updated",
    ))
    if not diff:
        print("   (no changes)")
        return
    for line in diff[:60]:
        if line.startswith("+") and not line.startswith("+++"):
            print(f"\033[32m{line}\033[0m")
        elif line.startswith("-") and not line.startswith("---"):
            print(f"\033[31m{line}\033[0m")
        else:
            print(line)



def ask_action(prompt="What next?", options=("r", "e", "f", "s")):
    labels = {
        "r": "[r]un this code",
        "e": "[e]dit it yourself in $EDITOR-style inline paste",
        "f": "[f]eedback -- describe what to change and I'll regenerate",
        "s": "[s]kip this task",
    }
    print(prompt)
    for o in options:
        print(f"  {labels[o]}")
    while True:
        choice = input("> ").strip().lower()
        if choice in options:
            return choice
        print(f"Please choose one of: {', '.join(options)}")


def validate_result(goal, code, stdout, registry, stats_store, cost_tracker):
    """The Validator stage: exit_code == 0 only proves the code didn't
    crash, not that it did what was asked. This runs a second, independent
    LLM pass (the 'validation' capability -- Gemini primary, OpenRouter
    fallback) that actually checks the output against the goal.

    Returns (passed: bool, reason: str, ran: bool) -- ran=False means no
    validation provider was available, so the caller should treat this as
    'unverified' rather than 'failed'."""
    validator = registry.get_best("validation", stats_store=stats_store)
    if not validator:
        return True, "No validation provider available -- unverified.", False

    check_prompt = (
        f"Goal: {goal}\n\n"
        f"Code that was run:\n{code}\n\n"
        f"Output produced:\n{stdout[:1500]}\n\n"
        "Does this output actually accomplish the stated goal? This is a real "
        "check, not a rubber stamp -- look for wrong values, missing parts of "
        "the request, or output that runs without error but doesn't answer "
        "what was asked.\n"
        "Respond with exactly one line: 'PASS' or 'FAIL: <one-sentence reason>'."
    )
    try:
        raw = stream_generate(
            validator, check_prompt, max_tokens=100, label="Validating", render="text",
            stats_store=stats_store, capability="validation", cost_tracker=cost_tracker,
        )
    except Exception as e:
        return True, f"Validator call failed ({e}) -- unverified.", False

    verdict = raw.strip()
    if verdict.upper().startswith("PASS"):
        return True, "Validator confirmed the output matches the goal.", True
    return False, verdict, True


def run_with_autofix_interactive(goal, plan, coder, workspace, registry, stats_store, task_id, cost_tracker):
    """Interactive plan->code->review loop. Unlike the old one-shot
    generate-and-run, this shows you the code before executing and lets
    you approve, hand-edit, or give free-text feedback to regenerate --
    the conversation history is kept so feedback compounds."""
    history = []
    code = None
    provider = None
    ws_path = workspace.root / f"task_{task_id}"

    risk_profile = TaskClassifier().classify(goal)
    if risk_profile.risk != RiskLevel.LOW:
        kw = ", ".join(risk_profile.keywords) if risk_profile.keywords else "goal wording"
        print(_c(f"   Risk assessment: {risk_profile.risk.value.upper()} (flagged on: {kw})", "yellow"))

    gen_prompt = f"Write Python code for this goal:\nGoal: {goal}\nPlan: {plan}\n\nOutput only the code, no explanation."
    raw = stream_generate(coder, gen_prompt, max_tokens=1024, label="Generating code", render="code", stats_store=stats_store, capability="coding", cost_tracker=cost_tracker)
    code = extract_code(raw)
    provider = coder.name
    history.append({"role": "user", "content": gen_prompt})
    history.append({"role": "assistant", "content": raw})

    while True:
        print(format_code_block(code, title=f"{goal[:60]}"))

        action = ask_action()

        if action == "e":
            print("Paste replacement code, then a line with just EOF:")
            lines = []
            while True:
                line = input()
                if line.strip() == "EOF":
                    break
                lines.append(line)
            new_code = "\n".join(lines)
            print("\nDiff vs. previous version:")
            show_diff(code, new_code)
            code = new_code
            continue

        if action == "f":
            feedback = input("What should change? ").strip()
            fix_prompt = f"The current code for goal '{goal}' needs this change:\n{feedback}\n\nCurrent code:\n{code}\n\nOutput only the corrected full code."
            raw = stream_generate(coder, fix_prompt, max_tokens=1024, history=history, label="Applying feedback", render="code", stats_store=stats_store, capability="coding", cost_tracker=cost_tracker)
            new_code = extract_code(raw)
            print("Diff vs. previous version:")
            show_diff(code, new_code)
            history.append({"role": "user", "content": fix_prompt})
            history.append({"role": "assistant", "content": raw})
            code = new_code
            continue

        if action == "s":
            return None, None, None, None

        if risk_profile.risk == RiskLevel.HIGH:
            confirm = input(_c(
                f"   This goal was flagged HIGH RISK. Type 'yes' to run it anyway, anything else to go back: ",
                "yellow")).strip().lower()
            if confirm != "yes":
                print("   Not running. Back to review.")
                continue

        # action == "r": run it, with auto-fix on failure
        for attempt in range(MAX_FIX_ATTEMPTS):
            workspace.save_code(ws_path, code)
            print("Executing (sandboxed)...")
            stdout, stderr, exit_code = execute_code(code)
            workspace.save_execution(ws_path, stdout, stderr, exit_code)

            if exit_code == 0:
                print(_c("Execution succeeded.", "green"))
                if stdout:
                    print(f"Output:\n{stdout}")

                passed, reason, validator_ran = validate_result(goal, code, stdout, registry, stats_store, cost_tracker)
                if validator_ran:
                    if passed:
                        print(_c(f"Validator: PASS -- {reason}", "green"))
                    else:
                        print(_c(f"Validator: FAIL -- {reason}", "red"))
                else:
                    print(_c(f"Validator: {reason}", "grey"))

                if passed or attempt == MAX_FIX_ATTEMPTS - 1:
                    return code, stdout, stderr, provider

                # Validator caught something exit_code==0 couldn't --
                # treat it like an execution failure and try again,
                # same as the exit_code != 0 branch below.
                fix_prompt = (f"The Python code for goal '{goal}' ran without crashing, but a "
                              f"reviewer found this problem: {reason}\n\nCurrent code:\n{code}\n\n"
                              "Output only the corrected code.")
                raw = stream_generate(coder, fix_prompt, max_tokens=1024, history=history,
                                       label=f"Auto-fixing after failed validation (attempt {attempt + 1})",
                                       render="code", stats_store=stats_store, capability="coding",
                                       cost_tracker=cost_tracker)
                new_code = extract_code(raw)
                show_diff(code, new_code)
                history.append({"role": "user", "content": fix_prompt})
                history.append({"role": "assistant", "content": raw})
                code = new_code
                continue

            print(_c(f"Execution failed (exit code {exit_code}).", "red"))
            if stderr:
                print(f"Error: {stderr[:300]}")
            if attempt == MAX_FIX_ATTEMPTS - 1:
                break

            fix_prompt = f"The Python code for goal '{goal}' failed:\n\n{stderr}\n\nCurrent code:\n{code}\n\nOutput only the corrected code."
            raw = stream_generate(coder, fix_prompt, max_tokens=1024, history=history, label=f"Auto-fixing (attempt {attempt + 1})", render="code", stats_store=stats_store, capability="coding", cost_tracker=cost_tracker)
            new_code = extract_code(raw)
            show_diff(code, new_code)
            history.append({"role": "user", "content": fix_prompt})
            history.append({"role": "assistant", "content": raw})
            code = new_code

        print("Max fix attempts reached for this run. Back to review.")
        # loop back to review menu instead of silently failing


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
        goal = input("\nYour goal/question (or 'exit', 'providers', 'reset', 'cost', 'quota'): ").strip()
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
                except Exception as e:
                    print(f"Provider {provider.name} failed: {e}")
                    continue
            continue

        ws_path = workspace.create_workspace(task_id)
        context = session.as_context()
        plan_prompt = f"{context}Create a short, clear, numbered plan for: {goal}"
        planner = registry.get_best("planning", stats_store=stats_store,
                                     min_context=estimate_tokens(plan_prompt) + 200)
        coder = registry.get_best("coding", stats_store=stats_store)
        if not planner or not coder:
            print("No provider configured for planning/coding (or none with enough context -- try 'reset').")
            continue

        cost_tracker = CostTracker()
        plan = stream_generate(planner, plan_prompt,
                                max_tokens=1024, label="Planning", stats_store=stats_store, capability="planning",
                                cost_tracker=cost_tracker)
        workspace.save_plan(ws_path, plan)

        code, stdout, stderr, provider = run_with_autofix_interactive(
            goal, plan, coder, workspace, registry, stats_store, task_id, cost_tracker)

        if cost_tracker.total > 0:
            print(_c(f"Task cost so far: ${cost_tracker.total:.6f} across {cost_tracker.calls} calls", "grey"))

        if code and stdout is not None:
            print("\nTask completed and saved to memory.")
            memory_db.save_task(task_id=task_id, goal=goal, plan=plan, code=code, answer=stdout,
                                 provider=provider, workspace_path=str(ws_path), success=True,
                                 cost_usd=cost_tracker.total)
            session.add(goal, "code", f"Wrote and ran code for: {goal}. Output: {stdout[:150]}")
        else:
            print("\nTask skipped or not completed.")
            memory_db.save_task(task_id=task_id, goal=goal, plan=plan, code=code or "",
                                 answer=f"INCOMPLETE: {stderr or ''}", provider=provider or "unknown",
                                 workspace_path=str(ws_path), success=False, cost_usd=cost_tracker.total)


if __name__ == "__main__":
    main()
