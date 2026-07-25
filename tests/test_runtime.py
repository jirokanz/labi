import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labi.runtime import main
from labi.tools.python.security import validate_code
from labi.tools.python.limits import make_preexec_fn, enforce_output_limit
from labi.workspace.manager import WorkspaceManager
from labi.memory.database import MemoryDatabase
from labi.memory.guard import ReplayGuard
from labi.replay.manager import ReplayManager
from labi.core.logger import Logger


def test_import():
    assert main is not None


# ---- security.validate_code ----

def test_validate_code_allows_safe_code():
    assert validate_code("x = 1 + 1\nprint(x)") == []


def test_validate_code_blocks_direct_import():
    violations = validate_code("import os\nos.system('ls')")
    assert any("os" in v for v in violations)


def test_validate_code_blocks_dynamic_import_bypass():
    # The old blocklist only checked ast.Import/ImportFrom nodes and
    # missed this entirely.
    violations = validate_code("m = __import__('os')\nm.system('ls')")
    assert violations, "dynamic __import__ bypass should be caught"


def test_validate_code_blocks_eval_exec():
    assert validate_code("eval('1+1')") != []
    assert validate_code("exec('print(1)')") != []


def test_validate_code_blocks_dunder_escape():
    violations = validate_code("().__class__.__base__.__subclasses__()")
    assert violations


def test_validate_code_syntax_error():
    violations = validate_code("def f(:\n  pass")
    assert violations and "Syntax error" in violations[0]


# ---- limits.py ----

def test_enforce_output_limit_truncates():
    out = enforce_output_limit("a" * 100, 10)
    assert out.endswith("[TRUNCATED]")
    assert len(out.encode()) < 200


def test_enforce_output_limit_passthrough():
    assert enforce_output_limit("short", 100) == "short"


def test_make_preexec_fn_returns_callable_or_none():
    fn = make_preexec_fn()
    assert fn is None or callable(fn)


# ---- workspace clone (previously called a method that didn't exist) ----

def test_workspace_clone_task_workspace(tmp_path):
    ws = WorkspaceManager(root=str(tmp_path))
    ws.create_workspace("task_a")
    ws.save_code("task_a", "print('hi')")
    cloned = ws.clone_task_workspace("task_a", "task_b", new_goal="do it again")
    assert (Path(cloned) / "code.py").exists()
    assert (Path(cloned) / "goal.txt").read_text() == "do it again"


def test_workspace_clone_missing_source_does_not_crash(tmp_path):
    ws = WorkspaceManager(root=str(tmp_path))
    cloned = ws.clone_task_workspace("nonexistent", "task_new")
    assert Path(cloned).exists()


# ---- replay depth guard (previously never incremented anywhere) ----

def test_replay_depth_increments(tmp_path):
    # Before the fix, replay_depth was checked by ReplayGuard but never
    # written anywhere, so it stayed 0 forever. Confirm it now actually
    # accumulates across a chain of replays.
    db = MemoryDatabase(":memory:")
    ws = WorkspaceManager(root=str(tmp_path))
    logger = Logger()
    replay_mgr = ReplayManager(ws, db, logger)

    db.record_task("task_0", "do a thing", success=True)
    ws.create_workspace("task_0")

    current_id = "task_0"
    for _ in range(3):
        result = replay_mgr.create_replay_from_task(current_id)
        assert result is not None
        current_id = result["new_task_id"]

    final_task = db.get_task(current_id)
    assert final_task["replay_depth"] == 3


def test_replay_guard_blocks_at_max_depth():
    guard = ReplayGuard(max_depth=2)
    deep_candidate = {"success": True, "archived": False, "replay_depth": 2}
    result = guard.check(deep_candidate)
    assert result["allowed"] is False
    assert "replay_depth_exceeded" in result["reason"]


def test_replay_guard_allows_within_depth():
    guard = ReplayGuard(max_depth=2)
    shallow_candidate = {"success": True, "archived": False, "replay_depth": 1}
    result = guard.check(shallow_candidate)
    assert result["allowed"] is True


# ---- config.py (previously always empty regardless of config_path) ----

def test_config_loads_yaml(tmp_path):
    from labi.core.config import Config
    config_file = tmp_path / "config.yaml"
    config_file.write_text("memory:\n  thresholds:\n    reuse: 0.9\n")
    cfg = Config(str(config_file))
    assert cfg.get("memory.thresholds.reuse") == 0.9
    assert cfg.get("memory.thresholds.adapt", 0.5) == 0.5


def test_config_missing_file_falls_back_to_default():
    from labi.core.config import Config
    cfg = Config("/nonexistent/path.yaml")
    assert cfg.get("anything", "fallback") == "fallback"


# ---- adaptive provider ranking (was static priority only, never updated) ----

def test_provider_stats_recorded_and_retrieved(tmp_path):
    from labi.providers.stats import ProviderStatsStore
    db = ProviderStatsStore(str(tmp_path / "stats.db"))
    db.record_provider_call("groq", "coding", True, 500)
    db.record_provider_call("groq", "coding", True, 700)
    db.record_provider_call("groq", "coding", False, 900)
    stats = db.get_provider_stats("groq", "coding")
    assert stats["calls"] == 3
    assert stats["successes"] == 2
    assert stats["total_latency_ms"] == 2100


def test_registry_falls_back_to_static_priority_below_min_samples(tmp_path):
    from labi.providers.stats import ProviderStatsStore
    from labi.agent import BaseProvider, ProviderRegistry
    db = ProviderStatsStore(str(tmp_path / "stats.db"))
    registry = ProviderRegistry()
    fast_but_new = BaseProvider("fast", "m", "", "k", ["coding"], priority=50)
    slow_but_established = BaseProvider("slow", "m", "", "k", ["coding"], priority=10)
    registry.register(fast_but_new)
    registry.register(slow_but_established)
    # Only 1 data point for "fast" -- below MIN_SAMPLES, so static priority
    # (lower number wins) should still decide the ranking.
    db.record_provider_call("fast", "coding", True, 100)
    best = registry.get_best("coding", stats_store=db)
    assert best.name == "slow"


def test_registry_prefers_measured_success_once_enough_samples(tmp_path):
    from labi.providers.stats import ProviderStatsStore
    from labi.agent import BaseProvider, ProviderRegistry
    db = ProviderStatsStore(str(tmp_path / "stats.db"))
    registry = ProviderRegistry()
    unreliable_but_prioritized = BaseProvider("unreliable", "m", "", "k", ["coding"], priority=10)
    reliable_but_deprioritized = BaseProvider("reliable", "m", "", "k", ["coding"], priority=90)
    registry.register(unreliable_but_prioritized)
    registry.register(reliable_but_deprioritized)

    for _ in range(10):
        db.record_provider_call("unreliable", "coding", False, 200)
    for _ in range(10):
        db.record_provider_call("reliable", "coding", True, 200)

    best = registry.get_best("coding", stats_store=db)
    assert best.name == "reliable"


def test_registry_with_no_memory_db_uses_static_priority():
    from labi.agent import BaseProvider, ProviderRegistry
    registry = ProviderRegistry()
    registry.register(BaseProvider("b", "m", "", "k", ["coding"], priority=50))
    registry.register(BaseProvider("a", "m", "", "k", ["coding"], priority=10))
    best = registry.get_best("coding")  # no memory_db passed
    assert best.name == "a"


# ---- session continuation detection (was misrouting follow-ups to coding) ----

def test_followup_without_action_verb_routes_to_answering():
    from labi.agent import SessionContext, is_question_or_followup
    session = SessionContext()
    session.add("what is the best for coding?", "answer", "VS Code, PyCharm, etc.")
    # This is the exact phrase that got misrouted in practice.
    assert is_question_or_followup("in term of ai token provider?", session) is True


def test_standalone_action_request_routes_to_coding():
    from labi.agent import SessionContext, is_question_or_followup
    session = SessionContext()
    assert is_question_or_followup("write a script to check disk usage", session) is False


def test_plain_question_still_detected():
    from labi.agent import SessionContext, is_question_or_followup
    session = SessionContext()
    assert is_question_or_followup("what is the capital of France?", session) is True


def test_session_context_rolls_and_resets():
    from labi.agent import SessionContext
    session = SessionContext(max_turns=2)
    session.add("goal1", "answer", "summary1")
    session.add("goal2", "answer", "summary2")
    session.add("goal3", "answer", "summary3")
    assert len(session.turns) == 2
    assert session.turns[0]["goal"] == "goal2"
    session.reset()
    assert session.turns == []
    assert session.as_context() == ""


# ---- per-capability priority override ----

def test_provider_ranks_differently_per_capability():
    from labi.agent import BaseProvider, ProviderRegistry
    registry = ProviderRegistry()
    # groq: fast generalist, best for planning, deliberately worse for coding
    groq = BaseProvider("groq", "m", "", "k", ["planning", "coding"], priority=10,
                         capability_priority={"coding": 25})
    # deepseek: code-specialized, made top pick for coding, ranked below groq for planning
    deepseek = BaseProvider("deepseek", "m", "", "k", ["coding", "planning"], priority=20,
                             capability_priority={"coding": 5})
    registry.register(groq)
    registry.register(deepseek)

    assert registry.get_best("planning").name == "groq"
    assert registry.get_best("coding").name == "deepseek"


def test_priority_for_falls_back_to_default_priority():
    from labi.agent import BaseProvider
    p = BaseProvider("x", "m", "", "k", ["a", "b"], priority=42, capability_priority={"a": 1})
    assert p.priority_for("a") == 1
    assert p.priority_for("b") == 42  # no override for "b" -- falls back to default


# ---- cost tracking ----

def test_cost_tracker_accumulates():
    from labi.agent import CostTracker
    ct = CostTracker()
    ct.add(0.0012)
    ct.add(0.0008)
    ct.add(None)  # a failed/uncosted call shouldn't crash accumulation
    assert round(ct.total, 4) == 0.002
    assert ct.calls == 3


def test_provider_stats_store_records_and_sums_cost(tmp_path):
    from labi.providers.stats import ProviderStatsStore
    db = ProviderStatsStore(str(tmp_path / "stats.db"))
    db.record_provider_call("groq", "coding", True, 500, cost_usd=0.001)
    db.record_provider_call("groq", "coding", True, 500, cost_usd=0.002)
    db.record_provider_call("deepseek", "coding", True, 500, cost_usd=0.0005)
    stats = db.get_provider_stats("groq", "coding")
    assert round(stats["total_cost_usd"], 4) == 0.003
    assert round(db.get_total_cost(), 4) == 0.0035


def test_save_task_persists_cost(tmp_path):
    from labi.agent import MemoryDB as AgentMemoryDB
    db = AgentMemoryDB(str(tmp_path / "mem.db"))
    db.save_task(task_id="t1", goal="do a thing", plan="plan", code="code",
                 answer="ok", provider="groq", success=True, cost_usd=0.0042)
    cur = db.conn.execute("SELECT cost_usd FROM tasks WHERE id=?", ("t1",))
    assert round(cur.fetchone()[0], 4) == 0.0042


# ---- providers/registry.py (was missing entirely -- the actual ImportError
# blocking the whole test suite from collecting) ----

def test_provider_registry_select_best_provider():
    from labi.providers.registry import ProviderRegistry
    reg = ProviderRegistry()
    reg.register("groq", "coding", routing_score=0.6)
    reg.register("deepseek", "coding", routing_score=0.9)
    reg.register("groq", "planning", routing_score=0.8)
    assert reg.select_best_provider("coding") == "deepseek:coding"
    assert reg.select_best_provider("planning") == "groq:planning"


def test_provider_registry_unavailable_excluded():
    from labi.providers.registry import ProviderRegistry
    reg = ProviderRegistry()
    reg.register("flaky", "coding", available=False, routing_score=0.99)
    reg.register("steady", "coding", available=True, routing_score=0.5)
    assert reg.select_best_provider("coding") == "steady:coding"


def test_provider_registry_unknown_key_degrades_gracefully():
    from labi.providers.registry import ProviderRegistry
    reg = ProviderRegistry()
    status = reg.get_provider_status("nonexistent:role")
    assert status.available is True
    assert status.routing_score == 0.5


def test_provider_registry_update_score_moves_toward_outcome():
    from labi.providers.registry import ProviderRegistry
    reg = ProviderRegistry()
    key = reg.register("groq", "coding", routing_score=0.5)
    reg.update_score(key, success=True, decay=0.5)
    assert reg.get_provider_status(key).routing_score == 0.75
    reg.update_score(key, success=False, decay=0.5)
    assert reg.get_provider_status(key).routing_score == 0.375


# ---- TaskClassifier risk gate (wired into agent.py's run action) ----

def test_classifier_flags_high_risk_goal():
    from labi.intelligence.classifier import TaskClassifier
    from labi.intelligence.types import RiskLevel
    profile = TaskClassifier().classify("delete all rows from the database")
    assert profile.risk == RiskLevel.HIGH


def test_classifier_low_risk_for_benign_goal():
    from labi.intelligence.classifier import TaskClassifier
    from labi.intelligence.types import RiskLevel
    profile = TaskClassifier().classify("reverse a string")
    assert profile.risk == RiskLevel.LOW


# ---- Live model discovery for all providers (was hardcoded names that went stale) ----

def test_pick_cerebras_model_prefers_gpt_oss_over_llama():
    # gpt-oss-120b now outranks llama-3.3 in PREFERRED_CEREBRAS_MODELS,
    # since Cerebras has since dropped Llama from its free-tier catalog
    # entirely -- "prefer llama" was the stale assumption this replaced.
    from labi.providers.adaptive_registry import pick_cerebras_model
    fake_response = {"data": [{"id": "llama-3.3-70b"}, {"id": "gpt-oss-120b"}, {"id": "zai-glm-4.7"}]}
    result = pick_cerebras_model("fake-key", fetch_fn=lambda key: fake_response)
    assert result == "gpt-oss-120b"


def test_pick_cerebras_model_falls_back_to_first_when_nothing_preferred():
    from labi.providers.adaptive_registry import pick_cerebras_model
    fake_response = {"data": [{"id": "some-brand-new-model"}, {"id": "another-one"}]}
    result = pick_cerebras_model("fake-key", fetch_fn=lambda key: fake_response)
    assert result == "some-brand-new-model"


def test_pick_cerebras_model_returns_none_on_empty_catalog():
    from labi.providers.adaptive_registry import pick_cerebras_model
    result = pick_cerebras_model("fake-key", fetch_fn=lambda key: {"data": []})
    assert result is None


def test_pick_cerebras_model_returns_none_on_fetch_failure():
    from labi.providers.adaptive_registry import pick_cerebras_model
    def broken_fetch(key):
        raise ConnectionError("simulated network failure")
    result = pick_cerebras_model("fake-key", fetch_fn=broken_fetch)
    assert result is None


def test_pick_groq_model_prefers_gpt_oss_over_deprecated_llama():
    # llama-3.3-70b-versatile (the old hardcoded model) is deprecated by
    # Groq, shutdown 08/16/2026.
    from labi.providers.adaptive_registry import pick_groq_model
    fake_response = {"data": [{"id": "llama-3.3-70b-versatile"}, {"id": "openai/gpt-oss-120b"}]}
    result = pick_groq_model("fake-key", fetch_fn=lambda key: fake_response)
    assert result == "openai/gpt-oss-120b"


def test_pick_gemini_model_prefers_3_5_flash_and_filters_non_generative():
    from labi.providers.adaptive_registry import pick_gemini_model
    fake_response = {"models": [
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]},
    ]}
    result = pick_gemini_model("fake-key", fetch_fn=lambda key: fake_response)
    assert result == "gemini-3.5-flash"


def test_pick_openrouter_model_excludes_paid_variant():
    # The old hardcoded id had no ':free' suffix, i.e. it was the paid
    # route. This must never be picked even if it's ranked first by name.
    from labi.providers.adaptive_registry import pick_openrouter_model
    fake_response = {"data": [
        {"id": "meta-llama/llama-3.1-70b-instruct", "pricing": {"prompt": "0.0000009", "completion": "0.0000009"}},
        {"id": "meta-llama/llama-3.3-70b-instruct:free", "pricing": {"prompt": "0", "completion": "0"}},
    ]}
    result = pick_openrouter_model("fake-key", fetch_fn=lambda key: fake_response)
    assert result == "meta-llama/llama-3.3-70b-instruct:free"


def test_pick_openrouter_model_returns_none_when_no_free_models():
    from labi.providers.adaptive_registry import pick_openrouter_model
    fake_response = {"data": [{"id": "some/paid-model", "pricing": {"prompt": "0.001", "completion": "0.001"}}]}
    result = pick_openrouter_model("fake-key", fetch_fn=lambda key: fake_response)
    assert result is None


# ---- daily usage / quota tracking ----

def test_record_and_get_daily_usage(tmp_path):
    from labi.providers.stats import ProviderStatsStore
    db = ProviderStatsStore(str(tmp_path / "stats.db"))
    db.record_daily_usage("groq", tokens=1500, requests=1, day="2026-07-24")
    db.record_daily_usage("groq", tokens=2000, requests=1, day="2026-07-24")
    db.record_daily_usage("groq", tokens=500, requests=1, day="2026-07-23")  # different day, separate bucket
    usage = db.get_daily_usage("groq", day="2026-07-24")
    assert usage["requests"] == 2
    assert usage["tokens"] == 3500


def test_get_all_daily_usage_scoped_to_one_day(tmp_path):
    from labi.providers.stats import ProviderStatsStore
    db = ProviderStatsStore(str(tmp_path / "stats.db"))
    db.record_daily_usage("groq", tokens=100, requests=1, day="2026-07-24")
    db.record_daily_usage("gemini", tokens=200, requests=1, day="2026-07-24")
    db.record_daily_usage("groq", tokens=999, requests=1, day="2026-07-01")
    all_today = db.get_all_daily_usage(day="2026-07-24")
    assert set(all_today.keys()) == {"groq", "gemini"}
    assert all_today["groq"]["tokens"] == 100


def test_compute_quota_status_known_provider():
    from labi.agent import compute_quota_status
    status = compute_quota_status("groq", {"requests": 300, "tokens": 50000})
    assert status["limit"] == 1000
    assert status["used"] == 300
    assert status["remaining"] == 700
    assert status["pct_used"] == 30.0


def test_compute_quota_status_unknown_provider_returns_none():
    from labi.agent import compute_quota_status
    status = compute_quota_status("some_new_provider_not_in_table", {"requests": 5, "tokens": 100})
    assert status is None


def test_compute_quota_status_over_limit_clamps_at_zero_remaining():
    from labi.agent import compute_quota_status
    status = compute_quota_status("groq", {"requests": 1200, "tokens": 999999})
    assert status["remaining"] == 0
    assert status["pct_used"] == 120.0
