from labi.agents.planner import PlannerAgent
from labi.context.artifact import ArtifactStore
from labi.context.prompt_builder import PromptBuilder
from labi.context.snapshot import ContextSnapshot
from labi.providers.adaptive_registry import AdaptiveProviderRegistry
from tests.fakes import FakeProvider


def _agent(response_text, should_fail=False):
    registry = AdaptiveProviderRegistry()
    registry.register(FakeProvider("planner-model", ["planning"], response_text, should_fail=should_fail))
    prompt_builder = PromptBuilder(ArtifactStore())
    return PlannerAgent(registry, prompt_builder)


def test_process_parses_numbered_steps():
    agent = _agent("1. Set up the project\n2. Write the function\n3. Add tests")
    snapshot = ContextSnapshot(task_id="t1", goal="build a calculator", status="pending")

    update = agent.process(snapshot)

    assert update.status == "planning"
    assert update.plan == ["Set up the project", "Write the function", "Add tests"]
    assert update.current_step == "Set up the project"


def test_process_parses_bullet_steps():
    agent = _agent("- First do this\n- Then do that")
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="pending")

    update = agent.process(snapshot)
    assert update.plan == ["First do this", "Then do that"]


def test_process_falls_back_to_whole_text_as_single_step():
    agent = _agent("Just write a hello world script.")
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="pending")

    update = agent.process(snapshot)
    assert update.plan == ["Just write a hello world script."]


def test_process_returns_failed_update_when_no_provider_available():
    registry = AdaptiveProviderRegistry()  # empty -- nothing registered
    prompt_builder = PromptBuilder(ArtifactStore())
    agent = PlannerAgent(registry, prompt_builder)
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="pending")

    update = agent.process(snapshot)
    assert update.status == "failed"
    assert update.error is not None


def test_process_returns_failed_update_when_provider_raises():
    agent = _agent("", should_fail=True)
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="pending")

    update = agent.process(snapshot)
    assert update.status == "failed"
    assert "failed" in update.error.lower() or "down" in update.error.lower()


def test_process_falls_through_to_second_provider_and_records_first_failure(tmp_path):
    from labi.providers.stats import ProviderStatsStore

    stats = ProviderStatsStore(str(tmp_path / "stats.db"))
    registry = AdaptiveProviderRegistry()
    # Same capability, different priority -- flaky is tried first (lower
    # priority number), backup is the one that actually succeeds.
    registry.register(FakeProvider("flaky", ["planning"], "unused", priority=10, should_fail=True))
    registry.register(FakeProvider("backup", ["planning"], "1. Step one\n2. Step two", priority=20))
    prompt_builder = PromptBuilder(ArtifactStore())
    agent = PlannerAgent(registry, prompt_builder, stats_store=stats)
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="pending")

    update = agent.process(snapshot)

    assert update.status == "planning"
    assert update.plan == ["Step one", "Step two"]
    assert agent.last_provider == "backup"

    recorded = stats.get_recent_failures(provider="flaky")
    assert len(recorded) == 1
    assert recorded[0]["capability"] == "planning"


def test_can_handle_false_once_plan_already_exists():
    agent = _agent("1. Step")
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="coding", plan=["Step"])
    assert agent.can_handle(snapshot) is False


def test_can_handle_true_when_no_plan_yet():
    agent = _agent("1. Step")
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="pending")
    assert agent.can_handle(snapshot) is True
