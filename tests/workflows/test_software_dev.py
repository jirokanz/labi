from labi.agents.executor import ExecutorAgent
from labi.agents.planner import PlannerAgent
from labi.agents.validator import ValidatorAgent
from labi.context.artifact import ArtifactStore
from labi.context.manager import ContextManager
from labi.context.prompt_builder import PromptBuilder
from labi.core.task import Task
from labi.providers.adaptive_registry import AdaptiveProviderRegistry
from labi.providers.stats import ProviderStatsStore
from labi.workflows.software_dev import SoftwareDevelopmentWorkflow
from tests.fakes import FakeProvider


def _build_workflow(registry, stats_store=None):
    store = ArtifactStore()
    prompt_builder = PromptBuilder(store)
    context_manager = ContextManager(artifact_store=store)
    planner = PlannerAgent(registry, prompt_builder, stats_store=stats_store)
    executor = ExecutorAgent(registry, prompt_builder, stats_store=stats_store)
    validator = ValidatorAgent(registry, prompt_builder, stats_store=stats_store)
    workflow = SoftwareDevelopmentWorkflow(context_manager)
    workflow.add_agent(planner).add_agent(executor).add_agent(validator)
    return workflow, context_manager


def test_happy_path_plan_code_validate_completes(tmp_path):
    registry = AdaptiveProviderRegistry()
    registry.register(FakeProvider("planner-model", ["planning"], "1. Write hello world script"))
    registry.register(FakeProvider("coder-model", ["coding"], "```python\nprint('hello world')\n```"))
    registry.register(FakeProvider("validator-model", ["validation"], "PASS"))
    stats_store = ProviderStatsStore(str(tmp_path / "stats.db"))

    workflow, context_manager = _build_workflow(registry, stats_store)
    conv_id = context_manager.create_conversation()
    task = Task(id="task_1", goal="write a hello world script", context_id=conv_id)

    result = workflow.execute(task)

    assert result["status"] == "completed"
    assert len(result["artifacts"]) == 1
    assert "hello world" in result["artifacts"][0].content


def test_validation_failure_retries_coding_not_planning(tmp_path):
    """The whole point of the can_handle overrides: a validation failure
    should re-run the executor (with a fresh code attempt) and validator
    again, without paying for a second planning call."""
    registry = AdaptiveProviderRegistry()
    plan_calls = {"count": 0}

    class CountingPlannerProvider(FakeProvider):
        def generate_stream(self, *a, **kw):
            plan_calls["count"] += 1
            return super().generate_stream(*a, **kw)

    registry.register(CountingPlannerProvider("planner-model", ["planning"], "1. Write the script"))
    registry.register(FakeProvider("coder-model", ["coding"], "```python\nprint('x')\n```"))

    responses = iter(["This has a bug", "PASS"])

    class SequencedValidatorProvider(FakeProvider):
        def generate_stream(self, *a, **kw):
            yield next(responses)

    registry.register(SequencedValidatorProvider("validator-model", ["validation"], ""))

    workflow, context_manager = _build_workflow(registry)
    conv_id = context_manager.create_conversation()
    task = Task(id="task_1", goal="write a script", context_id=conv_id)

    result = workflow.execute(task)

    assert result["status"] == "completed"
    assert plan_calls["count"] == 1  # planner only ran once despite the retry


def test_persistent_validation_failure_exhausts_retries(tmp_path):
    registry = AdaptiveProviderRegistry()
    registry.register(FakeProvider("planner-model", ["planning"], "1. Write the script"))
    registry.register(FakeProvider("coder-model", ["coding"], "```python\nprint('x')\n```"))
    registry.register(FakeProvider("validator-model", ["validation"], "This always has an error"))

    workflow, context_manager = _build_workflow(registry)
    conv_id = context_manager.create_conversation()
    task = Task(id="task_1", goal="write a script", context_id=conv_id)

    result = workflow.execute(task)

    assert result["status"] == "failed"
    assert "error" in result["error"].lower()


def test_missing_capability_fails_cleanly_without_a_provider():
    registry = AdaptiveProviderRegistry()  # nothing registered at all
    workflow, context_manager = _build_workflow(registry)
    conv_id = context_manager.create_conversation()
    task = Task(id="task_1", goal="write a script", context_id=conv_id)

    result = workflow.execute(task)

    assert result["status"] == "failed"
    assert result["task_id"] == "task_1"


def test_planner_fallback_to_second_provider_on_first_failure():
    registry = AdaptiveProviderRegistry()
    registry.register(FakeProvider("flaky-planner", ["planning"], "", priority=5, should_fail=True))
    registry.register(FakeProvider("reliable-planner", ["planning"], "1. Write the script", priority=20))
    registry.register(FakeProvider("coder-model", ["coding"], "```python\nprint('x')\n```"))
    registry.register(FakeProvider("validator-model", ["validation"], "PASS"))

    workflow, context_manager = _build_workflow(registry)
    conv_id = context_manager.create_conversation()
    task = Task(id="task_1", goal="write a script", context_id=conv_id)

    result = workflow.execute(task)
    assert result["status"] == "completed"
