from labi.agents.executor import ExecutorAgent
from labi.context.artifact import ArtifactStore
from labi.context.prompt_builder import PromptBuilder
from labi.context.snapshot import ContextSnapshot
from labi.providers.adaptive_registry import AdaptiveProviderRegistry
from tests.fakes import FakeProvider


def _agent(response_text, should_fail=False):
    registry = AdaptiveProviderRegistry()
    registry.register(FakeProvider("coder-model", ["coding"], response_text, should_fail=should_fail))
    store = ArtifactStore()
    prompt_builder = PromptBuilder(store)
    return ExecutorAgent(registry, prompt_builder), store


def test_process_extracts_fenced_code_and_stores_artifact():
    agent, store = _agent("Here you go:\n```python\nprint('hello world')\n```")
    snapshot = ContextSnapshot(task_id="t1", goal="print hello world", status="planning",
                                plan=["Write the script"], current_step="Write the script")

    update = agent.process(snapshot)

    assert update.status == "coding"
    assert len(update.artifact_ids) == 1
    artifact = store.get(update.artifact_ids[0])
    assert artifact.content.strip() == "print('hello world')"
    assert artifact.task_id == "t1"
    assert artifact.created_by == "executor"


def test_process_falls_back_to_raw_text_when_no_code_fence():
    agent, store = _agent("print('no fence here')")
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="planning",
                                plan=["step"], current_step="step")

    update = agent.process(snapshot)
    artifact = store.get(update.artifact_ids[0])
    assert artifact.content == "print('no fence here')"


def test_process_returns_failed_update_with_no_plan():
    agent, _ = _agent("irrelevant")
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="pending", plan=None)

    update = agent.process(snapshot)
    assert update.status == "failed"
    assert "plan" in update.error.lower()


def test_get_next_step_advances_to_next_plan_item():
    agent, _ = _agent("```python\nx = 1\n```")
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="planning",
                                plan=["step 1", "step 2"], current_step="step 1")

    update = agent.process(snapshot)
    assert update.next_step == "step 2"


def test_get_next_step_returns_validate_after_last_step():
    agent, _ = _agent("```python\nx = 1\n```")
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="planning",
                                plan=["only step"], current_step="only step")

    update = agent.process(snapshot)
    assert update.next_step == "validate"


def test_can_handle_requires_a_plan():
    agent, _ = _agent("irrelevant")
    assert agent.can_handle(ContextSnapshot(task_id="t1", goal="g", status="pending", plan=None)) is False
    assert agent.can_handle(ContextSnapshot(task_id="t1", goal="g", status="pending", plan=["a"])) is True
