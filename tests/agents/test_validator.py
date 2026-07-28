from labi.agents.validator import ValidatorAgent
from labi.context.artifact import ArtifactStore, ArtifactType
from labi.context.prompt_builder import PromptBuilder
from labi.context.snapshot import ContextSnapshot
from labi.providers.adaptive_registry import AdaptiveProviderRegistry
from tests.fakes import FakeProvider


def _agent_and_snapshot(response_text, artifact_content="print('hi')"):
    registry = AdaptiveProviderRegistry()
    registry.register(FakeProvider("validator-model", ["validation"], response_text))
    store = ArtifactStore()
    prompt_builder = PromptBuilder(store)
    agent = ValidatorAgent(registry, prompt_builder)

    store.store_artifact("out.py", artifact_content, ArtifactType.CODE, task_id="t1")
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="coding",
                                artifacts=store.list_by_task("t1"))
    return agent, snapshot


def test_process_marks_completed_on_pass():
    agent, snapshot = _agent_and_snapshot("PASS")
    update = agent.process(snapshot)
    assert update.completed is True
    assert update.status == "completed"


def test_process_marks_failed_status_coding_when_issues_found():
    agent, snapshot = _agent_and_snapshot("There is a bug on line 3: unhandled exception")
    update = agent.process(snapshot)
    assert update.completed is False
    assert update.status == "coding"
    assert update.error is not None


def test_process_defaults_to_pass_when_no_problem_keywords_present():
    agent, snapshot = _agent_and_snapshot("Looks good, follows best practices.")
    update = agent.process(snapshot)
    assert update.completed is True


def test_process_returns_failed_update_with_no_artifacts():
    registry = AdaptiveProviderRegistry()
    registry.register(FakeProvider("validator-model", ["validation"], "PASS"))
    prompt_builder = PromptBuilder(ArtifactStore())
    agent = ValidatorAgent(registry, prompt_builder)
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="coding", artifacts=[])

    update = agent.process(snapshot)
    assert update.status == "failed"
    assert "artifact" in update.error.lower()


def test_can_handle_requires_artifacts():
    agent, snapshot = _agent_and_snapshot("PASS")
    assert agent.can_handle(snapshot) is True
    empty_snapshot = ContextSnapshot(task_id="t1", goal="goal", status="coding", artifacts=[])
    assert agent.can_handle(empty_snapshot) is False
