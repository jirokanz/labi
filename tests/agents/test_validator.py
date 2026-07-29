from labi.agents.validator import ValidatorAgent
from labi.context.artifact import ArtifactStore, ArtifactType
from labi.context.prompt_builder import PromptBuilder
from labi.context.snapshot import ContextSnapshot
from labi.providers.adaptive_registry import AdaptiveProviderRegistry
from tests.fakes import FakeProvider


def _agent_and_snapshot(response_text, stdout="hello world\n", exit_code=0, artifact_content="print('hello world')"):
    registry = AdaptiveProviderRegistry()
    registry.register(FakeProvider("validator-model", ["validation"], response_text))
    store = ArtifactStore()
    prompt_builder = PromptBuilder(store)
    agent = ValidatorAgent(registry, prompt_builder)

    store.store_artifact("out.py", artifact_content, ArtifactType.CODE, task_id="t1")
    snapshot = ContextSnapshot(task_id="t1", goal="print hello world", status="coding",
                                artifacts=store.list_by_task("t1"),
                                execution_stdout=stdout, execution_exit_code=exit_code)
    return agent, snapshot


def test_process_marks_completed_on_pass():
    agent, snapshot = _agent_and_snapshot("PASS")
    update = agent.process(snapshot)
    assert update.completed is True
    assert update.status == "completed"


def test_process_marks_failed_status_coding_on_fail_with_reason():
    agent, snapshot = _agent_and_snapshot("FAIL: output does not print hello world", stdout="goodbye\n")
    update = agent.process(snapshot)
    assert update.completed is False
    assert update.status == "coding"
    assert "does not print hello world" in update.error


def test_process_degrades_to_unverified_pass_when_no_provider_available():
    """Matches the old validate_result()'s 'ran=False -> unverified, not
    failed' behavior -- missing validation infra shouldn't block a task
    that otherwise ran successfully."""
    registry = AdaptiveProviderRegistry()  # no validation provider registered
    store = ArtifactStore()
    prompt_builder = PromptBuilder(store)
    agent = ValidatorAgent(registry, prompt_builder)
    store.store_artifact("out.py", "print('hi')", ArtifactType.CODE, task_id="t1")
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="coding",
                                artifacts=store.list_by_task("t1"),
                                execution_stdout="hi\n", execution_exit_code=0)

    update = agent.process(snapshot)
    assert update.completed is True
    assert "unverified" in update.previous_output.lower()


def test_process_returns_failed_update_with_no_artifacts():
    registry = AdaptiveProviderRegistry()
    registry.register(FakeProvider("validator-model", ["validation"], "PASS"))
    prompt_builder = PromptBuilder(ArtifactStore())
    agent = ValidatorAgent(registry, prompt_builder)
    snapshot = ContextSnapshot(task_id="t1", goal="goal", status="coding", artifacts=[],
                                execution_stdout="x", execution_exit_code=0)

    update = agent.process(snapshot)
    assert update.status == "failed"
    assert "artifact" in update.error.lower()


def test_can_handle_requires_artifacts_and_successful_execution():
    agent, snapshot = _agent_and_snapshot("PASS")
    assert agent.can_handle(snapshot) is True

    no_artifacts = ContextSnapshot(task_id="t1", goal="goal", status="coding", artifacts=[],
                                    execution_stdout="x", execution_exit_code=0)
    assert agent.can_handle(no_artifacts) is False

    not_yet_executed = ContextSnapshot(task_id="t1", goal="goal", status="coding",
                                        artifacts=snapshot.artifacts, execution_exit_code=None)
    assert agent.can_handle(not_yet_executed) is False

    crashed = ContextSnapshot(task_id="t1", goal="goal", status="coding",
                               artifacts=snapshot.artifacts, execution_exit_code=1)
    assert agent.can_handle(crashed) is False
