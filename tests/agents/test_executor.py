from labi.agents.executor import ExecutorAgent
from labi.context.artifact import ArtifactStore
from labi.context.prompt_builder import PromptBuilder
from labi.context.snapshot import ContextSnapshot
from labi.providers.adaptive_registry import AdaptiveProviderRegistry
from tests.fakes import FakeProvider


def _agent(response_text, should_fail=False, confirm_high_risk_fn=None):
    registry = AdaptiveProviderRegistry()
    registry.register(FakeProvider("coder-model", ["coding"], response_text, should_fail=should_fail))
    store = ArtifactStore()
    prompt_builder = PromptBuilder(store)
    return ExecutorAgent(registry, prompt_builder, confirm_high_risk_fn=confirm_high_risk_fn), store


def _snapshot(**overrides):
    base = dict(task_id="t1", goal="write a hello world script", status="planning",
                plan=["Write the script"], current_step="Write the script")
    base.update(overrides)
    return ContextSnapshot(**base)


def test_process_actually_executes_the_code_not_just_stores_it():
    agent, store = _agent("```python\nprint('hello world')\n```")
    update = agent.process(_snapshot())

    assert update.execution_exit_code == 0
    assert "hello world" in update.execution_stdout
    assert update.status == "coding"
    assert update.error is None
    artifact = store.get(update.artifact_ids[0])
    assert artifact.content.strip() == "print('hello world')"


def test_process_reports_execution_failure_as_error_for_retry():
    agent, store = _agent("```python\nraise ValueError('boom')\n```")
    update = agent.process(_snapshot())

    assert update.execution_exit_code != 0
    assert update.error is not None
    assert "boom" in update.execution_stderr
    assert update.status == "coding"  # not "failed" -- workflow should retry, not give up
    # Code is still stored even though it crashed, so the retry prompt can
    # reference the previous attempt.
    assert store.get(update.artifact_ids[0]) is not None


def test_process_returns_failed_update_with_no_plan():
    agent, _ = _agent("irrelevant")
    update = agent.process(_snapshot(plan=None, current_step=None))
    assert update.status == "failed"
    assert "plan" in update.error.lower()


def test_process_returns_failed_update_when_no_provider_available():
    registry = AdaptiveProviderRegistry()
    prompt_builder = PromptBuilder(ArtifactStore())
    agent = ExecutorAgent(registry, prompt_builder)
    update = agent.process(_snapshot())
    assert update.status == "failed"
    assert update.error is not None


def test_high_risk_goal_asks_for_confirmation_before_running():
    calls = []

    def confirm_fn(goal, keywords):
        calls.append((goal, keywords))
        return True  # approve

    agent, _ = _agent("```python\nprint('ok')\n```", confirm_high_risk_fn=confirm_fn)
    update = agent.process(_snapshot(goal="delete everything in the database"))

    assert len(calls) == 1
    assert calls[0][0] == "delete everything in the database"
    assert "delete" in calls[0][1] or "database" in calls[0][1]
    assert update.execution_exit_code == 0  # ran, since confirm_fn approved


def test_declining_high_risk_confirmation_prevents_execution():
    agent, store = _agent("```python\nprint('should not run')\n```", confirm_high_risk_fn=lambda g, k: False)
    update = agent.process(_snapshot(goal="delete everything in the database"))

    assert update.status == "failed"
    assert "declined" in update.error.lower() or "not executed" in update.error.lower()
    # No artifact should be stored for code that was never approved to run.
    assert update.artifact_ids is None


def test_low_risk_goal_never_calls_confirmation_function():
    calls = []
    agent, _ = _agent("```python\nprint('fine')\n```", confirm_high_risk_fn=lambda g, k: calls.append(1) or True)
    agent.process(_snapshot(goal="write a hello world script"))
    assert calls == []


def test_can_handle_requires_a_plan():
    agent, _ = _agent("irrelevant")
    assert agent.can_handle(_snapshot(plan=None, current_step=None)) is False
    assert agent.can_handle(_snapshot(plan=["a"])) is True


def test_get_next_step_advances_to_next_plan_item():
    agent, _ = _agent("```python\nprint('x')\n```")
    update = agent.process(_snapshot(plan=["step 1", "step 2"], current_step="step 1"))
    assert update.next_step == "step 2"


def test_get_next_step_returns_validate_after_last_step():
    agent, _ = _agent("```python\nprint('x')\n```")
    update = agent.process(_snapshot(plan=["only step"], current_step="only step"))
    assert update.next_step == "validate"
