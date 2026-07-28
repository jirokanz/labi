import pytest

from labi.context.manager import ContextManager
from labi.context.update import TaskUpdate
from labi.context.artifact import ArtifactStore, ArtifactType


def test_create_task_context_and_snapshot_reflect_goal():
    cm = ContextManager()
    conv_id = cm.create_conversation()
    task_id = cm.create_task_context(conv_id, "write a hello world script")

    snap = cm.snapshot(task_id)
    assert snap.task_id == task_id
    assert snap.goal == "write a hello world script"
    assert snap.status == "pending"
    assert snap.plan is None
    assert snap.artifacts == []


def test_create_task_context_with_explicit_task_id():
    cm = ContextManager()
    conv_id = cm.create_conversation()
    task_id = cm.create_task_context(conv_id, "goal", task_id="task_explicit")
    assert task_id == "task_explicit"
    assert cm.snapshot("task_explicit").goal == "goal"


def test_snapshot_unknown_task_raises():
    cm = ContextManager()
    with pytest.raises(ValueError):
        cm.snapshot("nonexistent")


def test_apply_update_writes_plan_and_status():
    cm = ContextManager()
    conv_id = cm.create_conversation()
    task_id = cm.create_task_context(conv_id, "goal")

    cm.apply_update(task_id, TaskUpdate(plan=["step 1", "step 2"], status="planning", current_step="step 1"))

    snap = cm.snapshot(task_id)
    assert snap.plan == ["step 1", "step 2"]
    assert snap.status == "planning"
    assert snap.current_step == "step 1"


def test_apply_update_none_fields_do_not_overwrite_existing_values():
    cm = ContextManager()
    conv_id = cm.create_conversation()
    task_id = cm.create_task_context(conv_id, "goal")
    cm.apply_update(task_id, TaskUpdate(plan=["step 1"], status="planning"))

    # A later update that only sets status shouldn't wipe out the plan.
    cm.apply_update(task_id, TaskUpdate(status="coding"))

    snap = cm.snapshot(task_id)
    assert snap.plan == ["step 1"]
    assert snap.status == "coding"


def test_apply_update_completed_sets_status_to_completed():
    cm = ContextManager()
    conv_id = cm.create_conversation()
    task_id = cm.create_task_context(conv_id, "goal")

    cm.apply_update(task_id, TaskUpdate(status="validating", completed=True))

    snap = cm.snapshot(task_id)
    assert snap.status == "completed"


def test_apply_update_unknown_task_raises():
    cm = ContextManager()
    with pytest.raises(ValueError):
        cm.apply_update("nonexistent", TaskUpdate(status="x"))


def test_snapshot_includes_artifacts_for_task():
    store = ArtifactStore()
    cm = ContextManager(artifact_store=store)
    conv_id = cm.create_conversation()
    task_id = cm.create_task_context(conv_id, "goal")

    store.store_artifact("out.py", "print('hi')", ArtifactType.CODE, task_id=task_id, created_by="executor")

    snap = cm.snapshot(task_id)
    assert len(snap.artifacts) == 1
    assert snap.artifacts[0].content == "print('hi')"


def test_get_task_artifacts_matches_snapshot_artifacts():
    cm = ContextManager()
    conv_id = cm.create_conversation()
    task_id = cm.create_task_context(conv_id, "goal")
    cm.artifact_store.store_artifact("a.py", "x = 1", ArtifactType.CODE, task_id=task_id)

    assert cm.get_task_artifacts(task_id) == cm.snapshot(task_id).artifacts


def test_add_message_to_unknown_conversation_raises():
    cm = ContextManager()
    with pytest.raises(ValueError):
        cm.add_message("nonexistent", "user", "hello")


def test_artifact_store_list_by_task_filters_correctly():
    store = ArtifactStore()
    store.store_artifact("a.py", "content a", ArtifactType.CODE, task_id="task_1")
    store.store_artifact("b.py", "content b", ArtifactType.CODE, task_id="task_2")

    task_1_artifacts = store.list_by_task("task_1")
    assert len(task_1_artifacts) == 1
    assert task_1_artifacts[0].name == "a.py"
