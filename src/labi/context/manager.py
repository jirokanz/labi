"""ContextManager -- owns per-task mutable state and per-conversation
message history, and is the only thing that writes it. Agents only ever
see a read-only ContextSnapshot and hand back a TaskUpdate describing
what they want changed; apply_update() is where that actually lands."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from .snapshot import ContextSnapshot
from .update import TaskUpdate
from .artifact import ArtifactStore


@dataclass
class _TaskContext:
    """Internal, mutable per-task state. Not exposed directly -- callers
    only ever see a ContextSnapshot of it via ContextManager.snapshot()."""
    task_id: str
    goal: str
    status: str = "pending"
    plan: Optional[List[str]] = None
    current_step: Optional[str] = None
    next_step: Optional[str] = None
    previous_output: Optional[str] = None
    error: Optional[str] = None
    execution_stdout: Optional[str] = None
    execution_stderr: Optional[str] = None
    execution_exit_code: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


@dataclass
class _Conversation:
    conv_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)


# TaskUpdate fields that map directly onto a _TaskContext attribute of the
# same name -- the whitelist of what apply_update() is allowed to write.
# (artifact_ids and completed are handled separately below: artifact_ids
# don't have a _TaskContext slot, and completed drives status instead of
# overwriting a field 1:1.)
_UPDATABLE_FIELDS = (
    "plan", "current_step", "next_step", "status", "previous_output", "error",
    "execution_stdout", "execution_stderr", "execution_exit_code",
)


class ContextManager:
    def __init__(self, artifact_store: Optional[ArtifactStore] = None):
        self.artifact_store = artifact_store or ArtifactStore()
        self._task_contexts: Dict[str, _TaskContext] = {}
        self._conversations: Dict[str, _Conversation] = {}

    def create_conversation(self) -> str:
        conv_id = str(uuid.uuid4())[:8]
        self._conversations[conv_id] = _Conversation(conv_id=conv_id)
        return conv_id

    def add_message(self, conv_id: str, role: str, content: str) -> None:
        conv = self._conversations.get(conv_id)
        if conv is None:
            raise ValueError(f"Conversation {conv_id} not found")
        conv.messages.append({"role": role, "content": content})

    def create_task_context(self, conv_id: str, goal: str, task_id: Optional[str] = None) -> str:
        """Registers a new task under a conversation. task_id can be
        supplied (e.g. from an existing core.task.Task.id) so the Task
        and its ContextManager-side state share one id; otherwise one is
        generated here."""
        task_id = task_id or str(uuid.uuid4())[:8]
        self._task_contexts[task_id] = _TaskContext(task_id=task_id, goal=goal)
        conv = self._conversations.get(conv_id)
        if conv is not None:
            conv.facts.setdefault("task_ids", []).append(task_id)
        return task_id

    def get_task_context(self, task_id: str) -> Optional[_TaskContext]:
        return self._task_contexts.get(task_id)

    def snapshot(self, task_id: str) -> ContextSnapshot:
        task = self._task_contexts.get(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        return ContextSnapshot(
            task_id=task_id,
            goal=task.goal,
            status=task.status,
            plan=task.plan,
            current_step=task.current_step,
            previous_output=task.previous_output,
            error=task.error,
            artifacts=self.artifact_store.list_by_task(task_id),
            facts={},
            execution_stdout=task.execution_stdout,
            execution_stderr=task.execution_stderr,
            execution_exit_code=task.execution_exit_code,
        )

    def apply_update(self, task_id: str, update: TaskUpdate) -> None:
        task = self._task_contexts.get(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        for field_name in _UPDATABLE_FIELDS:
            value = getattr(update, field_name)
            if value is not None:
                setattr(task, field_name, value)
        task.updated_at = datetime.now(timezone.utc)
        if update.completed:
            task.status = "completed"
            task.completed_at = datetime.now(timezone.utc)

    def get_task_artifacts(self, task_id: str) -> List:
        return self.artifact_store.list_by_task(task_id)
