"""TaskUpdate -- the set of changes an agent proposes after processing a
ContextSnapshot. ContextManager.apply_update() is what actually writes
these onto the task's stored state; the agent itself never mutates
anything directly (see snapshot.py's docstring)."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TaskUpdate:
    plan: Optional[List[str]] = None
    current_step: Optional[str] = None
    next_step: Optional[str] = None
    status: Optional[str] = None
    previous_output: Optional[str] = None
    error: Optional[str] = None
    artifact_ids: Optional[List[str]] = None
    completed: bool = False
