"""Immutable-in-practice snapshot of one task's context, handed to an
agent's process() so it never mutates shared state directly -- an agent
returns a TaskUpdate describing what it wants changed, and
ContextManager.apply_update() is the only thing that actually writes."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .artifact import Artifact


@dataclass
class ContextSnapshot:
    task_id: str
    goal: str
    status: str
    plan: Optional[List[str]] = None
    current_step: Optional[str] = None
    previous_output: Optional[str] = None
    error: Optional[str] = None
    artifacts: List[Artifact] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)
