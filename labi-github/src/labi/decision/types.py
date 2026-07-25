"""Shared decision types.""" 

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ExecutionAction(Enum):
    EXECUTE_MEMORY = "execute_memory"
    ADAPT_MEMORY = "adapt_memory"
    FULL_GENERATION = "full_generation"

@dataclass
class ExecutionPlan:
    action: ExecutionAction
    provider: Optional[str]
    memory_id: Optional[str]
    requires_planning: bool
    requires_coding: bool
    requires_validation: bool
    reason: str
