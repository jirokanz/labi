"""Mock executors for Phase 2.4.7 integration testing.""" 

from abc import ABC, abstractmethod
from typing import Any, Dict

from labi.decision.types import ExecutionPlan

class BaseExecutor(ABC):
    @abstractmethod
    def execute(self, plan: ExecutionPlan) -> Dict[str, Any]:
        pass

class MockMemoryExecutor(BaseExecutor):
    def execute(self, plan: ExecutionPlan) -> Dict[str, Any]:
        return {"status": "success", "action": "memory_execution", "memory_id": plan.memory_id, "output": f"Simulated execution of {plan.memory_id}"}

class MockAdaptationExecutor(BaseExecutor):
    def execute(self, plan: ExecutionPlan) -> Dict[str, Any]:
        return {"status": "success", "action": "adaptation", "memory_id": plan.memory_id, "provider": plan.provider, "output": "Simulated adaptation"}

class MockGenerationExecutor(BaseExecutor):
    def execute(self, plan: ExecutionPlan) -> Dict[str, Any]:
        return {"status": "success", "action": "full_generation", "provider": plan.provider, "output": "Simulated full generation"}
