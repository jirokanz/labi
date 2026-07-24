"""ExecutionDispatcher - Routes plans to executors.""" 

from typing import Dict, Any

from labi.decision.types import ExecutionAction, ExecutionPlan
from labi.execution.executors import BaseExecutor, MockMemoryExecutor, MockAdaptationExecutor, MockGenerationExecutor

class ExecutionDispatcher:
    def __init__(self, memory_executor: BaseExecutor = None, adapt_executor: BaseExecutor = None, gen_executor: BaseExecutor = None):
        self.memory_executor = memory_executor or MockMemoryExecutor()
        self.adapt_executor = adapt_executor or MockAdaptationExecutor()
        self.gen_executor = gen_executor or MockGenerationExecutor()

    def execute(self, plan: ExecutionPlan) -> Dict[str, Any]:
        if plan.action == ExecutionAction.EXECUTE_MEMORY:
            return self.memory_executor.execute(plan)
        if plan.action == ExecutionAction.ADAPT_MEMORY:
            return self.adapt_executor.execute(plan)
        if plan.action == ExecutionAction.FULL_GENERATION:
            return self.gen_executor.execute(plan)
        raise RuntimeError(f"Unknown action: {plan.action}")
