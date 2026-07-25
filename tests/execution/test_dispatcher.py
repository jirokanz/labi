"""Test ExecutionDispatcher.""" 

from labi.execution.dispatcher import ExecutionDispatcher
from labi.decision.types import ExecutionAction, ExecutionPlan

def test_dispatcher_memory():
    plan = ExecutionPlan(ExecutionAction.EXECUTE_MEMORY, None, "mem", False, False, False, "x")
    assert ExecutionDispatcher().execute(plan)["action"] == "memory_execution"

def test_dispatcher_adapt():
    plan = ExecutionPlan(ExecutionAction.ADAPT_MEMORY, "p", "mem", False, True, True, "x")
    assert ExecutionDispatcher().execute(plan)["action"] == "adaptation"

def test_dispatcher_gen():
    plan = ExecutionPlan(ExecutionAction.FULL_GENERATION, "p", None, True, True, True, "x")
    assert ExecutionDispatcher().execute(plan)["action"] == "full_generation"
