"""ExecutionPlan validator.""" 

from labi.decision.types import ExecutionAction, ExecutionPlan

class ExecutionPlanError(Exception):
    pass

def validate_plan(plan: ExecutionPlan) -> bool:
    if plan.action == ExecutionAction.EXECUTE_MEMORY:
        if plan.memory_id is None:
            raise ExecutionPlanError("EXECUTE_MEMORY requires memory_id.")
        if plan.requires_planning or plan.requires_coding:
            raise ExecutionPlanError("EXECUTE_MEMORY should not require planning/coding.")
    elif plan.action == ExecutionAction.ADAPT_MEMORY:
        if plan.memory_id is None:
            raise ExecutionPlanError("ADAPT_MEMORY requires memory_id.")
        if not plan.requires_coding or not plan.requires_validation:
            raise ExecutionPlanError("ADAPT_MEMORY requires coding and validation.")
    elif plan.action == ExecutionAction.FULL_GENERATION:
        if plan.memory_id is not None:
            raise ExecutionPlanError("FULL_GENERATION should have no memory_id.")
        if not (plan.requires_planning and plan.requires_coding and plan.requires_validation):
            raise ExecutionPlanError("FULL_GENERATION requires planning, coding, validation.")
    else:
        raise ExecutionPlanError(f"Unknown action: {plan.action}")
    return True
