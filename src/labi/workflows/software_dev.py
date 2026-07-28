"""Software development workflow: Plan -> Code -> Validate.

Each pass through _agents re-snapshots the task first, so an agent that
skips itself via can_handle() (e.g. PlannerAgent after a plan already
exists) still sees up-to-date state. On any update.error, the whole pass
restarts from the first agent whose can_handle() says yes for the
current snapshot -- which, thanks to the can_handle overrides in
agents/planner.py, executor.py, validator.py, means a validation failure
retries coding + validation, not the whole pipeline from planning.
"""

from .base import BaseWorkflow
from labi.core.task import Task


class SoftwareDevelopmentWorkflow(BaseWorkflow):
    max_retries = 3

    def execute(self, task: Task) -> dict:
        conv_id = task.context_id
        task_id = task.id
        self.context_manager.add_message(conv_id, "user", task.goal)

        if self.context_manager.get_task_context(task_id) is None:
            self.context_manager.create_task_context(conv_id, task.goal, task_id=task_id)

        retries = 0
        while retries < self.max_retries:
            errored = False
            ran_any_agent = False

            for agent in self._agents:
                snapshot = self.context_manager.snapshot(task_id)
                if not agent.can_handle(snapshot):
                    continue
                ran_any_agent = True

                update = agent.process(snapshot)
                self.context_manager.apply_update(task_id, update)

                if update.completed:
                    return {
                        "status": "completed",
                        "task_id": task_id,
                        "artifacts": self.context_manager.get_task_artifacts(task_id),
                    }

                if update.error:
                    errored = True
                    break

            if not ran_any_agent:
                return {
                    "status": "failed",
                    "task_id": task_id,
                    "error": "No agent could handle this task's current state",
                }

            if errored:
                retries += 1
                continue

            # A full pass ran every applicable agent with no error and no
            # completion -- nothing will change on another identical
            # pass, so stop instead of spinning until max_retries.
            break

        final_snapshot = self.context_manager.snapshot(task_id)
        return {
            "status": "failed",
            "task_id": task_id,
            "error": final_snapshot.error or "Max retries exceeded",
        }
