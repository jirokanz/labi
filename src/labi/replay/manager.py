import uuid


class ReplayManager:
    def __init__(self, workspace_manager, memory_db, logger):
        self.workspace = workspace_manager
        self.memory = memory_db
        self.logger = logger

    def create_replay_from_task(self, source_task_id, trigger_task_id=None):
        original = self.memory.get_task(source_task_id)
        if not original:
            self.logger.error(f"Task {source_task_id} not found")
            return None

        new_id = f"task_{uuid.uuid4().hex[:8]}"
        new_goal = f"Replay of: {original.get('goal', 'unknown')}"
        new_depth = (original.get("replay_depth") or 0) + 1

        try:
            new_workspace = self.workspace.clone_task_workspace(source_task_id, new_id, new_goal)
        except Exception as e:
            self.logger.error(f"Clone failed: {e}")
            return None

        self.memory.register_replay_attempt(source_task_id)
        self.memory.record_task(
            new_id, new_goal,
            success=False,
            workspace_path=str(new_workspace),
            replay_depth=new_depth,
        )
        self.memory.update_task_metadata(new_id, {"task_type": "replay", "parent_task_id": source_task_id})

        return {"new_task_id": new_id, "workspace_path": str(new_workspace), "replay_depth": new_depth}
