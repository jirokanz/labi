import json
import os
import shutil
from datetime import datetime
from pathlib import Path


class WorkspaceManager:
    def __init__(self, root="/tmp/workspace"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, task_id):
        return self.root / task_id

    def create_workspace(self, task_id):
        path = self._path_for(task_id)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def save_code(self, task_id, code):
        path = self._path_for(task_id)
        path.mkdir(parents=True, exist_ok=True)
        code_path = path / "code.py"
        code_path.write_text(code)
        return str(code_path)

    def save_plan(self, task_id, plan):
        path = self._path_for(task_id)
        path.mkdir(parents=True, exist_ok=True)
        plan_path = path / "plan.txt"
        plan_path.write_text(plan)
        return str(plan_path)

    def save_execution(self, task_id, stdout, stderr, exit_code):
        path = self._path_for(task_id)
        path.mkdir(parents=True, exist_ok=True)
        exec_path = path / "execution.json"
        exec_path.write_text(json.dumps({
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "timestamp": datetime.now().isoformat(),
        }, indent=2))
        return str(exec_path)

    def clone_task_workspace(self, source_task_id, new_task_id, new_goal=None):
        """
        Clone a previous task's workspace (its code.py, plan.txt, etc.) into
        a new workspace directory for a replay attempt. This was called by
        ReplayManager but never existed on this class — replays crashed
        with AttributeError before this was added.
        """
        source_path = self._path_for(source_task_id)
        new_path = self._path_for(new_task_id)
        if not source_path.exists():
            # Nothing to clone from; just hand back a fresh empty workspace
            # so callers can still proceed (e.g. regenerate from scratch).
            new_path.mkdir(parents=True, exist_ok=True)
            return str(new_path)

        shutil.copytree(source_path, new_path, dirs_exist_ok=True)
        # Stale execution result from the source task shouldn't be mistaken
        # for the replay's own result.
        exec_file = new_path / "execution.json"
        if exec_file.exists():
            exec_file.unlink()
        if new_goal is not None:
            (new_path / "goal.txt").write_text(new_goal)
        return str(new_path)
