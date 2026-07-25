
class ReplayGuard:
    def __init__(self, config=None, max_depth=3):
        self.config = config
        self.max_depth = max_depth

    def check(self, candidate):
        if not candidate.get("success"):
            return {"allowed": False, "reason": "task_not_successful"}
        if candidate.get("archived"):
            return {"allowed": False, "reason": "task_archived"}
        depth = candidate.get("replay_depth", 0)
        if depth >= self.max_depth:
            return {"allowed": False, "reason": f"replay_depth_exceeded ({depth})"}
        return {"allowed": True, "reason": None}
