
import uuid
import time
from labi.core.event_bus import SyncEventBus
from labi.core.scheduler import Scheduler
from labi.core.task_manager import TaskManager
from labi.memory.database import MemoryDatabase
from labi.memory.router import MemoryRouter
from labi.memory.confidence import ConfidenceEvaluator
from labi.memory.policy import MemoryDecisionPolicy
from labi.memory.guard import ReplayGuard
from labi.workspace.manager import WorkspaceManager
from labi.replay.manager import ReplayManager
from labi.core.logger import Logger

def main():
    print("Labi v0.1.0 (Full Memory)")
    logger = Logger()
    db = MemoryDatabase(":memory:")
    evaluator = ConfidenceEvaluator()
    policy = MemoryDecisionPolicy()
    guard = ReplayGuard()
    router = MemoryRouter(db, evaluator, policy, guard)

    # Simulate a task
    task_id = "test_001"
    goal = "Print hello world"
    decision = router.route(goal)
    print(f"Decision for '{goal}': {decision}")
    if decision["decision"] == "regenerate":
        print("Generating fresh solution (no memory)")
    elif decision["decision"] == "reuse":
        print(f"Reusing candidate: {decision['candidate']['id']}")
    elif decision["decision"] == "adapt":
        print(f"Adapting candidate: {decision['candidate']['id']}")

if __name__ == "__main__":
    main()
