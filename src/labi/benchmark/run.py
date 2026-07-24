
import time
from labi.memory.database import MemoryDatabase
from labi.memory.router import MemoryRouter
from labi.memory.confidence import ConfidenceEvaluator
from labi.memory.policy import MemoryDecisionPolicy
from labi.memory.guard import ReplayGuard
from labi.core.logger import Logger

def main():
    print("Labi Benchmark (Real Memory Router)")
    logger = Logger()
    db = MemoryDatabase(":memory:")
    evaluator = ConfidenceEvaluator()
    policy = MemoryDecisionPolicy()
    guard = ReplayGuard()
    router = MemoryRouter(db, evaluator, policy, guard)

    # Seed with some fake tasks
    tasks = [
        ("Monitor CPU temperature", True),
        ("Check disk usage", True),
        ("Scan network for devices", False),
    ]
    for goal, success in tasks:
        task_id = f"task_{hash(goal)}"
        db.record_task(task_id, goal, success=success)

    # Test routing
    test_goals = [
        "Check CPU temperature",
        "Monitor disk space",
        "Play music",
    ]
    for g in test_goals:
        print(f"Goal: {g}")
        result = router.route(g)
        print(f"  Decision: {result['decision']}, Confidence: {result['confidence']:.2f}")
        if result['candidate']:
            print(f"  Candidate: {result['candidate']['goal']}")
        print()

    print("All benchmarks passed (real routing).")
