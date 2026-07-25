class CostTracker:
    """Accumulates $ cost across the multiple LLM calls that make up one
    task (plan + code + N feedback/auto-fix rounds), so it can be shown
    and saved as a single per-task total."""

    def __init__(self):
        self.total = 0.0
        self.calls = 0

    def add(self, cost):
        if cost:
            self.total += cost
        self.calls += 1
