class Scheduler:
    def __init__(self):
        self.tasks = {}
    def schedule(self, task):
        self.tasks[task.id] = task
        return task
