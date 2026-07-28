"""A unit of work handed to a workflow -- deliberately minimal, matching
what TaskManager (core/task_manager.py) already expects: any object with
an .id attribute. Workflows read .goal and .context_id off it; nothing
else in this repo currently depends on its shape, so it stays small
rather than anticipating fields nothing uses yet."""

from dataclasses import dataclass


@dataclass
class Task:
    id: str
    goal: str
    context_id: str
