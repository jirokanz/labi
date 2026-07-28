"""Base workflow interface."""

from abc import ABC, abstractmethod
from typing import List

from labi.core.task import Task
from labi.context.manager import ContextManager
from labi.agents.base import BaseAgent


class BaseWorkflow(ABC):
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
        self._agents: List[BaseAgent] = []

    def add_agent(self, agent: BaseAgent) -> "BaseWorkflow":
        self._agents.append(agent)
        return self

    @abstractmethod
    def execute(self, task: Task) -> dict:
        raise NotImplementedError
