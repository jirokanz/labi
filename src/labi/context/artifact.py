"""Artifact management -- what an agent produces (code, a plan, a report)
during a workflow run, kept separate from ContextSnapshot/TaskUpdate so
large content (a generated file) doesn't get copied into every snapshot."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid
from datetime import datetime, timezone


class ArtifactType(Enum):
    CODE = "code"
    FILE = "file"
    OUTPUT = "output"
    REPORT = "report"
    PLAN = "plan"


@dataclass
class Artifact:
    artifact_id: str
    type: ArtifactType
    name: str
    content: str
    created_by: Optional[str] = None
    task_id: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class ArtifactStore:
    """In-memory only -- artifacts live for the duration of one workflow
    run. Anything that needs to persist past that (the final answer, the
    provider that produced it, its cost) already goes through MemoryDB
    in agent.py; this store isn't a replacement for that."""

    def __init__(self):
        self._artifacts: Dict[str, Artifact] = {}

    def store_artifact(self, name: str, content: str, artifact_type: ArtifactType,
                        task_id: str = None, created_by: str = None,
                        metadata: Dict[str, Any] = None) -> str:
        artifact_id = str(uuid.uuid4())[:8]
        artifact = Artifact(
            artifact_id=artifact_id,
            type=artifact_type,
            name=name,
            content=content,
            created_by=created_by,
            task_id=task_id,
            metadata=metadata or {},
        )
        self._artifacts[artifact_id] = artifact
        return artifact_id

    def get(self, artifact_id: str) -> Optional[Artifact]:
        return self._artifacts.get(artifact_id)

    def list_by_task(self, task_id: str) -> List[Artifact]:
        return [a for a in self._artifacts.values() if a.task_id == task_id]
