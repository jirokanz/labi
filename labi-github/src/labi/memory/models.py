"""MemoryRecord and MemoryCandidate dataclasses."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class MemoryRecord:
    id: str
    goal: str
    category: str = "unknown"
    plan: Optional[str] = None
    code: Optional[str] = None
    result: Optional[str] = None
    success: bool = False
    execution_hash: str = ""
    zr_version: str = "0.2.0"
    deps_hash: str = "unknown"
    provider_key: str = "unknown:unknown"
    role: str = "answer"
    task_complexity: float = 0.5
    task_category: str = "unknown"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    usage_count: int = 0

    @classmethod
    def from_db_row(cls, row: dict) -> "MemoryRecord":
        return cls(
            id=row["id"],
            goal=row["goal"],
            category=row.get("category", "unknown"),
            plan=row.get("plan"),
            code=row.get("code"),
            result=row.get("result"),
            success=bool(row.get("success", 0)),
            execution_hash=row.get("execution_hash", ""),
            zr_version=row.get("zr_version", "0.2.0"),
            deps_hash=row.get("deps_hash", "unknown"),
            provider_key=row.get("provider_key", "unknown:unknown"),
            role=row.get("role", "answer"),
            task_complexity=row.get("task_complexity", 0.5),
            task_category=row.get("task_category", "unknown"),
            created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"],
            last_used_at=datetime.fromisoformat(row["last_used_at"]) if row.get("last_used_at") else None,
            usage_count=row.get("usage_count", 0),
        )

    def to_db_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "category": self.category,
            "plan": self.plan,
            "code": self.code,
            "result": self.result,
            "success": 1 if self.success else 0,
            "execution_hash": self.execution_hash,
            "zr_version": self.zr_version,
            "deps_hash": self.deps_hash,
            "provider_key": self.provider_key,
            "role": self.role,
            "task_complexity": self.task_complexity,
            "task_category": self.task_category,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "usage_count": self.usage_count,
        }

@dataclass
class MemoryCandidate:
    record: MemoryRecord
    similarity: float
    similarity_method: str = "jaccard"
