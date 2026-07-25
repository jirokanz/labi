import json
"""MemoryDB - Long-term memory substrate with hard compatibility filters."""

import sqlite3
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from labi.memory.models import MemoryRecord, MemoryCandidate
from labi.memory.migration_manager import MigrationManager
from labi.core.version import LABI_VERSION, get_major_version

class MemoryDB:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._current_major = get_major_version(LABI_VERSION)
        migrations_dir = Path(__file__).parent / "migrations"
        MigrationManager(db_path, migrations_dir).initialize()

    def find_similar(self, goal: str, limit: int = 5) -> List[MemoryCandidate]:
        goal_words = set(goal.lower().split())
        if not goal_words:
            return []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, goal, plan, code, answer AS result,
                       success, execution_hash, zr_version, deps_hash,
                       provider_key, role, category, task_complexity, task_category,
                       created_at, last_used_at, usage_count
                FROM tasks
                WHERE success = 1 AND substr(zr_version, 1, 1) = ?
                ORDER BY created_at DESC LIMIT 50
            """, (self._current_major,))
            candidates = []
            for row in cursor.fetchall():
                record = MemoryRecord.from_db_row(dict(row))
                stored_words = set(record.goal.lower().split())
                if not stored_words:
                    continue
                overlap = len(goal_words & stored_words) / max(len(goal_words), len(stored_words))
                candidates.append(MemoryCandidate(record=record, similarity=overlap))
            candidates.sort(key=lambda x: x.similarity, reverse=True)
            return candidates[:limit]

    def save(self, record: MemoryRecord) -> None:
        d = record.to_db_dict()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tasks (
                    id, goal, plan, code, answer, success,
                    execution_hash, zr_version, deps_hash,
                    provider_key, role, category,
                    task_complexity, task_category,
                    created_at, last_used_at, usage_count
                ) VALUES (
                    :id, :goal, :plan, :code, :result, :success,
                    :execution_hash, :zr_version, :deps_hash,
                    :provider_key, :role, :category,
                    :task_complexity, :task_category,
                    :created_at, :last_used_at, :usage_count
                )
            """, d)

    def mark_used(self, memory_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE tasks SET last_used_at = CURRENT_TIMESTAMP, usage_count = usage_count + 1
                WHERE id = ?
            """, (memory_id,))

    def get_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, goal, plan, code, answer AS result,
                       success, execution_hash, zr_version, deps_hash,
                       provider_key, role, category, task_complexity, task_category,
                       created_at, last_used_at, usage_count
                FROM tasks WHERE id = ?
            """, (memory_id,))
            row = cursor.fetchone()
            return MemoryRecord.from_db_row(dict(row)) if row else None


# ---------------------------------------------------------------------------
# Legacy MemoryDatabase -- kept alongside MemoryDB above, not replaced by it.
# The refactor that introduced MemoryDB (with MemoryRecord/MemoryCandidate,
# execution_hash/deps_hash compatibility checks, and SQL migrations) deleted
# this class from the file, but runtime.py, benchmark/run.py, memory/router.py
# (old ConfidenceEvaluator/MemoryDecisionPolicy/ReplayGuard pipeline), and
# replay/manager.py all still call its API (find_similar_goals, record_task,
# get_task, update_task_metadata, register_replay_attempt/result) -- that's
# a different shape than MemoryDB's (find_similar, save, mark_used,
# get_by_id). Deleting it broke every import of those four modules and made
# the test suite fail to even collect. Restoring it here until the two
# memory subsystems are consolidated into one (see PR discussion).
# ---------------------------------------------------------------------------

class MemoryDatabase:
    def __init__(self, path=":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                goal TEXT,
                success BOOLEAN,
                replay_count INTEGER DEFAULT 0,
                replay_success_count INTEGER DEFAULT 0,
                replay_depth INTEGER DEFAULT 0,
                archived BOOLEAN DEFAULT 0,
                workspace_path TEXT,
                metadata TEXT
            )
        """)
        self.conn.commit()

    def record_task(self, task_id, goal, success=False, workspace_path=None, replay_depth=0):
        self.conn.execute(
            "INSERT OR REPLACE INTO tasks (id, goal, success, workspace_path, replay_depth) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, goal, success, workspace_path, replay_depth)
        )
        self.conn.commit()

    def get_task(self, task_id):
        cur = self.conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,))
        row = cur.fetchone()
        if row:
            return {
                "id": row[0],
                "goal": row[1],
                "success": row[2],
                "replay_count": row[3],
                "replay_success_count": row[4],
                "replay_depth": row[5],
                "archived": row[6],
                "workspace_path": row[7],
                "metadata": row[8],
            }
        return None

    def find_similar_goals(self, goal, limit=3):
        words = set(goal.lower().split())
        cursor = self.conn.execute(
            "SELECT id, goal, success, replay_count, replay_success_count, replay_depth, archived FROM tasks"
        )
        candidates = []
        for row in cursor.fetchall():
            task_words = set(row[1].lower().split())
            similarity = len(words & task_words) / len(words | task_words) if words else 0
            if similarity > 0.3:
                candidates.append({
                    "id": row[0],
                    "goal": row[1],
                    "success": row[2],
                    "replay_count": row[3],
                    "replay_success_count": row[4],
                    "replay_depth": row[5],
                    "archived": row[6],
                    "similarity": similarity,
                })
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:limit]

    def update_task_metadata(self, task_id, metadata):
        self.conn.execute(
            "UPDATE tasks SET metadata = ? WHERE id = ?",
            (json.dumps(metadata), task_id)
        )
        self.conn.commit()

    def register_replay_attempt(self, task_id):
        self.conn.execute(
            "UPDATE tasks SET replay_count = replay_count + 1 WHERE id = ?",
            (task_id,)
        )
        self.conn.commit()

    def register_replay_result(self, task_id, success):
        if success:
            self.conn.execute(
                "UPDATE tasks SET replay_success_count = replay_success_count + 1 WHERE id = ?",
                (task_id,)
            )
        self.conn.commit()
