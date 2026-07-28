"""
Source memory -- records which URLs were consulted for which task, so web
answers can be cited and a later, near-identical search can check whether
a source's content actually changed before re-summarizing it.

This is task-adjacent telemetry, not core task memory, so -- following
the same split that gave provider stats their own store instead of more
columns on MemoryDB (see providers/stats.py's docstring) -- it gets its
own small sqlite store rather than growing memory/database.py further.
"""

import hashlib
import sqlite3
from datetime import datetime, timezone


class SourceStore:
    def __init__(self, path="sources.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                task_id TEXT,
                url TEXT,
                title TEXT,
                timestamp TEXT,
                content_hash TEXT,
                PRIMARY KEY (task_id, url)
            )
        """)
        self.conn.commit()

    @staticmethod
    def hash_content(content):
        return hashlib.sha256((content or "").encode("utf-8", errors="replace")).hexdigest()

    def record(self, task_id, url, title, content):
        self.conn.execute(
            "INSERT OR REPLACE INTO sources (task_id, url, title, timestamp, content_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, url, title, datetime.now(timezone.utc).isoformat(), self.hash_content(content)),
        )
        self.conn.commit()

    def get_sources(self, task_id):
        cur = self.conn.execute(
            "SELECT url, title, timestamp, content_hash FROM sources WHERE task_id=?",
            (task_id,),
        )
        return [
            {"url": r[0], "title": r[1], "timestamp": r[2], "content_hash": r[3]}
            for r in cur.fetchall()
        ]

    def unchanged_since(self, url, content, within_hours=24):
        """True if this URL was already recorded, with identical content
        (by hash), within the last within_hours -- lets a caller skip
        re-summarizing (and re-spending an LLM call on) a source that
        hasn't actually changed since it was last read."""
        cur = self.conn.execute(
            "SELECT timestamp, content_hash FROM sources WHERE url=? ORDER BY timestamp DESC LIMIT 1",
            (url,),
        )
        row = cur.fetchone()
        if not row:
            return False
        ts, stored_hash = row
        if stored_hash != self.hash_content(content):
            return False
        age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() / 3600
        return age_hours < within_hours
