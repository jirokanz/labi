"""Apply SQL migrations on boot, idempotently."""

import sqlite3
from pathlib import Path

class MigrationManager:
    def __init__(self, db_path: str, migrations_dir: str):
        self.db_path = db_path
        self.migrations_dir = Path(migrations_dir)

    def initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor = conn.execute("SELECT filename FROM migrations")
            applied = {row[0] for row in cursor.fetchall()}
            for sql_file in sorted(self.migrations_dir.glob("*.sql")):
                if sql_file.name in applied:
                    continue
                print(f"📦 Applying migration: {sql_file.name}")
                with open(sql_file, "r") as f:
                    conn.executescript(f.read())
                conn.execute("INSERT INTO migrations (filename) VALUES (?)", (sql_file.name,))
                conn.commit()
