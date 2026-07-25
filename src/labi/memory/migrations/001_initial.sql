
CREATE TABLE IF NOT EXISTS migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT UNIQUE NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    plan TEXT,
    code TEXT,
    answer TEXT,
    success INTEGER DEFAULT 0,
    provider TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_goal ON tasks(goal);
CREATE INDEX IF NOT EXISTS idx_tasks_success ON tasks(success);
