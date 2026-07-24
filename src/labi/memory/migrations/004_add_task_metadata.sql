
ALTER TABLE tasks ADD COLUMN task_complexity REAL DEFAULT 0.5;
ALTER TABLE tasks ADD COLUMN task_category TEXT DEFAULT 'unknown';
CREATE INDEX IF NOT EXISTS idx_tasks_complexity ON tasks(task_complexity);
