
ALTER TABLE tasks ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE tasks ADD COLUMN zr_version TEXT DEFAULT '0.2.0';
ALTER TABLE tasks ADD COLUMN deps_hash TEXT DEFAULT 'unknown';
ALTER TABLE tasks ADD COLUMN execution_hash TEXT DEFAULT '';
UPDATE tasks SET zr_version = '0.2.0' WHERE zr_version IS NULL;
UPDATE tasks SET deps_hash = 'unknown' WHERE deps_hash IS NULL;
CREATE INDEX IF NOT EXISTS idx_tasks_zr_version ON tasks(zr_version);
