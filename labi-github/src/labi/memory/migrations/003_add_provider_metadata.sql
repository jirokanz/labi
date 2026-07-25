
ALTER TABLE tasks ADD COLUMN provider_key TEXT DEFAULT 'unknown:unknown';
ALTER TABLE tasks ADD COLUMN role TEXT DEFAULT 'answer';
ALTER TABLE tasks ADD COLUMN category TEXT DEFAULT 'unknown';
ALTER TABLE tasks ADD COLUMN last_used_at TIMESTAMP;
ALTER TABLE tasks ADD COLUMN usage_count INTEGER DEFAULT 0;
UPDATE tasks SET provider_key = provider WHERE provider IS NOT NULL AND provider != '';
CREATE INDEX IF NOT EXISTS idx_tasks_provider_key ON tasks(provider_key);
