"""
Provider telemetry persistence -- success/failure counts, latency, cost,
and daily request/token usage per provider+capability.

This used to live as extra tables/methods bolted onto agent.py's task
MemoryDB, which was a real instance of the "everything ends up in one
file" problem: provider operational telemetry isn't task memory, and
conflating them made both harder to reason about. Split out here so the
provider layer owns its own data.
"""

import sqlite3
from datetime import datetime, timezone


class ProviderStatsStore:
    def __init__(self, path="provider_stats.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS provider_stats (
                provider TEXT,
                capability TEXT,
                calls INTEGER DEFAULT 0,
                successes INTEGER DEFAULT 0,
                total_latency_ms REAL DEFAULT 0,
                total_cost_usd REAL DEFAULT 0,
                PRIMARY KEY (provider, capability)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_usage (
                provider TEXT,
                day TEXT,
                requests INTEGER DEFAULT 0,
                tokens INTEGER DEFAULT 0,
                PRIMARY KEY (provider, day)
            )
        """)
        self.conn.commit()

    # ---- per-capability call stats (success/latency/cost) ----

    def record_provider_call(self, provider, capability, success, latency_ms, cost_usd=0.0):
        self.conn.execute(
            "INSERT INTO provider_stats (provider, capability, calls, successes, total_latency_ms, total_cost_usd) "
            "VALUES (?, ?, 1, ?, ?, ?) "
            "ON CONFLICT(provider, capability) DO UPDATE SET "
            "calls = calls + 1, "
            "successes = successes + ?, "
            "total_latency_ms = total_latency_ms + ?, "
            "total_cost_usd = total_cost_usd + ?",
            (provider, capability, int(success), latency_ms, cost_usd,
             int(success), latency_ms, cost_usd),
        )
        self.conn.commit()

    def get_provider_stats(self, provider, capability):
        cur = self.conn.execute(
            "SELECT calls, successes, total_latency_ms, total_cost_usd FROM provider_stats WHERE provider=? AND capability=?",
            (provider, capability),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"calls": row[0], "successes": row[1], "total_latency_ms": row[2], "total_cost_usd": row[3] or 0.0}

    def get_total_cost(self):
        cur = self.conn.execute("SELECT COALESCE(SUM(total_cost_usd), 0) FROM provider_stats")
        return cur.fetchone()[0]

    def get_all_provider_stats(self):
        cur = self.conn.execute("SELECT provider, capability, calls, successes, total_latency_ms, total_cost_usd FROM provider_stats")
        return [
            {"provider": r[0], "capability": r[1], "calls": r[2], "successes": r[3],
             "total_latency_ms": r[4], "total_cost_usd": r[5] or 0.0}
            for r in cur.fetchall()
        ]

    # ---- daily request/token usage (for quota tracking) ----

    def record_daily_usage(self, provider, tokens, requests=1, day=None):
        day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.conn.execute(
            "INSERT INTO daily_usage (provider, day, requests, tokens) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(provider, day) DO UPDATE SET "
            "requests = requests + ?, tokens = tokens + ?",
            (provider, day, requests, tokens, requests, tokens),
        )
        self.conn.commit()

    def get_daily_usage(self, provider, day=None):
        day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cur = self.conn.execute(
            "SELECT requests, tokens FROM daily_usage WHERE provider=? AND day=?",
            (provider, day),
        )
        row = cur.fetchone()
        return {"requests": row[0], "tokens": row[1]} if row else {"requests": 0, "tokens": 0}

    def get_all_daily_usage(self, day=None):
        day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cur = self.conn.execute("SELECT provider, requests, tokens FROM daily_usage WHERE day=?", (day,))
        return {r[0]: {"requests": r[1], "tokens": r[2]} for r in cur.fetchall()}
