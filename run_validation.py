#!/usr/bin/env python3
"""
Labi -- Validation Benchmark

Unlike the earlier version of this script, every task here is actually
*executed* (sandboxed, via labi.agent.execute_code) and its real
pass/fail is what gets recorded -- "success" here means the generated
code ran and exited 0, not "the LLM call didn't throw."
"""

import os
import random
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv
import litellm
from litellm.exceptions import RateLimitError

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from labi.agent import extract_code, execute_code  # noqa: E402

load_dotenv()

MODEL = "groq/llama-3.3-70b-versatile"
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    print("No API key found. Set GROQ_API_KEY in .env")
    exit(1)

DB_PATH = "validation.db"


def call_llm(prompt, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = litellm.completion(
        model=MODEL, messages=messages, api_key=API_KEY, max_tokens=512, temperature=0.3,
    )
    return response.choices[0].message.content


def call_llm_with_retry(prompt, system_prompt=None, max_retries=5, base_delay=2):
    for attempt in range(max_retries):
        try:
            return call_llm(prompt, system_prompt)
        except RateLimitError:
            wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit hit. Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay)
    raise Exception("Max retries exceeded")


class SimpleMemoryDB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                goal TEXT,
                code TEXT,
                success BOOLEAN,
                replay_count INTEGER DEFAULT 0,
                replay_success_count INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def record_task(self, goal, code, success):
        import uuid
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        self.conn.execute(
            "INSERT OR REPLACE INTO tasks (id, goal, code, success) VALUES (?, ?, ?, ?)",
            (task_id, goal, code, success),
        )
        self.conn.commit()
        return task_id

    def find_similar(self, goal, limit=3):
        words = set(goal.lower().split())
        cursor = self.conn.execute("SELECT id, goal, code, success, replay_count, replay_success_count FROM tasks")
        candidates = []
        for row in cursor.fetchall():
            task_words = set(row[1].lower().split())
            similarity = len(words & task_words) / len(words | task_words) if words else 0
            if similarity > 0.3:
                candidates.append({
                    "id": row[0], "goal": row[1], "code": row[2], "success": row[3],
                    "replay_count": row[4], "replay_success_count": row[5], "similarity": similarity,
                })
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:limit]

    def register_replay_attempt(self, task_id):
        self.conn.execute("UPDATE tasks SET replay_count = replay_count + 1 WHERE id = ?", (task_id,))
        self.conn.commit()

    def register_replay_result(self, task_id, success):
        if success:
            self.conn.execute("UPDATE tasks SET replay_success_count = replay_success_count + 1 WHERE id = ?", (task_id,))
        self.conn.commit()


def run_task(db, goal, first_run=True):
    """Returns (success, llm_used) where success is the REAL outcome of executing the code."""
    if first_run:
        candidates = db.find_similar(goal)
        if candidates and candidates[0]["similarity"] > 0.8 and candidates[0]["success"]:
            best = candidates[0]
            db.register_replay_attempt(best["id"])
            _, _, exit_code = execute_code(best["code"])
            success = exit_code == 0
            db.register_replay_result(best["id"], success)
            return success, False

        print(f"New task: {goal}")
        plan = call_llm_with_retry(f"Create a short numbered plan for: {goal}")
        time.sleep(1)
        raw_code = call_llm_with_retry(f"Write Python code for: {goal}\nPlan: {plan}\nOutput only code.")
        code = extract_code(raw_code)
        _, stderr, exit_code = execute_code(code)
        success = exit_code == 0
        if not success:
            print(f"   (execution failed: {stderr[:120] if stderr else 'no output'})")
        db.record_task(goal, code, success)
        return success, True
    else:
        candidates = db.find_similar(goal)
        if candidates and candidates[0]["similarity"] > 0.3:
            best = candidates[0]
            db.register_replay_attempt(best["id"])
            _, _, exit_code = execute_code(best["code"])
            success = exit_code == 0
            db.register_replay_result(best["id"], success)
            return success, False
        else:
            raw_code = call_llm_with_retry(f"Write Python code for: {goal}\nOutput only code.")
            code = extract_code(raw_code)
            _, _, exit_code = execute_code(code)
            success = exit_code == 0
            db.record_task(goal, code, success)
            return success, True


def benchmark():
    print("Labi - Validation")
    print("=" * 50)

    db = SimpleMemoryDB(DB_PATH)

    # Goals restricted to things that can genuinely run headless & be
    # verified pass/fail in this environment (the original list included
    # things like GPIO/DHT22/camera/Telegram that can't succeed here,
    # which would have made "real" execution results meaningless).
    goals = [
        "Check disk usage and print it",
        "Log system uptime to a string and print it",
        "Analyze a CSV file: count its rows",
        "Compute the sum of squares from 1 to 100",
        "Sort a list of random numbers and print them",
        "Print the current time in ISO format",
        "Count words in a string",
        "Reverse a string",
        "Check if a number is prime",
        "Generate the Fibonacci sequence up to n=10",
    ]

    print(f"Tasks: {len(goals)}\n" + "-" * 50)

    print("\nPhase 1: First run (cold start)")
    total_llm_calls, total_tasks, phase1_success = 0, 0, 0
    for i, g in enumerate(goals, 1):
        success, llm_used = run_task(db, g, first_run=True)
        if llm_used:
            total_llm_calls += 1
        if success:
            phase1_success += 1
        total_tasks += 1
        print(f"  {i:3d}. {g[:40]:40s} {'PASS' if success else 'FAIL'} {'LLM' if llm_used else 'MEM'}")
        time.sleep(0.5)

    print(f"\nPhase 1 complete. LLM calls: {total_llm_calls}/{total_tasks}. Real pass rate: {100 * phase1_success / total_tasks:.0f}%")

    print("\nPhase 2: Warm memory (reuse)")
    llm_calls_phase2, phase2_success = 0, 0
    for i, g in enumerate(goals, 1):
        success, llm_used = run_task(db, g, first_run=False)
        if llm_used:
            llm_calls_phase2 += 1
        if success:
            phase2_success += 1
        print(f"  {i:3d}. {g[:40]:40s} {'PASS' if success else 'FAIL'} {'LLM' if llm_used else 'MEM'}")
        time.sleep(0.5)

    print(f"\nPhase 2 complete. LLM calls: {llm_calls_phase2}/{len(goals)}. Real pass rate: {100 * phase2_success / len(goals):.0f}%")

    total_llm_saved = total_llm_calls - llm_calls_phase2
    reduction = (total_llm_saved / total_llm_calls) * 100 if total_llm_calls > 0 else 0

    print("\n" + "=" * 50)
    print("Validation Report (real execution, not assumed)")
    print("=" * 50)
    print(f"Total unique tasks:        {len(goals)}")
    print(f"LLM calls (cold start):    {total_llm_calls}")
    print(f"LLM calls (warm memory):   {llm_calls_phase2}")
    print(f"LLM calls saved:           {total_llm_saved}")
    print(f"LLM reduction:             {reduction:.1f}%")
    print(f"Phase 1 real pass rate:    {100 * phase1_success / total_tasks:.1f}%")
    print(f"Phase 2 real pass rate:    {100 * phase2_success / len(goals):.1f}%")

    os.remove(DB_PATH)


if __name__ == "__main__":
    benchmark()
