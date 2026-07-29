"""Code extraction + sandboxed execution.

Pulled out of agent.py for the same reason providers/generation.py was:
agents/executor.py needs real sandboxed execution (static validation +
subprocess boundary + CPU/memory rlimits), not just LLM-generated code
that's never actually run. Duplicating this logic inside agents/executor.py
instead of importing it would mean two places could drift on what
"safe to run" means -- this module is the one place that decides that,
and both agent.py's CLI loop and the agents/ workflow package depend on
it rather than on each other.
"""

import os
import subprocess
import sys
import tempfile
import re

from labi.tools.python.security import validate_code as static_validate_code
from labi.tools.python.limits import make_preexec_fn, enforce_output_limit

DEFAULT_EXECUTION_TIMEOUT = 20
DEFAULT_MAX_OUTPUT_BYTES = 1_048_576


def extract_code(text):
    match = re.search(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def execute_code(code, timeout=DEFAULT_EXECUTION_TIMEOUT, max_output_bytes=DEFAULT_MAX_OUTPUT_BYTES):
    """Run generated code with: static validation, a subprocess boundary,
    and (on POSIX) CPU/memory rlimits -- never a bare subprocess.run with
    no checks. Returns (stdout, stderr, exit_code); exit_code -2 means
    static validation blocked it before it ever ran, -1 means it timed
    out."""
    code = extract_code(code)

    violations = static_validate_code(code)
    if violations:
        return "", "Blocked before execution:\n- " + "\n- ".join(violations), -2

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout,
            preexec_fn=make_preexec_fn(cpu_seconds=timeout, memory_mb=256),
        )
        stdout, stderr, exit_code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, exit_code = "", "Execution timed out", -1
    finally:
        os.unlink(fname)

    stdout = enforce_output_limit(stdout, max_output_bytes)
    stderr = enforce_output_limit(stderr, max_output_bytes)
    return stdout, stderr, exit_code
