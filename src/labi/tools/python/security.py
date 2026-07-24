"""
Static validation for LLM-generated code before it is ever executed.

This is a defense-in-depth layer, NOT a sandbox by itself — it must always
be paired with process-level isolation (see limits.py + runner.py:
subprocess + resource limits + restricted filesystem). A determined model
or adversarial prompt can still produce code that this checker misses;
the goal is to catch the common/obvious cases cheaply before spending a
subprocess on them.
"""

import ast

# Modules that grant filesystem, process, network, or interpreter-escape
# capabilities. Broader than the original list (which missed importlib,
# pickle, pathlib, and networking modules).
BLOCKED_IMPORTS = {
    "os", "subprocess", "socket", "shutil", "sys", "ctypes", "fcntl",
    "posix", "pty", "signal", "importlib", "pickle", "marshal", "code",
    "codecs", "multiprocessing", "threading", "pathlib", "io",
    "urllib", "http", "ftplib", "telnetlib", "smtplib", "asyncio",
    "platform", "pip", "setuptools", "resource", "mmap", "tty",
}

# Builtins that let code route around the import-based checks above
# (dynamic import, arbitrary code execution, attribute-string access).
BLOCKED_CALLS = {
    "eval", "exec", "compile", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "open", "input",
    "breakpoint", "exit", "quit",
}

# Dunder attribute access is how sandboxes get escaped in practice
# (e.g. ().__class__.__base__.__subclasses__()).
BLOCKED_ATTR_PREFIXES = ("__",)


def validate_code(code: str):
    """Return a list of human-readable violation strings; empty == passed static checks."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in BLOCKED_IMPORTS:
                violations.append(f"Blocked import: {node.module}")

        elif isinstance(node, ast.Call):
            fn = node.func
            name = None
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            if name in BLOCKED_CALLS:
                violations.append(f"Blocked call: {name}(...)")

        elif isinstance(node, ast.Attribute):
            if node.attr.startswith(BLOCKED_ATTR_PREFIXES) and node.attr not in ("__init__",):
                violations.append(f"Blocked dunder attribute access: .{node.attr}")

        elif isinstance(node, ast.Name):
            # Catches bare references used to smuggle builtins, e.g.
            # `x = __builtins__` then `x.eval(...)`.
            if node.id in ("__builtins__", "__loader__", "__import__"):
                violations.append(f"Blocked name reference: {node.id}")

    # De-duplicate while preserving order.
    seen = set()
    deduped = []
    for v in violations:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped
