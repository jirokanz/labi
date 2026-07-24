"""Dependency fingerprinting for memory compatibility."""

import hashlib
import subprocess

def dependency_hash() -> str:
    """Compute SHA256 hash of current pip freeze output."""
    try:
        output = subprocess.check_output(["pip", "freeze"], text=True, stderr=subprocess.DEVNULL)
        return hashlib.sha256(output.encode()).hexdigest()[:16]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
