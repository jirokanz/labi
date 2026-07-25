import sys

try:
    import resource
    HAVE_RESOURCE = True
except ImportError:
    # `resource` is POSIX-only (no-op shim on Windows). We degrade
    # gracefully rather than crashing at import time, since this module
    # gets imported by anything that touches the tools package.
    resource = None
    HAVE_RESOURCE = False


def set_resource_limits(cpu_seconds=5.0, memory_mb=128, output_bytes=1048576):
    if not HAVE_RESOURCE:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_seconds), int(cpu_seconds) + 1))
    except (ValueError, OSError):
        pass
    mem_bytes = memory_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
    except (ValueError, OSError):
        pass
    try:
        # Cap number of child processes/threads the sandboxed script can spawn.
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    except (ValueError, OSError, AttributeError):
        pass


def make_preexec_fn(cpu_seconds=5.0, memory_mb=128):
    """
    Returns a callable suitable for subprocess.Popen/run(preexec_fn=...)
    on POSIX, or None on platforms where that's unsupported (e.g. Windows,
    where subprocess doesn't accept preexec_fn at all).
    """
    if not HAVE_RESOURCE or sys.platform == "win32":
        return None
    return lambda: set_resource_limits(cpu_seconds=cpu_seconds, memory_mb=memory_mb)


def enforce_output_limit(output, max_bytes):
    encoded = output.encode("utf-8")
    if len(encoded) > max_bytes:
        return encoded[:max_bytes].decode("utf-8", errors="ignore") + "\n... [TRUNCATED]"
    return output
