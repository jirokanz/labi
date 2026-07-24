"""Labi version management."""

LABI_VERSION = "0.2.0"

def get_major_version(version: str) -> str:
    """Extract major version (e.g., '0.2.0' → '0')."""
    return version.split(".")[0]

def compatible(a: str, b: str) -> bool:
    """Check if two versions are compatible (same major version)."""
    return get_major_version(a) == get_major_version(b)
