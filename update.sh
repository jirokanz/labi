#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "No venv/ found -- this looks like a fresh checkout, not an existing install."
    echo "Run ./install.sh instead."
    exit 1
fi

echo "Pulling latest changes..."
git pull

source venv/bin/activate

echo
echo "Reinstalling (--force-reinstall picks up new/changed packages, e.g. new top-level"
echo "modules added since your last install, which a plain 'pip install -e .' can silently miss)..."
pip install -e . --force-reinstall --no-deps

echo
echo "Running test suite as a sanity check..."
if python3 -m pytest tests/ -q; then
    echo "All tests passed."
else
    echo "WARNING: some tests failed after the update -- see output above before relying on 'labi run'."
fi

echo
echo "Update complete."
