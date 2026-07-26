#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    echo "venv/ already exists -- reusing it. (Delete it first if you want a fully clean rebuild.)"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -e .

echo
echo "Running test suite as a sanity check..."
if python3 -m pytest tests/ -q; then
    echo "All tests passed."
else
    echo "WARNING: some tests failed -- the install completed, but something may be broken. See output above."
fi

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo
    echo "No .env found. Copy .env.example to .env and fill in your API keys:"
    echo "  cp .env.example .env"
fi

if [ "$1" == "--local" ]; then
    echo
    echo "Installation complete (local venv only)."
    echo "Activate with: source venv/bin/activate"
    echo "Or re-run this script without --local to also install a system-wide 'labi' command."
else
    echo
    echo "Setting up system-wide 'labi' command (requires sudo)..."
    sudo tee /usr/local/bin/labi > /dev/null << EOF
#!/bin/bash
exec "$SCRIPT_DIR/venv/bin/python" -m labi.cli "\$@"
EOF
    sudo chmod +x /usr/local/bin/labi
    echo "Done -- 'labi' is now available from any shell, no activation needed."
    echo "(Run this script with --local instead if you don't want a system-wide command / don't have sudo.)"
fi
