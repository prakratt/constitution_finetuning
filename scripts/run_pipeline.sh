#!/usr/bin/env bash
# End-to-end pipeline execution
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Load .env if present
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

echo "=== Running Constitutional Fine-Tuning Pipeline ==="
confine run --config configs/default.yaml "$@"
