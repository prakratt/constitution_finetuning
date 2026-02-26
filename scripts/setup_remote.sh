#!/usr/bin/env bash
# One-command remote GPU setup for constitutional fine-tuning
set -euo pipefail

echo "=== Constitutional Fine-Tuning: Remote Setup ==="

# Clone repo if not present
if [ ! -d "constitution_finetune" ]; then
    echo "Cloning repository..."
    git clone https://github.com/YOUR_USER/constitution_finetune.git
    cd constitution_finetune
else
    cd constitution_finetune
    echo "Repository already present, pulling latest..."
    git pull
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "Installing dependencies..."
pip install -e .

# Prompt for API keys
echo ""
echo "=== API Key Setup ==="
if [ -z "${TINKER_API_KEY:-}" ]; then
    read -rp "Enter your Tinker API key: " TINKER_API_KEY
    export TINKER_API_KEY
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    read -rp "Enter your OpenRouter API key: " OPENROUTER_API_KEY
    export OPENROUTER_API_KEY
fi

# Write .env file
cat > .env << EOF
TINKER_API_KEY=${TINKER_API_KEY}
OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
EOF

echo ""
echo "=== Setup complete! ==="
echo "Run the pipeline with: ./scripts/run_pipeline.sh"
