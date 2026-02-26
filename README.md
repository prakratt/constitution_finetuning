# Constitutional Fine-Tuning

Fine-tune open language models on [Claude's constitution](https://github.com/anthropics/claude-constitution) to improve alignment benchmarks (sycophancy, strongREJECT, etc.).

## How it works

1. **Fetch** — Downloads Claude's constitution from GitHub and parses it into structured principles
2. **Generate** — Uses Claude (via OpenRouter) to generate diverse, high-quality training conversations that demonstrate each principle
3. **Train** — LoRA fine-tunes a target model (default: Llama-3.1-8B-Instruct) on the generated data using the Tinker API

## Setup

```bash
# Install
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env with your TINKER_API_KEY and OPENROUTER_API_KEY
```

## Usage

```bash
# Full pipeline
confine run --config configs/default.yaml

# Individual steps
confine fetch    --config configs/default.yaml   # Download & parse constitution
confine generate --config configs/default.yaml   # Generate training data
confine train    --config configs/default.yaml   # Fine-tune with Tinker
```

## Configuration

Edit `configs/default.yaml` to customize:

- **Constitution**: source URL, target model name
- **Data generation**: API model, concurrency, examples per principle
- **Training**: base model, LoRA rank, learning rate, batch size, epochs

## Remote GPU Setup

```bash
# On a remote instance with GPU access:
bash scripts/setup_remote.sh
bash scripts/run_pipeline.sh
```

## Project Structure

```
configs/default.yaml          — Pipeline configuration
src/constitution_finetune/
  config.py                   — Pydantic config models + YAML loader
  cli.py                      — CLI entrypoint (confine command)
  constitution/               — Fetch, parse, and structure principles
  datagen/                    — Generate training data via Claude API
  training/                   — Tinker LoRA training loop + smoke tests
```
