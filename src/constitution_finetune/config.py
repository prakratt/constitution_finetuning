"""Pipeline configuration models and YAML loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ConstitutionConfig(BaseModel):
    url: str = "https://raw.githubusercontent.com/anthropics/claude-constitution/refs/heads/main/20260120-constitution.md"
    cache_path: str = "data/constitution.md"
    target_model_name: str = "Qwen"


class DatagenConfig(BaseModel):
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    model: str = "anthropic/claude-sonnet-4"
    examples_per_principle: int = 3
    max_principles_per_category: int = 30
    max_concurrency: int = 10
    output_path: str = "data/training_data.jsonl"
    max_retries: int = 3


class AdamConfig(BaseModel):
    learning_rate: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


class TrainingConfig(BaseModel):
    base_model: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    adam: AdamConfig = Field(default_factory=AdamConfig)
    batch_size: int = 32
    epochs: int = 3
    max_seq_length: int = 2048
    warmup_fraction: float = 0.05
    save_every_steps: int = 50
    run_name: str = "constitutional-qwen3-8b"
    tinker_base_url: str | None = None


class PipelineConfig(BaseModel):
    constitution: ConstitutionConfig = Field(default_factory=ConstitutionConfig)
    datagen: DatagenConfig = Field(default_factory=DatagenConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate pipeline config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig.model_validate(raw or {})
