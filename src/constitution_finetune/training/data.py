"""Load JSONL training data and convert to Tinker Datum objects."""

from __future__ import annotations

import json
from pathlib import Path

import tinker
from rich.console import Console
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

console = Console()


def load_jsonl(path: str | Path) -> list[dict]:
    """Load conversations from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    conversations = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))

    console.print(f"[dim]Loaded {len(conversations)} conversations from {path}[/dim]")
    return conversations


def prepare_datums(
    conversations: list[dict],
    model_name: str,
    max_seq_length: int,
) -> list[tinker.Datum]:
    """Convert conversation dicts to Tinker Datum objects.

    Uses tinker-cookbook's conversation_to_datum which handles tokenization,
    chat templates, and loss masking.
    """
    tokenizer = get_tokenizer(model_name)
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    datums: list[tinker.Datum] = []
    skipped = 0

    for conv in conversations:
        messages = conv["messages"]
        try:
            datum = conversation_to_datum(
                conversation=messages,
                renderer=renderer,
                max_length=max_seq_length,
                train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            datums.append(datum)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                console.print(f"[yellow]Skipped conversation: {e}[/yellow]")

    if skipped:
        console.print(f"[yellow]Skipped {skipped} conversations during tokenization[/yellow]")

    console.print(f"[green]Prepared {len(datums)} training datums[/green]")
    return datums
