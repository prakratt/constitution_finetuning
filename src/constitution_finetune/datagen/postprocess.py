"""Validate, deduplicate, and save training data as JSONL."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rich.console import Console

console = Console()


def _validate_conversation(conv: dict) -> bool:
    """Check that a conversation has valid structure."""
    messages = conv.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False

    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if msg.get("role") not in ("user", "assistant"):
            return False
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            return False

    # Check alternating roles (first should be user)
    if messages[0]["role"] != "user":
        return False
    for i in range(1, len(messages)):
        if messages[i]["role"] == messages[i - 1]["role"]:
            return False

    return True


def _conversation_hash(conv: dict) -> str:
    """Hash based on user messages to detect near-duplicates."""
    user_texts = [
        m["content"].strip().lower()
        for m in conv["messages"]
        if m["role"] == "user"
    ]
    combined = "|||".join(user_texts)
    return hashlib.sha256(combined.encode()).hexdigest()


def postprocess_and_save(
    conversations: list[dict],
    output_path: str,
) -> Path:
    """Validate, deduplicate, and save conversations to JSONL.

    Returns the Path to the saved file.
    """
    # Validate
    valid = [c for c in conversations if _validate_conversation(c)]
    n_invalid = len(conversations) - len(valid)
    if n_invalid:
        console.print(f"[yellow]Filtered {n_invalid} invalid conversations[/yellow]")

    # Deduplicate
    seen: set[str] = set()
    unique: list[dict] = []
    for conv in valid:
        h = _conversation_hash(conv)
        if h not in seen:
            seen.add(h)
            unique.append(conv)

    n_dupes = len(valid) - len(unique)
    if n_dupes:
        console.print(f"[yellow]Removed {n_dupes} duplicate conversations[/yellow]")

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for conv in unique:
            f.write(json.dumps(conv) + "\n")

    console.print(
        f"[green]Saved {len(unique)} conversations to {out}[/green]"
    )
    return out
