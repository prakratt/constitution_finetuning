"""Download the constitution markdown from GitHub."""

from __future__ import annotations

from pathlib import Path

import requests
from rich.console import Console

console = Console()


def fetch_constitution(url: str, cache_path: str) -> str:
    """Download constitution markdown, caching locally.

    Returns the markdown text content.
    """
    cache = Path(cache_path)

    if cache.exists():
        console.print(f"[dim]Using cached constitution: {cache}[/dim]")
        return cache.read_text()

    console.print(f"[bold]Downloading constitution from GitHub...[/bold]")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text)
    console.print(f"[green]Saved to {cache}[/green]")

    return resp.text
