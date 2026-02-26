"""Async orchestrator for generating training data via Claude API through OpenRouter."""

from __future__ import annotations

import asyncio
import json
import os
import random
from collections import defaultdict
from typing import AsyncIterator

from openai import AsyncOpenAI
from rich.console import Console

from ..config import DatagenConfig
from ..constitution.principles import Constitution, Principle, PrincipleCategory
from .categories import ScenarioCategory, get_relevant_categories
from .prompts import GENERATION_PROMPT, SYSTEM_PROMPT

console = Console()


def _build_client(config: DatagenConfig) -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointing at OpenRouter."""
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key: set the {config.api_key_env} environment variable"
        )
    return AsyncOpenAI(
        api_key=api_key,
        base_url=config.api_base_url,
    )


def _build_pairs(
    constitution: Constitution,
    max_per_category: int = 15,
) -> list[tuple[Principle, ScenarioCategory]]:
    """Cross-product each principle with its relevant scenario categories.

    When a principle category has more than max_per_category principles,
    a random sample is taken to keep total data volume manageable.
    """
    # Group principles by category and sample if needed
    by_cat: dict[PrincipleCategory, list[Principle]] = defaultdict(list)
    for p in constitution:
        by_cat[p.category].append(p)

    sampled: list[Principle] = []
    for cat, principles in by_cat.items():
        if len(principles) > max_per_category:
            sampled.extend(random.sample(principles, max_per_category))
        else:
            sampled.extend(principles)

    pairs: list[tuple[Principle, ScenarioCategory]] = []
    for principle in sampled:
        relevant = get_relevant_categories(principle.category)
        for category in relevant:
            pairs.append((principle, category))
    return pairs


async def _generate_one(
    client: AsyncOpenAI,
    principle: Principle,
    category: ScenarioCategory,
    model_name: str,
    config: DatagenConfig,
) -> dict | None:
    """Generate a single training conversation."""
    system = SYSTEM_PROMPT.format(model_name=model_name)
    user_msg = GENERATION_PROMPT.format(
        principle=principle.text,
        category_name=category.name,
        category_description=category.description,
        example_setup=category.example_setup,
    )

    for attempt in range(config.max_retries):
        try:
            response = await client.chat.completions.create(
                model=config.model,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
            )
            text = response.choices[0].message.content.strip()

            # Strip markdown fencing if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[: text.rfind("```")]
                text = text.strip()

            data = json.loads(text)
            if "messages" in data and isinstance(data["messages"], list):
                return data
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt == config.max_retries - 1:
                console.print(
                    f"[yellow]Failed to parse response for "
                    f"{category.name}/{principle.index}: {e}[/yellow]"
                )
        except Exception as e:
            if attempt == config.max_retries - 1:
                console.print(
                    f"[red]API error for {category.name}/{principle.index}: {e}[/red]"
                )
            else:
                await asyncio.sleep(2 ** attempt)

    return None


async def _generate_all(
    constitution: Constitution,
    config: DatagenConfig,
) -> AsyncIterator[dict]:
    """Generate all training examples, yielding conversations."""
    client = _build_client(config)
    pairs = _build_pairs(constitution, max_per_category=config.max_principles_per_category)
    semaphore = asyncio.Semaphore(config.max_concurrency)
    model_name = constitution.target_model_name

    console.print(
        f"[bold]Generating data: {len(pairs)} principle-category pairs, "
        f"{config.examples_per_principle} examples each[/bold]"
    )

    async def _sem_generate(
        principle: Principle, category: ScenarioCategory
    ) -> list[dict]:
        results = []
        for _ in range(config.examples_per_principle):
            async with semaphore:
                result = await _generate_one(
                    client, principle, category, model_name, config
                )
                if result:
                    results.append(result)
        return results

    tasks = [_sem_generate(p, c) for p, c in pairs]
    for coro in asyncio.as_completed(tasks):
        batch = await coro
        for item in batch:
            yield item


async def generate_training_data(
    constitution: Constitution,
    config: DatagenConfig,
) -> list[dict]:
    """Generate all training data and return as a list of conversations.

    Each conversation is a dict with a "messages" key containing a list of
    {"role": str, "content": str} dicts.
    """
    results: list[dict] = []
    count = 0
    async for conversation in _generate_all(constitution, config):
        results.append(conversation)
        count += 1
        if count % 25 == 0:
            console.print(f"[dim]  Generated {count} conversations...[/dim]")

    console.print(f"[green]Generated {len(results)} total conversations[/green]")
    return results
