"""CLI entrypoint: confine {fetch,generate,train,eval,run}."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console

from .config import PipelineConfig, load_config

console = Console()


def cmd_fetch(config: PipelineConfig) -> None:
    """Download and parse the constitution."""
    from .constitution import fetch_constitution, parse_constitution

    cfg = config.constitution
    markdown = fetch_constitution(cfg.url, cfg.cache_path)
    constitution = parse_constitution(
        markdown,
        target_model_name=cfg.target_model_name,
        source_url=cfg.url,
    )
    console.print(
        f"[green]Parsed {len(constitution)} principles "
        f"across {len(constitution.categories)} categories[/green]"
    )
    for cat in sorted(constitution.categories, key=lambda c: c.value):
        n = len(constitution.by_category(cat))
        console.print(f"  {cat.value}: {n} principles")


def cmd_generate(config: PipelineConfig) -> None:
    """Generate training data from constitutional principles."""
    from .constitution import fetch_constitution, parse_constitution
    from .datagen import generate_training_data, postprocess_and_save

    # Fetch and parse constitution
    cfg_const = config.constitution
    markdown = fetch_constitution(cfg_const.url, cfg_const.cache_path)
    constitution = parse_constitution(
        markdown,
        target_model_name=cfg_const.target_model_name,
        source_url=cfg_const.url,
    )
    console.print(f"[dim]Using {len(constitution)} principles[/dim]")

    # Generate
    cfg_gen = config.datagen
    conversations = asyncio.run(generate_training_data(constitution, cfg_gen))

    # Postprocess and save
    postprocess_and_save(conversations, cfg_gen.output_path)


def cmd_train(config: PipelineConfig) -> None:
    """Run LoRA fine-tuning with Tinker."""
    from .training import train, smoke_test

    base_client, ft_client = train(config.training, data_path=config.datagen.output_path)
    smoke_test(ft_client, config.training.base_model)


def cmd_eval(config: PipelineConfig) -> None:
    """Run evaluation comparing base vs finetuned model."""
    from .eval.compare import generate_report
    from .eval.judge import judge_results
    from .eval.runner import get_base_sampling_client, run_evaluation

    from .training import train

    console.print("[bold]Training model and preparing evaluation...[/bold]\n")

    # Train and get both clients
    base_client, ft_client = train(
        config.training, data_path=config.datagen.output_path
    )

    # Run evaluation prompts through both models
    results = run_evaluation(base_client, ft_client, config.training.base_model)

    # Judge with Claude
    scores = judge_results(results, config.datagen)

    # Generate report
    generate_report(results, scores)


def cmd_run(config: PipelineConfig) -> None:
    """Run the full pipeline: fetch → generate → train → eval."""
    console.print("[bold]Starting full pipeline[/bold]\n")

    console.rule("[bold]Step 1: Fetch Constitution")
    cmd_fetch(config)

    console.rule("[bold]Step 2: Generate Training Data")
    data_file = Path(config.datagen.output_path)
    if data_file.exists() and data_file.stat().st_size > 0:
        console.print(
            f"[green]Training data already exists at {data_file} "
            f"({data_file.stat().st_size // 1024}KB) — skipping generation[/green]"
        )
        console.print("[dim]To regenerate, delete the file and re-run[/dim]")
    else:
        cmd_generate(config)

    console.rule("[bold]Step 3: Train Model")
    from .training import train, smoke_test

    base_client, ft_client = train(
        config.training, data_path=config.datagen.output_path
    )
    smoke_test(ft_client, config.training.base_model)

    console.rule("[bold]Step 4: Evaluate")
    from .eval.compare import generate_report
    from .eval.judge import judge_results
    from .eval.runner import run_evaluation

    results = run_evaluation(base_client, ft_client, config.training.base_model)
    scores = judge_results(results, config.datagen)
    generate_report(results, scores)

    console.print("\n[bold green]Pipeline complete![/bold green]")


def _add_config_arg(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="confine",
        description="Constitutional fine-tuning pipeline",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("fetch", "Download and parse the constitution"),
        ("generate", "Generate training data via Claude API"),
        ("train", "Run LoRA fine-tuning with Tinker"),
        ("eval", "Evaluate base vs finetuned model"),
        ("run", "Run the full pipeline"),
    ]:
        sub = subparsers.add_parser(name, help=help_text)
        _add_config_arg(sub)

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    commands = {
        "fetch": cmd_fetch,
        "generate": cmd_generate,
        "train": cmd_train,
        "eval": cmd_eval,
        "run": cmd_run,
    }
    commands[args.command](config)


if __name__ == "__main__":
    main()
