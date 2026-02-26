"""Tinker LoRA training loop following the sl_loop.py pattern."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import tinker
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tinker_cookbook.supervised.common import compute_mean_nll

from ..config import TrainingConfig
from .data import load_jsonl, prepare_datums

console = Console()

CHECKPOINT_TTL = 7 * 24 * 3600  # 7 days


def _get_lr(step: int, total_steps: int, config: TrainingConfig) -> float:
    """Linear warmup then linear decay learning rate schedule."""
    warmup_steps = int(total_steps * config.warmup_fraction)
    base_lr = config.adam.learning_rate

    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step + 1) / warmup_steps
    else:
        # Linear decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base_lr * max(0.0, 1.0 - progress)


def train(
    config: TrainingConfig, data_path: str = "data/training_data.jsonl"
) -> tuple[tinker.SamplingClient, tinker.SamplingClient]:
    """Run the full LoRA training loop.

    Returns (base_sampling_client, finetuned_sampling_client).
    """
    # Load and prepare data
    console.print("[bold]Loading training data...[/bold]")
    conversations = load_jsonl(data_path)
    datums = prepare_datums(conversations, config.base_model, config.max_seq_length)

    if not datums:
        raise RuntimeError("No valid training datums — check your training data")

    # Create Tinker client
    console.print(f"[bold]Initializing Tinker client for {config.base_model}...[/bold]")
    service_kwargs = {}
    if config.tinker_base_url:
        service_kwargs["base_url"] = config.tinker_base_url
    service_client = tinker.ServiceClient(**service_kwargs)

    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
        rank=config.lora_rank,
    )

    # Save base model checkpoint before training (LoRA is zero-initialized)
    console.print("[bold]Saving base model checkpoint...[/bold]")
    base_path = training_client.save_weights_for_sampler(
        name=f"{config.run_name}-base", ttl_seconds=CHECKPOINT_TTL
    ).result()
    base_sampling_client = service_client.create_sampling_client(model_path=base_path)
    console.print(f"[dim]Base checkpoint: {base_path}[/dim]")

    # Calculate training schedule
    n_batches_per_epoch = math.ceil(len(datums) / config.batch_size)
    total_steps = n_batches_per_epoch * config.epochs

    console.print(
        f"[bold]Training: {len(datums)} examples, "
        f"{config.batch_size} batch size, "
        f"{config.epochs} epochs, "
        f"{total_steps} total steps[/bold]"
    )

    # Training loop
    global_step = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training", total=total_steps)

        for epoch in range(config.epochs):
            # Shuffle data each epoch
            indices = list(range(len(datums)))
            random.shuffle(indices)

            for batch_start in range(0, len(datums), config.batch_size):
                batch_indices = indices[batch_start : batch_start + config.batch_size]
                batch = [datums[i] for i in batch_indices]

                # Compute learning rate
                lr = _get_lr(global_step, total_steps, config)
                adam_params = tinker.AdamParams(
                    learning_rate=lr,
                    beta1=config.adam.beta1,
                    beta2=config.adam.beta2,
                    eps=config.adam.eps,
                )

                # Forward-backward and optimizer step (pipelined as futures)
                fwd_bwd_future = training_client.forward_backward(
                    batch, loss_fn="cross_entropy"
                )
                optim_future = training_client.optim_step(adam_params)

                # Collect results
                fwd_bwd_result = fwd_bwd_future.result()
                optim_future.result()

                # Compute and log loss
                train_logprobs = [
                    x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs
                ]
                train_weights = [d.loss_fn_inputs["weights"] for d in batch]
                nll = compute_mean_nll(train_logprobs, train_weights)

                global_step += 1
                progress.update(task, advance=1)

                if global_step % 10 == 0:
                    progress.console.print(
                        f"  [dim]Step {global_step}/{total_steps} | "
                        f"Epoch {epoch + 1}/{config.epochs} | "
                        f"Loss: {nll:.4f} | LR: {lr:.2e}[/dim]"
                    )

                # Checkpoint
                if config.save_every_steps and global_step % config.save_every_steps == 0:
                    progress.console.print(
                        f"  [blue]Saving checkpoint at step {global_step}...[/blue]"
                    )
                    training_client.save_weights_for_sampler(
                        name=f"{config.run_name}-step{global_step:06d}",
                        ttl_seconds=CHECKPOINT_TTL,
                    )

    # Save final model
    console.print("[bold green]Training complete! Saving final model...[/bold green]")
    final_path = training_client.save_weights_for_sampler(
        name=f"{config.run_name}-final", ttl_seconds=CHECKPOINT_TTL
    ).result()
    finetuned_sampling_client = service_client.create_sampling_client(model_path=final_path)
    console.print(f"[dim]Final checkpoint: {final_path}[/dim]")

    # Save checkpoint paths locally for later use
    paths_file = Path("data/checkpoint_paths.json")
    paths_file.parent.mkdir(parents=True, exist_ok=True)
    paths_file.write_text(json.dumps({
        "base": base_path,
        "final": final_path,
        "model": config.base_model,
    }, indent=2))
    console.print(f"[dim]Checkpoint paths saved to {paths_file}[/dim]")

    return base_sampling_client, finetuned_sampling_client
