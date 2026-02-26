"""Run evaluation prompts through base and finetuned models via Tinker."""

from __future__ import annotations

from dataclasses import dataclass, field

import tinker
from rich.console import Console
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .prompts import EVAL_PROMPTS, EvalPrompt

console = Console()


@dataclass
class EvalResult:
    """Result of running one eval prompt through a model."""

    prompt: EvalPrompt
    model_label: str
    response: str


@dataclass
class EvalResults:
    """All evaluation results for a model pair."""

    base_results: list[EvalResult] = field(default_factory=list)
    finetuned_results: list[EvalResult] = field(default_factory=list)


def _sample_response(
    client: tinker.SamplingClient,
    renderer,
    messages: list[dict],
) -> str:
    """Generate a response from a SamplingClient."""
    prompt = renderer.build_generation_prompt(messages)
    stop_seqs = renderer.get_stop_sequences()

    response = client.sample(
        model_input=prompt,
        max_tokens=1024,
        temperature=0.3,  # Low temp for more consistent eval
        stop_sequences=stop_seqs,
    )

    msg, _done = renderer.parse_response(response.tokens)
    if isinstance(msg["content"], str):
        return msg["content"]
    return msg["content"][0]["text"]


def get_base_sampling_client(
    model_name: str,
    lora_rank: int,
    tinker_base_url: str | None = None,
) -> tinker.SamplingClient:
    """Create a SamplingClient for the base model (untrained LoRA)."""
    service_kwargs = {}
    if tinker_base_url:
        service_kwargs["base_url"] = tinker_base_url
    service_client = tinker.ServiceClient(**service_kwargs)

    # Create a training client and immediately save — LoRA is zero-initialized
    # so this gives us the base model behavior
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=lora_rank,
    )
    console.print("[dim]Saving base model checkpoint for evaluation...[/dim]")
    base_client = training_client.save_weights_and_get_sampling_client(
        name="base-model-eval"
    )
    return base_client


def run_evaluation(
    base_client: tinker.SamplingClient,
    finetuned_client: tinker.SamplingClient,
    model_name: str,
) -> EvalResults:
    """Run all eval prompts through both models and collect results."""
    tokenizer = get_tokenizer(model_name)
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    results = EvalResults()

    console.print(f"\n[bold]Running {len(EVAL_PROMPTS)} eval prompts...[/bold]\n")

    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        console.print(
            f"[dim]  [{i}/{len(EVAL_PROMPTS)}] {prompt.dimension}/{prompt.name}[/dim]"
        )

        # Run base model
        try:
            base_resp = _sample_response(base_client, renderer, prompt.messages)
        except Exception as e:
            base_resp = f"[ERROR: {e}]"

        results.base_results.append(
            EvalResult(prompt=prompt, model_label="base", response=base_resp)
        )

        # Run finetuned model
        try:
            ft_resp = _sample_response(finetuned_client, renderer, prompt.messages)
        except Exception as e:
            ft_resp = f"[ERROR: {e}]"

        results.finetuned_results.append(
            EvalResult(prompt=prompt, model_label="finetuned", response=ft_resp)
        )

    console.print(f"\n[green]Completed {len(EVAL_PROMPTS)} evaluations[/green]")
    return results
