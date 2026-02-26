"""Generate comparison report from evaluation results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .runner import EvalResults

console = Console()


def generate_report(
    results: EvalResults,
    scores: list[dict],
    output_path: str = "data/eval_report.json",
) -> Path:
    """Generate and display a comparison report.

    Saves full results to JSON and prints a summary table.
    """
    # Build full report data
    report = {
        "summary": {},
        "by_dimension": {},
        "details": [],
    }

    for base_r, ft_r in zip(results.base_results, results.finetuned_results):
        detail = {
            "dimension": base_r.prompt.dimension,
            "name": base_r.prompt.name,
            "conversation": base_r.prompt.messages,
            "ideal_behavior": base_r.prompt.ideal_behavior,
            "base_response": base_r.response,
            "finetuned_response": ft_r.response,
        }
        report["details"].append(detail)

    # Aggregate scores by dimension
    if scores:
        dim_scores: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for score in scores:
            if "error" in score:
                continue
            dim = score["dimension"]
            for model in ("base", "finetuned"):
                for metric in ("alignment", "helpfulness", "naturalness"):
                    val = score[model].get(metric)
                    if val is not None:
                        dim_scores[dim][f"{model}_{metric}"].append(val)

        # Compute averages
        for dim, metrics in dim_scores.items():
            report["by_dimension"][dim] = {
                k: round(sum(v) / len(v), 2) for k, v in metrics.items()
            }

        # Overall summary
        all_base_align = []
        all_ft_align = []
        all_base_help = []
        all_ft_help = []
        for score in scores:
            if "error" in score:
                continue
            all_base_align.append(score["base"]["alignment"])
            all_ft_align.append(score["finetuned"]["alignment"])
            all_base_help.append(score["base"]["helpfulness"])
            all_ft_help.append(score["finetuned"]["helpfulness"])

        if all_base_align:
            report["summary"] = {
                "n_prompts": len(results.base_results),
                "n_scored": len([s for s in scores if "error" not in s]),
                "base_alignment_avg": round(sum(all_base_align) / len(all_base_align), 2),
                "finetuned_alignment_avg": round(sum(all_ft_align) / len(all_ft_align), 2),
                "alignment_delta": round(
                    sum(all_ft_align) / len(all_ft_align)
                    - sum(all_base_align) / len(all_base_align),
                    2,
                ),
                "base_helpfulness_avg": round(sum(all_base_help) / len(all_base_help), 2),
                "finetuned_helpfulness_avg": round(sum(all_ft_help) / len(all_ft_help), 2),
            }

    # Save to JSON
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"\n[dim]Full report saved to {out}[/dim]")

    # Print summary table
    _print_summary_table(report)
    _print_side_by_side(results)

    return out


def _print_summary_table(report: dict) -> None:
    """Print a summary scores table."""
    if not report.get("by_dimension"):
        console.print("[yellow]No judge scores available — showing responses only[/yellow]")
        return

    table = Table(title="Alignment Scores by Dimension", show_lines=True)
    table.add_column("Dimension", style="bold")
    table.add_column("Base Align", justify="center")
    table.add_column("FT Align", justify="center")
    table.add_column("Delta", justify="center")
    table.add_column("Base Help", justify="center")
    table.add_column("FT Help", justify="center")

    for dim, metrics in sorted(report["by_dimension"].items()):
        b_a = metrics.get("base_alignment", "-")
        f_a = metrics.get("finetuned_alignment", "-")
        delta = round(f_a - b_a, 2) if isinstance(b_a, (int, float)) else "-"
        b_h = metrics.get("base_helpfulness", "-")
        f_h = metrics.get("finetuned_helpfulness", "-")

        delta_style = "green" if isinstance(delta, (int, float)) and delta > 0 else "red" if isinstance(delta, (int, float)) and delta < 0 else "dim"
        table.add_row(
            dim,
            str(b_a),
            str(f_a),
            f"[{delta_style}]{delta:+.2f}[/]" if isinstance(delta, (int, float)) else str(delta),
            str(b_h),
            str(f_h),
        )

    # Overall row
    summary = report.get("summary", {})
    if summary:
        delta = summary.get("alignment_delta", 0)
        delta_style = "green" if delta > 0 else "red" if delta < 0 else "dim"
        table.add_row(
            "[bold]OVERALL[/bold]",
            str(summary.get("base_alignment_avg", "-")),
            str(summary.get("finetuned_alignment_avg", "-")),
            f"[bold {delta_style}]{delta:+.2f}[/]",
            str(summary.get("base_helpfulness_avg", "-")),
            str(summary.get("finetuned_helpfulness_avg", "-")),
        )

    console.print()
    console.print(table)


def _print_side_by_side(results: EvalResults) -> None:
    """Print a few side-by-side response comparisons."""
    console.print("\n[bold]Sample Comparisons[/bold]\n")

    # Show first 5
    for base_r, ft_r in zip(results.base_results[:5], results.finetuned_results[:5]):
        prompt = base_r.prompt
        last_user_msg = prompt.messages[-1]["content"]

        console.rule(f"[bold]{prompt.dimension} / {prompt.name}[/bold]")
        console.print(f"[cyan]User:[/cyan] {last_user_msg}\n")
        console.print(f"[yellow]Base:[/yellow] {base_r.response[:500]}\n")
        console.print(f"[green]Finetuned:[/green] {ft_r.response[:500]}\n")
