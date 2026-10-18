#!/usr/bin/env python3
"""Create Phoenix benchmark and learned-scheduler charts."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


MODES = ["no_draft", "fixed_draft", "learned_bandit"]
MODE_LABELS = {
    "no_draft": "No draft",
    "fixed_draft": "Fixed draft",
    "learned_bandit": "Learned bandit",
}
MODE_ALIASES = {
    "baseline": "no_draft",
    "fixed": "fixed_draft",
    "bandit": "learned_bandit",
    "learned": "learned_bandit",
}
MODEL_LABELS = {"stock": "Stock draft", "fine_tuned": "Fine-tuned draft"}
MODEL_ALIASES = {"finetuned": "fine_tuned", "fine-tuned": "fine_tuned", "tuned": "fine_tuned"}


def normalize(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def load_results(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)

    if isinstance(payload, dict) and isinstance(payload.get("modes"), dict):
        parsed = []
        for name, metrics in payload["modes"].items():
            if name == "target":
                mode, model = "no_draft", "none"
            elif name.startswith("fixed"):
                mode, model = "fixed_draft", "stock"
            elif name == "bandit":
                mode, model = "learned_bandit", "stock"
            else:
                raise ValueError(f"Unknown benchmark mode: {name}")
            parsed.append(
                {
                    "mode": mode,
                    "model": model,
                    "tokens_per_sec": float(metrics["tokens_per_second"]),
                    "p95_latency_ms": float(metrics["p95_latency_s"]) * 1_000,
                }
            )
        return parsed

    rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
    if not isinstance(rows, list):
        raise ValueError("results.json must be a list or contain a 'results' list")

    parsed = []
    for row in rows:
        mode = MODE_ALIASES.get(normalize(row["mode"]), normalize(row["mode"]))
        model = MODEL_ALIASES.get(
            normalize(row.get("draft_model", row.get("model", "stock"))),
            normalize(row.get("draft_model", row.get("model", "stock"))),
        )
        parsed.append(
            {
                "mode": mode,
                "model": model,
                "tokens_per_sec": float(row["tokens_per_sec"]),
                "p95_latency_ms": float(row["p95_latency_ms"]),
            }
        )
    return parsed


def aggregate_results(rows: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"tokens_per_sec": [], "p95_latency_ms": []}
    )
    for row in rows:
        if row["mode"] not in MODES:
            raise ValueError(f"Unknown mode: {row['mode']}")
        grouped[(row["mode"], row["model"])]["tokens_per_sec"].append(row["tokens_per_sec"])
        grouped[(row["mode"], row["model"])]["p95_latency_ms"].append(row["p95_latency_ms"])
    return {
        key: {metric: mean(values) for metric, values in metrics.items()}
        for key, metrics in grouped.items()
    }


def plot_benchmarks(results_path: Path, output_path: Path) -> None:
    data = aggregate_results(load_results(results_path))
    models = [model for model in ("stock", "fine_tuned") if any(key[1] == model for key in data)]
    if not models:
        raise ValueError("No stock or fine_tuned rows found")

    colors = {"stock": "#E76F51", "fine_tuned": "#167D8D"}
    x = list(range(len(MODES)))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)

    for axis, metric, ylabel in (
        (axes[0], "tokens_per_sec", "Tokens / second (higher is better)"),
        (axes[1], "p95_latency_ms", "P95 latency, ms (lower is better)"),
    ):
        baseline = next(
            metrics[metric] for (mode, _), metrics in data.items() if mode == "no_draft"
        )
        baseline_bar = axis.bar(
            x[0], baseline, width, color="#334E58", label="Target only", zorder=3
        )
        axis.bar_label(baseline_bar, fmt="%.1f", padding=3, fontsize=9)

        for model_index, model in enumerate(models):
            positions = [
                position + (model_index - (len(models) - 1) / 2) * width
                for position in x[1:]
            ]
            values = [
                data.get((mode, model), {}).get(metric, float("nan")) for mode in MODES[1:]
            ]
            bars = axis.bar(
                positions,
                values,
                width,
                color=colors[model],
                label=MODEL_LABELS[model],
                zorder=3,
            )
            axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22, zorder=0)

    for axis in axes:
        axis.set_xticks(x, [MODE_LABELS[mode] for mode in MODES])
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Phoenix inference benchmark", fontsize=17, fontweight="bold")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise ValueError(f"Expected one of these CSV columns: {', '.join(candidates)}")


def rolling_mean(values: list[float], window: int) -> list[float]:
    return [mean(values[max(0, index - window + 1) : index + 1]) for index in range(len(values))]


def plot_bandit(csv_path: Path, output_path: Path, window: int) -> None:
    with csv_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        step_column = find_column(
            fieldnames, ("step", "iteration", "request", "request_number", "round")
        )
        setting_column = find_column(
            fieldnames, ("draft_tokens", "chosen_draft_tokens", "setting", "action", "arm")
        )
        rows = list(reader)

    steps = [float(row[step_column]) for row in rows]
    settings = [float(row[setting_column]) for row in rows]
    if not steps:
        raise ValueError("Bandit CSV is empty")
    most_common_setting, most_common_count = Counter(settings).most_common(1)[0]

    fig, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    axis.step(steps, settings, where="post", color="#E9A23B", alpha=0.42, label="Chosen setting")
    axis.plot(
        steps,
        rolling_mean(settings, window),
        color="#0B5963",
        linewidth=2.6,
        label=f"{window}-request rolling mean",
    )
    axis.axhline(
        most_common_setting,
        color="#334E58",
        linestyle="--",
        linewidth=1.4,
        label=f"Most selected: {most_common_setting:g} ({most_common_count}/{len(settings)})",
    )
    axis.set(
        title="Bandit draft-length choices",
        xlabel="Inference request",
        ylabel="Draft tokens selected",
    )
    axis.set_yticks(sorted(set(settings)))
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results.json"))
    parser.add_argument("--bandit-csv", type=Path, default=Path("bandit_log.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("charts"))
    parser.add_argument("--rolling-window", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_benchmarks(args.results, args.output_dir / "benchmark.png")
    plot_bandit(args.bandit_csv, args.output_dir / "bandit_convergence.png", args.rolling_window)
    print(f"Wrote {args.output_dir / 'benchmark.png'}")
    print(f"Wrote {args.output_dir / 'bandit_convergence.png'}")


if __name__ == "__main__":
    main()
