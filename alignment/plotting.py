from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alignment.pairs import candidate_and_baseline
from alignment.text_stats import response_stats


PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
OUTCOME_COLORS = {"win": "#2ca02c", "tie": "#9e9e9e", "loss": "#d62728", "unknown": "#ff7f0e"}

LABELS = {
    "loss": "loss",
    "perplexity": "perplexity",
    "accuracy": "reward accuracy",
    "margin": "reward margin",
    "chosen_reward": "chosen reward",
    "rejected_reward": "rejected reward",
    "grad_norm": "gradient norm",
    "lr": "learning rate",
}

SFT_PANELS = [
    {"name": "loss", "title": "SFT response-token loss", "ylabel": "cross-entropy", "train": ["loss"], "val": ["loss"]},
    {"name": "perplexity", "title": "Validation perplexity", "ylabel": "perplexity", "train": [], "val": ["perplexity"]},
    {"name": "lr", "title": "Learning-rate schedule", "ylabel": "learning rate", "train": ["lr"], "val": []},
    {"name": "grad_norm", "title": "Gradient norm (pre-clip)", "ylabel": "norm", "train": ["grad_norm"], "val": []},
]

DPO_PANELS = [
    {"name": "loss", "title": "DPO loss", "ylabel": "loss", "train": ["loss"], "val": ["loss"]},
    {"name": "accuracy", "title": "Reward accuracy", "ylabel": "accuracy", "train": ["accuracy"], "val": ["accuracy"]},
    {"name": "margin", "title": "Reward margin", "ylabel": "margin", "train": ["margin"], "val": ["margin"]},
    {"name": "rewards", "title": "Implicit rewards", "ylabel": "reward", "train": ["chosen_reward", "rejected_reward"], "val": ["chosen_reward", "rejected_reward"]},
    {"name": "grad_norm", "title": "Gradient norm (pre-clip)", "ylabel": "norm", "train": ["grad_norm"], "val": []},
    {"name": "lr", "title": "Learning rate", "ylabel": "learning rate", "train": ["lr"], "val": []},
]


def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or window <= 1:
        return values
    window = min(window, len(values))
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def series(rows, split, key):
    pairs = [(row["step"], row[key]) for row in rows if row.get("split") == split and row.get(key) is not None]
    if not pairs:
        return np.array([]), np.array([])
    steps, values = zip(*pairs)
    return np.array(steps, dtype=float), np.array(values, dtype=float)


def save_figure(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def draw_panel(axis, rows, panel, best_step, window):
    color_index = 0
    for key in panel["train"]:
        steps, values = series(rows, "train", key)
        if len(steps) == 0:
            continue
        color = PALETTE[color_index % len(PALETTE)]
        color_index += 1
        label = f"train {LABELS[key]}"
        if panel["name"] != "lr":
            axis.plot(steps, values, color=color, alpha=0.2, linewidth=1)
        axis.plot(steps, moving_average(values, window), color=color, linewidth=2, label=f"{label} (smoothed)")

    for key in panel["val"]:
        steps, values = series(rows, "val", key)
        if len(steps) == 0:
            continue
        color = PALETTE[color_index % len(PALETTE)]
        color_index += 1
        axis.plot(steps, values, color=color, marker="o", linewidth=1.5, label=f"val {LABELS[key]}")

    if best_step:
        axis.axvline(best_step, color="black", linestyle="--", alpha=0.6, label=f"best checkpoint (step {best_step})")

    axis.set_title(panel["title"])
    axis.set_xlabel("optimizer step")
    axis.set_ylabel(panel["ylabel"])
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)


def plot_run(rows, panels, name, output_dir, best_step=None, window=10, columns=2):
    output_dir = Path(output_dir)
    count = len(panels)
    grid_rows = int(np.ceil(count / columns))

    figure, axes = plt.subplots(grid_rows, columns, figsize=(6 * columns, 4 * grid_rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, panel in zip(axes, panels):
        draw_panel(axis, rows, panel, best_step, window)
    for axis in axes[count:]:
        axis.axis("off")
    figure.suptitle(f"{name}: training curves", fontsize=14)
    save_figure(figure, output_dir / f"{name}_training_curves.png")

    for panel in panels:
        figure, axis = plt.subplots(figsize=(6, 4))
        draw_panel(axis, rows, panel, best_step, window)
        save_figure(figure, output_dir / f"{name}_{panel['name']}.png")


def plot_validation_comparison(runs, output_dir):
    metrics = [("loss", "Validation DPO loss"), ("accuracy", "Validation reward accuracy"), ("margin", "Validation reward margin")]
    figure, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))

    for axis, (key, title) in zip(axes, metrics):
        for index, (name, rows) in enumerate(runs.items()):
            steps, values = series(rows, "val", key)
            if len(steps):
                axis.plot(steps, values, marker="o", color=PALETTE[index % len(PALETTE)], label=name)
        axis.set_title(title)
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.3)
        axis.legend()

    save_figure(figure, Path(output_dir) / "dpo_validation_comparison.png")


def plot_outcomes(summaries, output_dir):
    names = [summary["track"] for summary in summaries]
    figure, axis = plt.subplots(figsize=(9, 1.0 * len(names) + 2))
    left = np.zeros(len(names))

    for outcome in ["win", "tie", "loss", "unknown"]:
        counts = np.array([summary["counts"][outcome] for summary in summaries], dtype=float)
        totals = np.array([max(1, summary["total"]) for summary in summaries], dtype=float)
        shares = counts / totals
        axis.barh(names, shares, left=left, color=OUTCOME_COLORS[outcome], label=outcome)
        for position, (share, count) in enumerate(zip(shares, counts)):
            if share > 0.04:
                axis.text(left[position] + share / 2, position, int(count), ha="center", va="center", color="white", fontsize=9)
        left += shares

    axis.set_xlim(0, 1)
    axis.set_xlabel("share of judged pairs (candidate perspective)")
    axis.set_title("Pairwise outcomes: candidate vs baseline")
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save_figure(figure, Path(output_dir) / "eval_outcomes.png")


def plot_win_rates(summaries, output_dir):
    names = [summary["track"] for summary in summaries]
    rates = np.array([summary["win_rate_excluding_ties"] for summary in summaries])
    lows = np.array([summary["win_rate_ci95"][0] for summary in summaries])
    highs = np.array([summary["win_rate_ci95"][1] for summary in summaries])

    figure, axis = plt.subplots(figsize=(8, 1.0 * len(names) + 2))
    positions = np.arange(len(names))
    axis.errorbar(rates, positions, xerr=[rates - lows, highs - rates], fmt="o", color=PALETTE[0], capsize=4)
    axis.axvline(0.5, color="black", linestyle="--", alpha=0.6)
    axis.set_yticks(positions)
    axis.set_yticklabels(names)
    axis.set_xlim(0, 1)
    axis.set_xlabel("win rate excluding ties (Wilson 95% interval)")
    axis.set_title("Candidate win rate against baseline")
    axis.grid(alpha=0.3)
    save_figure(figure, Path(output_dir) / "eval_win_rates.png")


def plot_category_outcomes(summary, output_dir):
    categories = sorted(summary["by_category"])
    if len(categories) < 2:
        return

    figure, axis = plt.subplots(figsize=(9, 0.8 * len(categories) + 2))
    left = np.zeros(len(categories))

    for outcome in ["win", "tie", "loss", "unknown"]:
        counts = np.array([summary["by_category"][category][outcome] for category in categories], dtype=float)
        totals = np.array([max(1, sum(summary["by_category"][category].values())) for category in categories], dtype=float)
        shares = counts / totals
        axis.barh(categories, shares, left=left, color=OUTCOME_COLORS[outcome], label=outcome)
        left += shares

    axis.set_xlim(0, 1)
    axis.set_xlabel("share of judged pairs")
    axis.set_title(f"{summary['track']}: outcomes by prompt category")
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save_figure(figure, Path(output_dir) / f"eval_categories_{summary['track']}.png")


def plot_length_distributions(summary, pairs, output_dir):
    candidate_words, baseline_words = [], []
    for pair in pairs:
        candidate, baseline = candidate_and_baseline(pair)
        candidate_words.append(len(candidate.split()))
        baseline_words.append(len(baseline.split()))

    upper = max(candidate_words + baseline_words + [1])
    bins = np.linspace(0, upper, 30)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(baseline_words, bins=bins, alpha=0.6, color=PALETTE[0], label=f"{summary['baseline_name']} (mean {np.mean(baseline_words):.0f})")
    axis.hist(candidate_words, bins=bins, alpha=0.6, color=PALETTE[1], label=f"{summary['candidate_name']} (mean {np.mean(candidate_words):.0f})")
    axis.set_xlabel("response length (words)")
    axis.set_ylabel("number of responses")
    axis.set_title(f"{summary['track']}: response length")
    axis.legend()
    axis.grid(alpha=0.3)
    save_figure(figure, Path(output_dir) / f"eval_length_{summary['track']}.png")


def plot_refusal_rates(summary, pairs, output_dir):
    categories = sorted({pair["category"] for pair in pairs})
    candidate_rates, baseline_rates = [], []

    for category in categories:
        members = [pair for pair in pairs if pair["category"] == category]
        candidate_flags = [response_stats(candidate_and_baseline(pair)[0])["refusal"] for pair in members]
        baseline_flags = [response_stats(candidate_and_baseline(pair)[1])["refusal"] for pair in members]
        candidate_rates.append(float(np.mean(candidate_flags)))
        baseline_rates.append(float(np.mean(baseline_flags)))

    positions = np.arange(len(categories))
    figure, axis = plt.subplots(figsize=(max(7, 1.4 * len(categories)), 4))
    axis.bar(positions - 0.2, baseline_rates, width=0.4, color=PALETTE[0], label=summary["baseline_name"])
    axis.bar(positions + 0.2, candidate_rates, width=0.4, color=PALETTE[1], label=summary["candidate_name"])
    axis.set_xticks(positions)
    axis.set_xticklabels(categories, rotation=30, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("heuristic refusal rate")
    axis.set_title(f"{summary['track']}: refusal-phrase rate by category")
    axis.legend()
    axis.grid(alpha=0.3, axis="y")
    save_figure(figure, Path(output_dir) / f"eval_refusal_{summary['track']}.png")


def plot_style_markers(summary, pairs, output_dir):
    names = ["exclamations_per_100_words", "caps_words_per_100_words"]
    titles = ["exclamation marks per 100 words", "ALL-CAPS words per 100 words"]

    figure, axes = plt.subplots(1, len(names), figsize=(10, 4))
    for axis, name, title in zip(axes, names, titles):
        candidate = [response_stats(candidate_and_baseline(pair)[0])[name] for pair in pairs]
        baseline = [response_stats(candidate_and_baseline(pair)[1])[name] for pair in pairs]
        axis.bar([summary["baseline_name"], summary["candidate_name"]], [np.mean(baseline), np.mean(candidate)], color=[PALETTE[0], PALETTE[1]])
        axis.set_title(title)
        axis.grid(alpha=0.3, axis="y")

    figure.suptitle(f"{summary['track']}: surface style markers")
    save_figure(figure, Path(output_dir) / f"eval_style_markers_{summary['track']}.png")
