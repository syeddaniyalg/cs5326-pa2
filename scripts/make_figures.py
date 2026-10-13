import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from tqdm import tqdm

from alignment.files import read_json, read_jsonl
from alignment.plotting import DPO_PANELS, SFT_PANELS, plot_category_outcomes, plot_length_distributions, plot_outcomes, plot_refusal_rates, plot_run, plot_style_markers, plot_validation_comparison, plot_win_rates


RUN_PANELS = {"sft": SFT_PANELS, "dpo_hh": DPO_PANELS, "dpo_uf": DPO_PANELS, "lora_dpo": DPO_PANELS}

SMOOTHING = {"sft": 5, "dpo_hh": 10, "dpo_uf": 10, "lora_dpo": 1}


def parse_args():
    parser = argparse.ArgumentParser(description="Save all training and evaluation figures.")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--figures_dir", default=None)
    return parser.parse_args()


def best_step_of(summary):
    return summary.get("best_step") or None


def main():
    args = parse_args()
    root = Path(args.output_root)
    figures_dir = Path(args.figures_dir) if args.figures_dir else root / "figures"
    dpo_runs = {}

    for name, panels in tqdm(RUN_PANELS.items(), desc="Training figures"):
        metrics_path = root / name / "metrics.jsonl"
        if not metrics_path.exists():
            print(f"Skipping {name}: no metrics file")
            continue

        rows = read_jsonl(metrics_path)
        summary_path = root / name / "summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {}
        plot_run(rows, panels, name, figures_dir, best_step_of(summary), SMOOTHING[name])
        if name in ("dpo_hh", "dpo_uf"):
            dpo_runs[name] = rows

    if len(dpo_runs) == 2:
        plot_validation_comparison(dpo_runs, figures_dir)

    summaries = [read_json(path) for path in sorted((root / "eval").glob("*_summary.json"))]
    if not summaries:
        print("No evaluation summaries found.")
        return

    plot_outcomes(summaries, figures_dir)
    plot_win_rates(summaries, figures_dir)

    for summary in tqdm(summaries, desc="Evaluation figures"):
        pairs = read_jsonl(summary["pairs_path"])
        plot_category_outcomes(summary, figures_dir)
        plot_length_distributions(summary, pairs, figures_dir)
        if summary["rubric"] == "safety":
            plot_refusal_rates(summary, pairs, figures_dir)
        if summary["rubric"] in ("persona_style", "persona_quality"):
            plot_style_markers(summary, pairs, figures_dir)


if __name__ == "__main__":
    main()
