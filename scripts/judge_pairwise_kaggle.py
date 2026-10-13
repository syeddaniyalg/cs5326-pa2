import argparse
from pathlib import Path

from alignment.files import read_jsonl, write_json
from alignment.judge import RUBRICS, judge_pairs, load_judge, summarize_judgments


def parse_args():
    parser = argparse.ArgumentParser(description="Judge pairwise comparisons with a Kaggle-hosted model.")
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--judge_model", default="google/gemini-3.6-flash")
    parser.add_argument("--rubric", choices=list(RUBRICS), default="general")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=float, default=5.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_chars", type=int, default=2500)
    parser.add_argument("--reuse_judgments", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    track = output_path.stem.replace("_judged", "")
    failure_path = output_path.parent / f"{track}_failures.jsonl"
    summary_path = output_path.parent / f"{track}_summary.json"

    pairs = read_jsonl(args.pairs_path)
    judge = load_judge(args.judge_model)

    sleep_seconds = max(args.sleep_seconds, 5.0)
    judged = judge_pairs(pairs, judge, args.rubric, args.judge_model, args.batch_size, sleep_seconds, args.max_retries, args.max_chars, output_path, failure_path, args.reuse_judgments)

    summary = summarize_judgments(pairs, judged, track, args.rubric, args.judge_model, args.pairs_path, output_path)
    write_json(summary_path, summary)
    print(summary["counts"], f"win rate (no ties) = {summary['win_rate_excluding_ties']:.3f}")


if __name__ == "__main__":
    main()
