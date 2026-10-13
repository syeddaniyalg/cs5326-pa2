import argparse
import random

from huggingface_hub import hf_hub_download

from alignment.files import read_json, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Create the fixed AlpacaEval prompt subset.")
    parser.add_argument("--output_path", default="eval_sets/alpaca_eval_300_seed42.json")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    path = hf_hub_download(repo_id="tatsu-lab/alpaca_eval", filename="alpaca_eval.json", repo_type="dataset")
    rows = read_json(path)

    indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.size))
    subset = [{"id": f"alpaca-eval-{index:04d}", "instruction": rows[index]["instruction"], "dataset": "alpaca_eval", "source_index": index} for index in indices]

    write_json(args.output_path, subset)
    print(f"Saved {len(subset)} of {len(rows)} prompts to {args.output_path}")


if __name__ == "__main__":
    main()
