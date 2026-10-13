import argparse

from alignment.files import read_json, write_jsonl
from alignment.pairs import build_pair_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Build randomized A/B pairwise comparison files.")
    parser.add_argument("--candidate_path", required=True)
    parser.add_argument("--baseline_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--candidate_name", required=True)
    parser.add_argument("--baseline_name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = build_pair_rows(read_json(args.candidate_path), read_json(args.baseline_path), args.candidate_name, args.baseline_name, args.seed)
    write_jsonl(args.output_path, pairs)

    candidate_first = sum(pair["a_source"] == "candidate" for pair in pairs)
    print(f"Saved {len(pairs)} pairs to {args.output_path} ({candidate_first} with the candidate as Response A)")


if __name__ == "__main__":
    main()
