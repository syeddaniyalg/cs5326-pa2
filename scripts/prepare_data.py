import argparse
from pathlib import Path

from alignment.data import SOURCES, build_task_splits
from alignment.files import read_json, write_json, write_jsonl


DEFAULT_SIZES = {"sft": (30000, 1000), "hh": (15000, 1000), "ultrafeedback": (15000, 1000)}


def parse_args():
    parser = argparse.ArgumentParser(description="Download and filter the SFT and preference datasets.")
    parser.add_argument("--task", choices=[*SOURCES, "all"], default="all")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--val_size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    tasks = list(SOURCES) if args.task == "all" else [args.task]
    for task in tasks:
        default_train, default_val = DEFAULT_SIZES[task]
        train_size = args.train_size or default_train
        val_size = args.val_size or default_val

        train_rows, val_rows = build_task_splits(task, train_size, val_size, args.seed)
        write_jsonl(output_dir / f"{task}_train.jsonl", train_rows)
        write_jsonl(output_dir / f"{task}_val.jsonl", val_rows)

        source = SOURCES[task]
        manifest[task] = {
            "dataset": source["name"],
            "train_split": source["train_split"],
            "val_split": source["val_split"],
            "seed": args.seed,
            "requested_train": train_size,
            "requested_val": val_size,
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
        }
        print(f"{task}: {len(train_rows)} train, {len(val_rows)} val")

    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
