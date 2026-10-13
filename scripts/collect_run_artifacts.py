import argparse
import shutil
from pathlib import Path

from tqdm import tqdm


SKIPPED_DIRECTORIES = {"best", "final", "merged", "adapter", "trainer_state"}


def parse_args():
    parser = argparse.ArgumentParser(description="Copy small run artifacts from an attached Kaggle dataset.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", default="outputs")
    parser.add_argument("--max_megabytes", type=float, default=100.0)
    return parser.parse_args()


def is_small_artifact(path, source, limit):
    if not path.is_file():
        return False
    if SKIPPED_DIRECTORIES.intersection(path.relative_to(source).parts):
        return False
    return path.stat().st_size <= limit


def main():
    args = parse_args()
    source = Path(args.source)
    destination = Path(args.destination)
    limit = args.max_megabytes * 2 ** 20

    files = [path for path in source.rglob("*") if is_small_artifact(path, source, limit)]

    for path in tqdm(files, desc="Copy artifacts"):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    print(f"Copied {len(files)} files to {destination}")


if __name__ == "__main__":
    main()
