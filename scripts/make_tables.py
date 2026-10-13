import argparse
from pathlib import Path

from alignment.files import read_json
from alignment.reporting import category_rows, config_rows, deviation_rows, example_rows, result_rows, text_stat_rows, training_summary_rows
from alignment.tables import write_table


def parse_args():
    parser = argparse.ArgumentParser(description="Save all report tables as CSV, Markdown, and LaTeX.")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--tables_dir", default=None)
    return parser.parse_args()


def ordered_columns(rows):
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def save(rows, directory, stem):
    if not rows:
        print(f"Skipping {stem}: no rows")
        return
    write_table(ordered_columns(rows), rows, directory, stem)
    print(f"Saved {stem} ({len(rows)} rows)")


def main():
    args = parse_args()
    root = Path(args.output_root)
    directory = Path(args.tables_dir) if args.tables_dir else root / "tables"

    save(config_rows(root), directory, "config_table")
    save(deviation_rows(root), directory, "deviations_table")
    save(training_summary_rows(root), directory, "training_summary_table")

    summaries = [read_json(path) for path in sorted((root / "eval").glob("*_summary.json"))]
    save(result_rows(summaries), directory, "results_table")
    save(category_rows(summaries), directory, "category_table")
    save(text_stat_rows(summaries), directory, "text_stats_table")
    save(example_rows(summaries), directory, "examples_table")


if __name__ == "__main__":
    main()
