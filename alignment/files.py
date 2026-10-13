import json
import time
from pathlib import Path


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_run_complete(run_dir):
    return (Path(run_dir) / "summary.json").exists()


class MetricsLogger:
    def __init__(self, path, resume=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.elapsed_offset = 0.0
        if resume and self.path.exists():
            rows = read_jsonl(self.path)
            if rows:
                self.elapsed_offset = float(rows[-1].get("elapsed", 0.0))
        else:
            self.path.write_text("", encoding="utf-8")
        self.start = time.time()

    def log(self, **row):
        row["elapsed"] = round(self.elapsed_offset + time.time() - self.start, 1)
        append_jsonl(self.path, row)
