from pathlib import Path
from types import SimpleNamespace

import yaml


def load_config(path, overrides=None):
    with open(path, "r", encoding="utf-8") as file:
        values = yaml.safe_load(file)

    for item in overrides or []:
        key, raw_value = item.split("=", 1)
        if key not in values:
            raise KeyError(f"Unknown config key: {key}")
        values[key] = yaml.safe_load(raw_value)

    return SimpleNamespace(**values)


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(vars(config), file, sort_keys=False)
