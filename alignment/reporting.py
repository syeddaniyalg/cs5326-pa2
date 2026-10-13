from pathlib import Path

import numpy as np
import yaml

from alignment.files import read_json, read_jsonl
from alignment.pairs import candidate_and_baseline
from alignment.text_stats import response_stats


RUNS = [("SFT", "sft"), ("HH-RLHF DPO", "dpo_hh"), ("UltraFeedback DPO", "dpo_uf"), ("LoRA-DPO", "lora_dpo")]

GENERATION_META = {"sft": "sft_alpaca", "dpo_hh": "dpo_hh_hh", "dpo_uf": "dpo_uf_alpaca", "lora_dpo": "lora_persona"}

GUIDE = {
    "sft": {
        "train_size": 30000,
        "val_size": 1000,
        "max_length": 512,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.03,
        "lr_scheduler": "cosine",
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "epochs": 1,
    },
    "dpo": {
        "train_size": 15000,
        "val_size": 1000,
        "max_length": 512,
        "max_prompt_length": 256,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "learning_rate": 1e-6,
        "lr_scheduler": "constant",
        "max_grad_norm": 1.0,
        "beta": 0.1,
        "epochs": 1,
    },
}


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_run(root, name):
    run_dir = Path(root) / name
    config_path = run_dir / "config.yaml"
    summary_path = run_dir / "summary.json"
    config = read_yaml(config_path) if config_path.exists() else None
    summary = read_json(summary_path) if summary_path.exists() else {}
    return config, summary


def decoding_description(root, name):
    meta_path = Path(root) / "generations" / f"{GENERATION_META[name]}.meta.json"
    if not meta_path.exists():
        return ""
    meta = read_json(meta_path)
    mode = "greedy" if meta["temperature"] == 0 else f"T={meta['temperature']}, top_p={meta['top_p']}"
    return f"{mode}, max_new_tokens={meta['max_new_tokens']}, repetition_penalty={meta['repetition_penalty']}, no_repeat_ngram={meta['no_repeat_ngram_size']}, format={meta['prompt_format']}"


def config_rows(root):
    rows = []
    for title, name in RUNS:
        config, summary = load_run(root, name)
        if config is None:
            continue

        is_lora = "lora_r" in config
        if is_lora:
            strategy = f"LoRA r={config['lora_r']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']}, targets={'/'.join(config['target_modules'])}"
            precision = "fp32 weights, fp16 AMP" if config.get("fp16") else "fp32"
            reference = "base model with adapter disabled"
        else:
            strategy = "full fine-tuning"
            precision = "fp32 weights, fp16 autocast, GradScaler"
            reference = config.get("sft_checkpoint", "-")

        per_device = config["per_device_batch_size"]
        accumulation = config["gradient_accumulation_steps"]
        row = {
            "run": title,
            "trainable parameters": strategy,
            "optimizer": config.get("optimizer", ""),
            "peak LR": config["learning_rate"],
            "scheduler": config["lr_scheduler"],
            "warmup": config["warmup_ratio"],
            "weight decay": config["weight_decay"],
            "grad clip": config["max_grad_norm"],
            "per-device batch": per_device,
            "grad accumulation": accumulation,
            "effective batch": per_device * accumulation,
            "precision": precision,
            "max length": config["max_length"],
            "epochs": config["epochs"],
            "optimizer steps": summary.get("total_steps", ""),
            "seed": config["seed"],
            "beta": config.get("beta", "-"),
            "reference checkpoint": reference,
            "decoding": decoding_description(root, name),
        }
        rows.append(row)
    return rows


def deviation_rows(root):
    rows = []
    for title, name in RUNS:
        config, _ = load_run(root, name)
        if config is None or name == "lora_dpo":
            continue

        guide = GUIDE["sft" if name == "sft" else "dpo"]
        for key, guide_value in guide.items():
            used = config.get(key)
            if used is None or used == guide_value:
                continue
            rows.append({"run": title, "parameter": key, "guide value": guide_value, "used value": used, "type": "", "reason and evidence": ""})

    if not rows:
        rows.append({"run": "all", "parameter": "none", "guide value": "-", "used value": "-", "type": "-", "reason and evidence": "No deviations from the guide configuration."})
    return rows


def training_summary_rows(root):
    rows = []
    for title, name in RUNS:
        config, summary = load_run(root, name)
        if not summary:
            continue

        row = {
            "run": title,
            "train examples": summary.get("train_examples", ""),
            "val examples": summary.get("val_examples", ""),
            "optimizer steps": summary.get("total_steps", ""),
            "best step": summary.get("best_step", ""),
            "train minutes": round(summary.get("train_seconds", 0) / 60, 1),
        }
        if name == "sft":
            row["best val loss"] = summary.get("best_val_loss", "")
            row["final val loss"] = summary.get("final_val_loss", "")
        elif name == "lora_dpo":
            final = summary.get("final_eval", {})
            row["final val loss"] = final.get("eval_loss", "")
            row["final val accuracy"] = final.get("eval_rewards/accuracies", "")
            row["final val margin"] = final.get("eval_rewards/margins", "")
            row["trainable parameters"] = summary.get("trainable_parameters", "")
        else:
            best = summary.get("best_val") or {}
            final = summary.get("final_val") or {}
            row["best val loss"] = best.get("loss", "")
            row["best val accuracy"] = best.get("accuracy", "")
            row["best val margin"] = best.get("margin", "")
            row["final val loss"] = final.get("loss", "")
            row["final val accuracy"] = final.get("accuracy", "")
            row["final val margin"] = final.get("margin", "")
        rows.append(row)
    return rows


def result_rows(summaries):
    rows = []
    for summary in summaries:
        counts = summary["counts"]
        low, high = summary["win_rate_ci95"]
        row = {
            "track": summary["track"],
            "candidate": summary["candidate_name"],
            "baseline": summary["baseline_name"],
            "rubric": summary["rubric"],
            "judge": summary["judge_model"],
            "pairs": summary["total"],
            "wins": counts["win"],
            "ties": counts["tie"],
            "losses": counts["loss"],
            "unknown": counts["unknown"],
            "failures": counts["failure"],
            "win rate (no ties)": summary["win_rate_excluding_ties"],
            "win rate (ties = 0.5)": summary["win_rate_ties_half"],
            "95% CI": f"[{low:.3f}, {high:.3f}]",
            "sign-test p": summary["sign_test_p_value"],
            "retried pairs": summary["retried_pairs"],
        }
        rows.append(row)
    return rows


def category_rows(summaries):
    rows = []
    for summary in summaries:
        for category, counts in sorted(summary["by_category"].items()):
            decided = counts["win"] + counts["loss"]
            win_rate = counts["win"] / decided if decided else ""
            rows.append({"track": summary["track"], "category": category, "wins": counts["win"], "ties": counts["tie"], "losses": counts["loss"], "unknown": counts["unknown"], "win rate (no ties)": win_rate})
    return rows


def text_stat_rows(summaries):
    rows = []
    for summary in summaries:
        pairs = read_jsonl(summary["pairs_path"])
        for side, name in ((0, summary["candidate_name"]), (1, summary["baseline_name"])):
            stats = [response_stats(candidate_and_baseline(pair)[side]) for pair in pairs]
            row = {
                "track": summary["track"],
                "model": name,
                "responses": len(stats),
                "mean words": float(np.mean([item["words"] for item in stats])),
                "exclamations / 100 words": float(np.mean([item["exclamations_per_100_words"] for item in stats])),
                "CAPS words / 100 words": float(np.mean([item["caps_words_per_100_words"] for item in stats])),
                "refusal-phrase rate": float(np.mean([item["refusal"] for item in stats])),
            }
            rows.append(row)
    return rows


def shorten(text, limit=350):
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def example_rows(summaries, per_outcome=2):
    rows = []
    for summary in summaries:
        pairs = {pair["pair_id"]: pair for pair in read_jsonl(summary["pairs_path"])}
        judged = read_jsonl(summary["judged_path"])
        latest = {row["pair_id"]: row for row in judged}

        for outcome, label in (("candidate", "win"), ("baseline", "loss"), ("tie", "tie")):
            matching = sorted(pair_id for pair_id, row in latest.items() if row["winner"] == outcome and pair_id in pairs)
            for pair_id in matching[:per_outcome]:
                pair = pairs[pair_id]
                candidate, baseline = candidate_and_baseline(pair)
                row = {
                    "track": summary["track"],
                    "outcome": label,
                    "id": pair_id,
                    "category": pair["category"],
                    "instruction": shorten(pair["instruction"], 200),
                    "candidate response": shorten(candidate),
                    "baseline response": shorten(baseline),
                }
                rows.append(row)
    return rows
