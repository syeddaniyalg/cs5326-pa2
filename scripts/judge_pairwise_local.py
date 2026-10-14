import argparse
from pathlib import Path

import torch

from alignment.files import read_jsonl, write_json
from alignment.judge import RUBRICS, judge_pairs, summarize_judgments
from alignment.modeling import DTYPES, load_causal_lm, load_tokenizer
from alignment.runtime import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Judge pairwise comparisons with a local Hugging Face causal LM."
    )
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--judge_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--rubric", choices=list(RUBRICS), default="general")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_chars", type=int, default=1800)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=list(DTYPES), default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse_judgments", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_local_judge(model_name, dtype, device, max_new_tokens):
    tokenizer = load_tokenizer(model_name)
    model = load_causal_lm(model_name, dtype, device)
    model.eval()

    @torch.inference_mode()
    def judge(prompt):
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(device)
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completion = generated[0, encoded["input_ids"].shape[1] :]
        return tokenizer.decode(completion, skip_special_tokens=True).strip()

    return judge


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_path = Path(args.output_path)
    track = output_path.stem.replace("_judged", "")
    failure_path = output_path.parent / f"{track}_failures.jsonl"
    summary_path = output_path.parent / f"{track}_summary.json"

    pairs = read_jsonl(args.pairs_path)
    judge = make_local_judge(
        args.judge_model, DTYPES[args.dtype], device, args.max_new_tokens
    )
    judged = judge_pairs(
        pairs,
        judge,
        args.rubric,
        f"local:{args.judge_model}",
        args.batch_size,
        0.0,
        args.max_retries,
        args.max_chars,
        output_path,
        failure_path,
        args.reuse_judgments,
    )
    summary = summarize_judgments(
        pairs,
        judged,
        track,
        args.rubric,
        f"local:{args.judge_model}",
        args.pairs_path,
        output_path,
    )
    write_json(summary_path, summary)
    print(
        summary["counts"],
        f"win rate (no ties) = {summary['win_rate_excluding_ties']:.3f}",
    )


if __name__ == "__main__":
    main()
