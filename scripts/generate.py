import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from alignment.files import read_json, write_json
from alignment.generation import run_generation
from alignment.modeling import DTYPES, load_causal_lm, load_tokenizer
from alignment.runtime import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses for a fixed prompt set.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--prompts_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--generator_name", required=True)
    parser.add_argument("--dataset_name", default="alpaca_eval")
    parser.add_argument("--prompt_format", choices=["template", "chat", "plain"], default="template")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_prompt_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dtype", choices=list(DTYPES), default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print_outputs", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    rows = read_json(args.prompts_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer = load_tokenizer(args.model_path)
    model = load_causal_lm(args.model_path, DTYPES[args.dtype], device)
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()

    entries = run_generation(model, tokenizer, rows, args, device, args.output_path, args.generator_name, args.dataset_name)

    meta = {key: value for key, value in vars(args).items()}
    meta["num_prompts"] = len(rows)
    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(Path(args.output_path).with_suffix(".meta.json"), meta)

    if args.print_outputs:
        for entry in tqdm(entries, desc="Print outputs"):
            tqdm.write(f"\n[{entry['id']}] {entry['instruction']}\n--> {entry['output']}")


if __name__ == "__main__":
    main()
