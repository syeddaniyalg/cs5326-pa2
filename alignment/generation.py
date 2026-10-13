import torch
from tqdm import tqdm

from alignment.files import read_json, write_json
from alignment.prompts import clean_response, format_prompt
from alignment.runtime import RunningMetrics, format_postfix


def build_prompt(tokenizer, instruction, prompt_format):
    if prompt_format == "template":
        return format_prompt(instruction)
    if prompt_format == "chat":
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return instruction


def count_generated(tokens, eos_token_id):
    if eos_token_id is None:
        return len(tokens), False
    positions = (tokens == eos_token_id).nonzero()
    if len(positions) == 0:
        return len(tokens), False
    return int(positions[0].item()) + 1, True


def generation_kwargs(tokenizer, settings):
    kwargs = {
        "max_new_tokens": settings.max_new_tokens,
        "repetition_penalty": settings.repetition_penalty,
        "no_repeat_ngram_size": settings.no_repeat_ngram_size,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if settings.temperature > 0:
        kwargs.update(do_sample=True, temperature=settings.temperature, top_p=settings.top_p)
    else:
        kwargs.update(do_sample=False)
    return kwargs


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, settings, device):
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=settings.max_prompt_tokens, add_special_tokens=False).to(device)

    sequences = model.generate(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], **generation_kwargs(tokenizer, settings))
    new_tokens = sequences[:, encoded["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    results = []
    for text, tokens in zip(texts, new_tokens):
        length, finished = count_generated(tokens, tokenizer.eos_token_id)
        results.append({"output": clean_response(text), "num_new_tokens": length, "finished": finished})
    return results


def make_entry(row, result, generator_name, dataset_name):
    entry = {
        "id": row["id"],
        "instruction": row["instruction"],
        "output": result["output"],
        "generator": generator_name,
        "dataset": row.get("dataset", dataset_name),
        "num_new_tokens": result["num_new_tokens"],
        "finished": result["finished"],
    }
    if "category" in row:
        entry["category"] = row["category"]
    return entry


def load_existing(path):
    try:
        return {entry["id"]: entry for entry in read_json(path)}
    except (FileNotFoundError, ValueError):
        return {}


def run_generation(model, tokenizer, rows, settings, device, output_path, generator_name, dataset_name):
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model.eval()

    done = load_existing(output_path)
    pending = [row for row in rows if row["id"] not in done]
    pending.sort(key=lambda row: len(row["instruction"]), reverse=True)

    stats = RunningMetrics()
    bar = tqdm(total=len(rows), initial=len(rows) - len(pending), desc=f"Generate {generator_name}")

    for start in range(0, len(pending), settings.batch_size):
        chunk = pending[start : start + settings.batch_size]
        prompts = [build_prompt(tokenizer, row["instruction"], settings.prompt_format) for row in chunk]
        results = generate_batch(model, tokenizer, prompts, settings, device)

        for row, result in zip(chunk, results):
            done[row["id"]] = make_entry(row, result, generator_name, dataset_name)
            stats.update(avg_len=result["num_new_tokens"], eos_rate=float(result["finished"]), empty=float(not result["output"]))

        write_json(output_path, [done[row["id"]] for row in rows if row["id"] in done])
        bar.update(len(chunk))
        bar.set_postfix(format_postfix(stats.compute()))

    bar.close()
    return [done[row["id"]] for row in rows if row["id"] in done]
