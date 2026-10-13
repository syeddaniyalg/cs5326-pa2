import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


def get_eos_token_id(tokenizer):
    return getattr(tokenizer, "eos_token_id", None)


def get_pad_token_id(tokenizer):
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return pad_token_id
    eos_token_id = get_eos_token_id(tokenizer)
    return eos_token_id if eos_token_id is not None else 0


def read_input_ids(encoded):
    try:
        ids = encoded["input_ids"]
    except (KeyError, TypeError, IndexError):
        ids = getattr(encoded, "input_ids", encoded)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def encode_text(tokenizer, text):
    if hasattr(tokenizer, "encode"):
        return read_input_ids(tokenizer.encode(text, add_special_tokens=False))
    return read_input_ids(tokenizer(text, add_special_tokens=False))


def encode_example(tokenizer, prompt, response, max_length, max_prompt_length=None):
    prompt_ids = encode_text(tokenizer, prompt)
    response_ids = encode_text(tokenizer, response)

    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[len(prompt_ids) - max_prompt_length:]

    eos_token_id = get_eos_token_id(tokenizer)
    if eos_token_id is not None:
        response_ids = response_ids + [eos_token_id]

    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def pad_examples(examples, pad_token_id):
    width = max([len(input_ids) for input_ids, _ in examples], default=1)
    size = len(examples)

    input_ids = torch.full((size, width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((size, width), dtype=torch.long)
    labels = torch.full((size, width), IGNORE_INDEX, dtype=torch.long)

    for row, (ids, label_ids) in enumerate(examples):
        length = len(ids)
        if length == 0:
            continue
        input_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(label_ids, dtype=torch.long)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_sft_batch(tokenizer, prompts, responses, max_length, max_prompt_length=None):
    pad_token_id = get_pad_token_id(tokenizer)
    examples = [encode_example(tokenizer, prompt, response, max_length, max_prompt_length) for prompt, response in zip(prompts, responses)]
    return pad_examples(examples, pad_token_id)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length, max_prompt_length=None):
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length, max_prompt_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length, max_prompt_length),
    }


def pad_to_width(tensor, width, value):
    return F.pad(tensor, (0, width - tensor.shape[1]), value=value)


def move_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}
