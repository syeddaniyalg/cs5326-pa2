import torch
import torch.nn.functional as F

from alignment.batching import IGNORE_INDEX, build_sft_batch, move_to_device, pad_to_width


def model_device(model):
    parameters = getattr(model, "parameters", None)
    parameter = next(iter(parameters()), None) if parameters is not None else None
    return parameter.device if parameter is not None else torch.device("cpu")


def extract_logits(outputs):
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, dict):
        return outputs["logits"]
    logits = getattr(outputs, "logits", None)
    return logits if logits is not None else outputs[0]


def response_logprobs_from_batch(model, batch):
    batch = move_to_device(batch, model_device(model))
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    shift_logits = extract_logits(outputs)[:, :-1]
    shift_labels = batch["labels"][:, 1:]
    mask = shift_labels != IGNORE_INDEX

    selected_logits = shift_logits[mask].float()
    targets = shift_labels[mask]
    token_logps = F.log_softmax(selected_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    rows = torch.arange(mask.shape[0], device=mask.device).unsqueeze(1).expand_as(mask)[mask]
    sums = torch.zeros(mask.shape[0], dtype=token_logps.dtype, device=token_logps.device)
    sums = sums.index_add(0, rows, token_logps)
    counts = mask.sum(dim=1)
    return sums, counts


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    sums, _ = response_logprobs_from_batch(model, batch)
    return sums


def paired_logprobs(model, chosen_batch, rejected_batch):
    width = max(chosen_batch["input_ids"].shape[1], rejected_batch["input_ids"].shape[1])
    pad_values = {"input_ids": 0, "attention_mask": 0, "labels": IGNORE_INDEX}

    merged = {}
    for key, value in pad_values.items():
        merged[key] = torch.cat([pad_to_width(chosen_batch[key], width, value), pad_to_width(rejected_batch[key], width, value)])

    sums, _ = response_logprobs_from_batch(model, merged)
    size = chosen_batch["input_ids"].shape[0]
    return sums[:size], sums[size:]
