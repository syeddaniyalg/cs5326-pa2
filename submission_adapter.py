from __future__ import annotations

from alignment import batching, logprobs, losses


def build_sft_batch(tokenizer, prompts, responses, max_length):
    return batching.build_sft_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    return logprobs.compute_response_logprobs(model, tokenizer, prompts, responses, max_length)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    return losses.compute_sft_loss(model, tokenizer, prompts, responses, max_length)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    return batching.build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    return losses.compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta)


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length):
    return losses.compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length)
