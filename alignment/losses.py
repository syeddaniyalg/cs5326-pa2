import torch
import torch.nn.functional as F

from alignment.batching import build_dpo_batch, build_sft_batch
from alignment.logprobs import paired_logprobs, response_logprobs_from_batch


def sft_nll_and_tokens(model, batch):
    sums, counts = response_logprobs_from_batch(model, batch)
    return -sums.sum(), counts.sum()


def sft_loss_from_batch(model, batch):
    nll, tokens = sft_nll_and_tokens(model, batch)
    return nll / tokens.clamp_min(1)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    return sft_loss_from_batch(model, batch)


def as_float_tensor(values):
    tensor = torch.as_tensor(values)
    return tensor if tensor.is_floating_point() else tensor.float()


def dpo_terms(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    policy_chosen_logps = as_float_tensor(policy_chosen_logps)
    policy_rejected_logps = as_float_tensor(policy_rejected_logps)
    reference_chosen_logps = as_float_tensor(reference_chosen_logps).detach().to(policy_chosen_logps.device)
    reference_rejected_logps = as_float_tensor(reference_rejected_logps).detach().to(policy_rejected_logps.device)

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    losses = -F.logsigmoid(chosen_rewards - rejected_rewards)
    return losses, chosen_rewards, rejected_rewards


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    losses, _, _ = dpo_terms(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta)
    return losses.mean()


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    policy_chosen, policy_rejected = paired_logprobs(policy_model, batch["chosen"], batch["rejected"])

    with torch.no_grad():
        reference_chosen, reference_rejected = paired_logprobs(reference_model, batch["chosen"], batch["rejected"])

    return compute_dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)


def preference_metrics(chosen_rewards, rejected_rewards):
    margins = chosen_rewards - rejected_rewards
    return {
        "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
        "margin": margins.mean().item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }
