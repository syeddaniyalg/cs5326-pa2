import copy

import torch

from alignment.batching import build_dpo_batch
from alignment.logprobs import paired_logprobs
from alignment.losses import compute_dpo_loss, compute_sft_loss, dpo_terms, preference_metrics
from tests.toy_lm import ToyCausalLM, ToyTokenizer


PROMPTS = ["Pick: ", "Choose: "]
CHOSEN = ["aaa", "aaa"]
REJECTED = ["bbb", "bbb"]
BETA = 0.1
MAX_LENGTH = 32


def frozen_copy(model):
    reference = copy.deepcopy(model)
    for parameter in reference.parameters():
        parameter.requires_grad = False
    return reference


def reward_metrics(policy, reference, tokenizer):
    batch = build_dpo_batch(tokenizer, PROMPTS, CHOSEN, REJECTED, MAX_LENGTH)
    with torch.no_grad():
        policy_chosen, policy_rejected = paired_logprobs(policy, batch["chosen"], batch["rejected"])
        reference_chosen, reference_rejected = paired_logprobs(reference, batch["chosen"], batch["rejected"])
    _, chosen_rewards, rejected_rewards = dpo_terms(policy_chosen, policy_rejected, reference_chosen, reference_rejected, BETA)
    return preference_metrics(chosen_rewards, rejected_rewards)


def test_a_few_sgd_steps_lower_the_dpo_loss_and_raise_the_reward_margin():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size)
    reference = frozen_copy(policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)

    before = reward_metrics(policy, reference, tokenizer)
    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        loss = compute_dpo_loss(policy, reference, tokenizer, PROMPTS, CHOSEN, REJECTED, BETA, MAX_LENGTH)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    after = reward_metrics(policy, reference, tokenizer)

    assert abs(losses[0] - 0.6931471805599453) < 1e-6
    assert all(later < earlier for earlier, later in zip(losses, losses[1:]))
    assert before["margin"] == 0.0 and before["accuracy"] == 0.0
    assert after["margin"] > 0.0
    assert after["accuracy"] == 1.0
    assert after["chosen_reward"] > 0.0 > after["rejected_reward"]


def test_optimization_steps_leave_the_reference_model_unchanged():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size)
    reference = frozen_copy(policy)
    reference_before = reference.logit_bias.detach().clone()
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)

    for _ in range(5):
        optimizer.zero_grad()
        compute_dpo_loss(policy, reference, tokenizer, PROMPTS, CHOSEN, REJECTED, BETA, MAX_LENGTH).backward()
        optimizer.step()

    assert torch.equal(reference.logit_bias.detach(), reference_before)
    assert reference.logit_bias.grad is None
    assert not torch.equal(policy.logit_bias.detach(), reference_before)


def test_a_few_sgd_steps_lower_the_sft_loss():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        loss = compute_sft_loss(model, tokenizer, PROMPTS, CHOSEN, MAX_LENGTH)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(later < earlier for earlier, later in zip(losses, losses[1:]))
    assert losses[-1] < losses[0]
