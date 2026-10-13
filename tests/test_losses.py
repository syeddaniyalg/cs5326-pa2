import torch
import torch.nn.functional as F

from alignment.logprobs import compute_response_logprobs, paired_logprobs
from alignment.batching import build_dpo_batch
from alignment.losses import compute_dpo_loss, compute_dpo_loss_from_logps, compute_sft_loss, preference_metrics
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_response_logprobs_match_a_manual_sum():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=1.0)
    token_ids = tokenizer.encode("ok", add_special_tokens=False) + [tokenizer.eos_token_id]

    result = compute_response_logprobs(model, tokenizer, ["Q: "], ["ok"], max_length=16)
    expected = F.log_softmax(model.logit_bias, dim=0)[token_ids].sum()

    torch.testing.assert_close(result[0], expected)


def test_prompt_content_does_not_change_response_logprobs():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)

    short = compute_response_logprobs(model, tokenizer, ["a: "], ["same"], max_length=32)
    long = compute_response_logprobs(model, tokenizer, ["zzzzzzzz: "], ["same"], max_length=32)

    torch.testing.assert_close(short, long)


def test_sft_loss_is_the_token_mean_negative_logprob():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.8)
    prompts = ["Do: ", "Say: "]
    responses = ["a", "longer"]

    loss = compute_sft_loss(model, tokenizer, prompts, responses, max_length=32)
    logps = compute_response_logprobs(model, tokenizer, prompts, responses, max_length=32)
    token_count = sum(len(tokenizer.encode(response, add_special_tokens=False)) + 1 for response in responses)

    torch.testing.assert_close(loss, -logps.sum() / token_count)


def test_dpo_loss_from_logps_matches_the_formula():
    policy_chosen = torch.tensor([-1.0, -3.0, -2.0])
    policy_rejected = torch.tensor([-2.0, -2.5, -4.0])
    reference_chosen = torch.tensor([-1.5, -2.0, -2.0])
    reference_rejected = torch.tensor([-1.0, -2.0, -3.0])
    beta = 0.1

    loss = compute_dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
    margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    expected = -F.logsigmoid(beta * margin).mean()

    torch.testing.assert_close(loss, expected)


def test_dpo_loss_does_not_send_gradients_to_the_reference_logps():
    policy_chosen = torch.tensor([-1.0, -2.0], requires_grad=True)
    policy_rejected = torch.tensor([-2.0, -1.0], requires_grad=True)
    reference_chosen = torch.tensor([-1.0, -1.0], requires_grad=True)
    reference_rejected = torch.tensor([-1.0, -1.0], requires_grad=True)

    loss = compute_dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, 0.3)
    loss.backward()

    assert policy_chosen.grad is not None
    assert reference_chosen.grad is None
    assert reference_rejected.grad is None


def test_dpo_loss_decreases_when_the_policy_prefers_the_chosen_response():
    equal = torch.zeros(4)
    better = compute_dpo_loss_from_logps(equal + 1.0, equal - 1.0, equal, equal, 0.5)
    worse = compute_dpo_loss_from_logps(equal - 1.0, equal + 1.0, equal, equal, 0.5)

    assert better < torch.log(torch.tensor(2.0)) < worse


def test_paired_logprobs_match_separate_scoring():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.9)
    prompts = ["Pick: ", "Choose: "]
    chosen = ["a", "long answer"]
    rejected = ["longer rejected", "b"]

    batch = build_dpo_batch(tokenizer, prompts, chosen, rejected, max_length=48)
    paired_chosen, paired_rejected = paired_logprobs(model, batch["chosen"], batch["rejected"])
    separate_chosen = compute_response_logprobs(model, tokenizer, prompts, chosen, max_length=48)
    separate_rejected = compute_response_logprobs(model, tokenizer, prompts, rejected, max_length=48)

    torch.testing.assert_close(paired_chosen, separate_chosen)
    torch.testing.assert_close(paired_rejected, separate_rejected)


def test_identical_models_give_zero_rewards_and_zero_accuracy():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.4)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.4)
    prompts = ["Pick: "]

    loss = compute_dpo_loss(policy, reference, tokenizer, prompts, ["a"], ["b"], beta=0.1, max_length=16)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))

    metrics = preference_metrics(torch.zeros(3), torch.zeros(3))
    assert metrics["accuracy"] == 0.0
    assert metrics["margin"] == 0.0
