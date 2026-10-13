import torch
import torch.nn.functional as F

from alignment.batching import IGNORE_INDEX, build_dpo_batch, build_sft_batch
from alignment.files import MetricsLogger, read_jsonl
from alignment.logprobs import compute_response_logprobs
from alignment.losses import compute_dpo_loss_from_logps, compute_sft_loss
from alignment.runtime import accumulation_group_size
from tests.toy_lm import ToyTokenizer


def code(text):
    return [ord(char) % 50 + 2 for char in text]


class BareTokenizer:
    def encode(self, text, add_special_tokens=False):
        return code(text)


class CallOnlyTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": code(text)}


class SharedPadEosTokenizer(ToyTokenizer):
    pad_token_id = 1


class ParameterlessLM(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.register_buffer("bias", torch.linspace(-0.3, 0.3, vocab_size))

    def logits(self, input_ids):
        batch, length = input_ids.shape
        return self.bias.view(1, 1, -1).expand(batch, length, -1)

    def forward(self, input_ids, attention_mask=None):
        return self.logits(input_ids)


class TupleOutputLM(ParameterlessLM):
    def forward(self, input_ids, attention_mask=None):
        return (self.logits(input_ids),)


def test_tokenizer_without_pad_or_eos_attributes():
    batch = build_sft_batch(BareTokenizer(), ["ab", "c"], ["de", "f"], max_length=16)

    assert batch["labels"][0].tolist() == [IGNORE_INDEX, IGNORE_INDEX] + code("de")
    assert batch["input_ids"][1, 2:].eq(0).all()
    assert batch["attention_mask"][1].tolist() == [1, 1, 0, 0]


def test_tokenizer_that_only_implements_call():
    batch = build_sft_batch(CallOnlyTokenizer(), ["ab"], ["cd"], max_length=16)

    assert batch["input_ids"][0].tolist() == code("ab") + code("cd") + [1]
    assert batch["labels"][0, -1] == 1


def test_pad_id_equal_to_eos_id_does_not_mask_the_real_eos():
    tokenizer = SharedPadEosTokenizer()
    batch = build_sft_batch(tokenizer, ["a: ", "bbbb: "], ["x", "yy"], max_length=32)

    assert (batch["labels"][0] != IGNORE_INDEX).sum() == 2
    assert batch["labels"][0][batch["labels"][0] != IGNORE_INDEX][-1] == tokenizer.eos_token_id
    assert batch["attention_mask"][0].sum() == 5
    assert batch["input_ids"][0, 5:].eq(tokenizer.pad_token_id).all()


def test_batches_are_long_tensors_with_matching_shapes():
    batch = build_dpo_batch(ToyTokenizer(), ["Q: ", "Task: "], ["yes", "a"], ["no", "bb"], max_length=16)

    for side in ("chosen", "rejected"):
        assert all(value.dtype == torch.long for value in batch[side].values())
        assert batch[side]["input_ids"].shape == batch[side]["labels"].shape == batch[side]["attention_mask"].shape


def test_zero_max_length_does_not_crash():
    batch = build_sft_batch(ToyTokenizer(), ["Q: "], ["yes"], max_length=0)

    assert batch["input_ids"].shape[0] == 1
    assert batch["labels"].eq(IGNORE_INDEX).all()


def test_model_without_parameters_and_tensor_output():
    tokenizer = ToyTokenizer()
    model = ParameterlessLM(tokenizer.vocab_size)
    token_ids = tokenizer.encode("ok", add_special_tokens=False) + [tokenizer.eos_token_id]

    result = compute_response_logprobs(model, tokenizer, ["Q: "], ["ok"], max_length=16)
    expected = F.log_softmax(model.bias, dim=0)[token_ids].sum()

    torch.testing.assert_close(result[0], expected)


def test_model_returning_a_tuple():
    tokenizer = ToyTokenizer()
    model = TupleOutputLM(tokenizer.vocab_size)

    tuple_result = compute_response_logprobs(model, tokenizer, ["Q: "], ["ok"], max_length=16)
    tensor_result = compute_response_logprobs(ParameterlessLM(tokenizer.vocab_size), tokenizer, ["Q: "], ["ok"], max_length=16)

    torch.testing.assert_close(tuple_result, tensor_result)


def test_empty_response_without_eos_scores_zero():
    model = ParameterlessLM(52)

    logps = compute_response_logprobs(model, BareTokenizer(), ["ab"], [""], max_length=8)
    loss = compute_sft_loss(model, BareTokenizer(), ["ab"], [""], max_length=8)

    assert logps.tolist() == [0.0]
    assert loss.item() == 0.0


def test_empty_response_with_eos_scores_only_the_eos():
    tokenizer = ToyTokenizer()
    model = ParameterlessLM(tokenizer.vocab_size)

    logps = compute_response_logprobs(model, tokenizer, ["Q: "], [""], max_length=8)
    expected = F.log_softmax(model.bias, dim=0)[tokenizer.eos_token_id]

    torch.testing.assert_close(logps[0], expected)


def test_dpo_loss_accepts_python_lists():
    loss = compute_dpo_loss_from_logps([-1.0, -2.0], [-2.0, -1.0], [-1.5, -1.5], [-1.5, -1.5], 0.1)
    margin = torch.tensor([-1.0, -2.0]) - torch.tensor([-2.0, -1.0])
    expected = -F.logsigmoid(0.1 * margin).mean()

    torch.testing.assert_close(loss, expected)


def test_dpo_loss_stays_finite_for_extreme_margins():
    zeros = torch.zeros(1)
    good = compute_dpo_loss_from_logps(torch.tensor([1000.0]), torch.tensor([-1000.0]), zeros, zeros, 1.0)
    bad = compute_dpo_loss_from_logps(torch.tensor([-1000.0]), torch.tensor([1000.0]), zeros, zeros, 1.0)

    assert torch.isfinite(good) and good.item() < 1e-6
    assert torch.isfinite(bad) and abs(bad.item() - 2000.0) < 1e-2


def test_dpo_loss_with_zero_beta_is_log_two():
    values = torch.tensor([-3.0, -1.0])
    loss = compute_dpo_loss_from_logps(values, values + 5.0, values - 2.0, values, 0.0)

    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))


def test_dpo_loss_matches_sequence_level_sums_of_token_logps():
    tokenizer = ToyTokenizer()
    model = ParameterlessLM(tokenizer.vocab_size)
    logps = F.log_softmax(model.bias, dim=0)
    chosen_ids = tokenizer.encode("good", add_special_tokens=False) + [tokenizer.eos_token_id]
    rejected_ids = tokenizer.encode("bad", add_special_tokens=False) + [tokenizer.eos_token_id]

    chosen = compute_response_logprobs(model, tokenizer, ["Q: "], ["good"], max_length=32)
    rejected = compute_response_logprobs(model, tokenizer, ["Q: "], ["bad"], max_length=32)

    torch.testing.assert_close(chosen[0], logps[chosen_ids].sum())
    torch.testing.assert_close(rejected[0], logps[rejected_ids].sum())


def test_partial_gradient_accumulation_group_uses_its_actual_size():
    assert [accumulation_group_size(index, 10, 4) for index in range(1, 11)] == [4] * 8 + [2] * 2


def test_metrics_logger_appends_when_resuming(tmp_path):
    path = tmp_path / "metrics.jsonl"
    MetricsLogger(path).log(split="train", step=1)
    MetricsLogger(path, resume=True).log(split="train", step=2)

    rows = read_jsonl(path)
    assert [row["step"] for row in rows] == [1, 2]
    assert rows[1]["elapsed"] >= rows[0]["elapsed"]
