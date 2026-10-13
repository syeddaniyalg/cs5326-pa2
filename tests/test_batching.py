import torch

from alignment.batching import IGNORE_INDEX, build_dpo_batch, build_sft_batch
from tests.toy_lm import ToyTokenizer


def test_eos_is_dropped_when_truncation_cuts_it():
    tokenizer = ToyTokenizer()
    batch = build_sft_batch(tokenizer, ["ab"], ["cd"], max_length=4)

    assert batch["input_ids"].shape == (1, 4)
    assert tokenizer.eos_token_id not in batch["input_ids"][0].tolist()
    assert batch["labels"][0, 2:].ne(IGNORE_INDEX).all()


def test_eos_is_kept_when_sequence_fits_exactly():
    tokenizer = ToyTokenizer()
    batch = build_sft_batch(tokenizer, ["ab"], ["cd"], max_length=5)

    assert batch["input_ids"].shape == (1, 5)
    assert batch["input_ids"][0, -1] == tokenizer.eos_token_id
    assert batch["labels"][0, -1] == tokenizer.eos_token_id


def test_prompt_longer_than_max_length_leaves_no_supervised_tokens():
    tokenizer = ToyTokenizer()
    batch = build_sft_batch(tokenizer, ["abcdef"], ["g"], max_length=4)

    assert batch["labels"].eq(IGNORE_INDEX).all()
    assert batch["attention_mask"].sum() == 4


def test_rows_keep_their_order_and_are_right_padded():
    tokenizer = ToyTokenizer()
    prompts = ["aaa: ", "b: "]
    responses = ["x", "yyyy"]
    batch = build_sft_batch(tokenizer, prompts, responses, max_length=32)

    for row, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert batch["input_ids"][row, : len(prompt_ids)].tolist() == prompt_ids

    shorter = int(batch["attention_mask"][0].sum())
    assert shorter < batch["input_ids"].shape[1]
    assert batch["input_ids"][0, shorter:].eq(tokenizer.pad_token_id).all()
    assert batch["attention_mask"][0, shorter:].eq(0).all()
    assert batch["labels"][0, shorter:].eq(IGNORE_INDEX).all()


def test_dpo_batches_share_the_same_prompt_tokens():
    tokenizer = ToyTokenizer()
    prompts = ["Task: ", "Q: "]
    batch = build_dpo_batch(tokenizer, prompts, ["good", "yes"], ["bad", "no"], max_length=32)

    for row, prompt in enumerate(prompts):
        length = len(tokenizer.encode(prompt, add_special_tokens=False))
        assert torch.equal(batch["chosen"]["input_ids"][row, :length], batch["rejected"]["input_ids"][row, :length])
        assert batch["chosen"]["labels"][row, :length].eq(IGNORE_INDEX).all()
        assert batch["rejected"]["labels"][row, :length].eq(IGNORE_INDEX).all()


def test_max_prompt_length_keeps_the_prompt_tail():
    tokenizer = ToyTokenizer()
    batch = build_sft_batch(tokenizer, ["abcdef"], ["g"], max_length=32, max_prompt_length=3)

    expected = tokenizer.encode("def", add_special_tokens=False)
    assert batch["input_ids"][0, :3].tolist() == expected
