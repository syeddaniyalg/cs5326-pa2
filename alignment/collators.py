from alignment.batching import build_dpo_batch, build_sft_batch
from alignment.prompts import format_prompt


class SFTCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        prompts = [format_prompt(row["instruction"]) for row in rows]
        responses = [row["response"] for row in rows]
        return build_sft_batch(self.tokenizer, prompts, responses, self.max_length)


class DPOCollator:
    def __init__(self, tokenizer, max_length, max_prompt_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def __call__(self, rows):
        prompts = [format_prompt(row["instruction"]) for row in rows]
        chosen = [row["chosen"] for row in rows]
        rejected = [row["rejected"] for row in rows]
        return build_dpo_batch(self.tokenizer, prompts, chosen, rejected, self.max_length, self.max_prompt_length)
