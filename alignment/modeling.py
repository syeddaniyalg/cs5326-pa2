import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(path, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    return model.to(device)


def freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def save_checkpoint(model, tokenizer, path, dtype):
    snapshot = AutoModelForCausalLM.from_config(model.config)
    snapshot.load_state_dict(model.state_dict())
    snapshot.config.use_cache = True
    snapshot.to(dtype).save_pretrained(path)
    tokenizer.save_pretrained(path)
