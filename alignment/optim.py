import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model, config):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    decay = [parameter for parameter in parameters if parameter.ndim >= 2]
    no_decay = [parameter for parameter in parameters if parameter.ndim < 2]
    groups = [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

    if config.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=config.learning_rate)
    if config.optimizer == "rmsprop":
        return torch.optim.RMSprop(groups, lr=config.learning_rate)
    if config.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(groups, lr=config.learning_rate)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def build_scheduler(optimizer, total_steps, warmup_ratio, kind):
    warmup_steps = int(total_steps * warmup_ratio)

    def factor(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if kind == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)
