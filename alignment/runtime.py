import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def resolve_reference_device(policy_device, reference_name):
    if policy_device.type != "cuda":
        return policy_device
    if torch.cuda.device_count() < 2:
        return policy_device
    return torch.device(reference_name)


def autocast_context(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def accumulation_group_size(batch_index, total_batches, accumulation_steps):
    """Return the number of microbatches contributing to this optimizer step.

    The final group can be shorter than ``accumulation_steps``. Scaling every
    loss by the configured accumulation count would otherwise underweight that
    final update.
    """
    if batch_index < 1 or batch_index > total_batches:
        raise ValueError("batch_index must be between 1 and total_batches")
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")

    group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, total_batches - group_start)


class RunningMean:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    @property
    def value(self):
        return self.total / max(1, self.count)


class RunningMetrics:
    def __init__(self):
        self.means = {}

    def update(self, **values):
        for name, value in values.items():
            self.means.setdefault(name, RunningMean()).update(value)

    def compute(self):
        return {name: mean.value for name, mean in self.means.items()}

    def reset(self):
        self.means = {}


def format_postfix(values, digits=4):
    return {name: f"{value:.{digits}g}" for name, value in values.items()}
