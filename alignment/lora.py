import inspect
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig
from trl import DPOTrainer as TRLDPOTrainer

from alignment.config import save_config
from alignment.files import MetricsLogger, read_json, write_json
from alignment.modeling import load_causal_lm, load_tokenizer
from alignment.runtime import resolve_device, set_seed


TRAIN_KEYS = {
    "loss": "loss",
    "rewards/accuracies": "accuracy",
    "rewards/margins": "margin",
    "rewards/chosen": "chosen_reward",
    "rewards/rejected": "rejected_reward",
    "grad_norm": "grad_norm",
    "learning_rate": "lr",
}

EVAL_KEYS = {
    "eval_loss": "loss",
    "eval_rewards/accuracies": "accuracy",
    "eval_rewards/margins": "margin",
    "eval_rewards/chosen": "chosen_reward",
    "eval_rewards/rejected": "rejected_reward",
}


def to_conversation(row):
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def load_persona_datasets(path, eval_fraction, seed):
    rows = [to_conversation(row) for row in read_json(path)]
    dataset = Dataset.from_list(rows).shuffle(seed=seed)
    eval_size = max(1, int(len(dataset) * eval_fraction))
    split = dataset.train_test_split(test_size=eval_size, shuffle=False)
    return split["train"], split["test"]


class ProgressCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger
        self.bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.bar = tqdm(total=state.max_steps, desc="LoRA DPO")

    def on_step_end(self, args, state, control, **kwargs):
        self.bar.update(1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        is_eval = "eval_loss" in logs
        keys = EVAL_KEYS if is_eval else TRAIN_KEYS
        row = {name: logs[key] for key, name in keys.items() if key in logs}
        if not row:
            return
        split = "val" if is_eval else "train"
        self.logger.log(split=split, step=state.global_step, epoch=state.epoch, **row)
        if self.bar is not None:
            postfix = {key: f"{value:.4g}" for key, value in row.items() if key in ("loss", "accuracy", "margin")}
            self.bar.set_postfix(postfix)

    def on_train_end(self, args, state, control, **kwargs):
        if self.bar is not None:
            self.bar.close()


def build_dpo_config(config):
    values = {
        "output_dir": str(Path(config.run_dir) / "trainer_state"),
        "per_device_train_batch_size": config.per_device_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.epochs,
        "lr_scheduler_type": config.lr_scheduler,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "beta": config.beta,
        "max_length": config.max_length,
        "max_prompt_length": config.max_prompt_length,
        "fp16": bool(config.fp16 and torch.cuda.is_available()),
        "logging_steps": config.log_every_steps,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "report_to": "none",
        "seed": config.seed,
        "remove_unused_columns": False,
        "disable_tqdm": True,
        "optim": config.optimizer,
    }
    parameters = inspect.signature(DPOConfig).parameters
    values["eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"] = "epoch"
    return DPOConfig(**{key: value for key, value in values.items() if key in parameters})


def build_lora_config(config):
    return LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=list(config.target_modules), task_type="CAUSAL_LM")


def run_lora_dpo(config, resume=False):
    set_seed(config.seed)
    start = time.time()
    run_dir = Path(config.run_dir)
    save_config(config, run_dir / "config.yaml")

    device = resolve_device(config.device)
    tokenizer = load_tokenizer(config.model_name)
    model = load_causal_lm(config.model_name, torch.float32, device)
    train_dataset, eval_dataset = load_persona_datasets(config.data_path, config.eval_fraction, config.seed)
    logger = MetricsLogger(run_dir / "metrics.jsonl", resume=resume)

    trainer_parameters = inspect.signature(TRLDPOTrainer.__init__).parameters
    tokenizer_key = "processing_class" if "processing_class" in trainer_parameters else "tokenizer"
    tokenizer_kwargs = {tokenizer_key: tokenizer}
    trainer = TRLDPOTrainer(model=model, ref_model=None, args=build_dpo_config(config), train_dataset=train_dataset, eval_dataset=eval_dataset, peft_config=build_lora_config(config), callbacks=[ProgressCallback(logger)], **tokenizer_kwargs)

    checkpoint = get_last_checkpoint(str(Path(config.run_dir) / "trainer_state")) if resume else None
    trainer.train(resume_from_checkpoint=checkpoint)
    evaluation = trainer.evaluate()

    trainer.save_model(str(run_dir / "adapter"))
    tokenizer.save_pretrained(run_dir / "adapter")

    trainable = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in trainer.model.parameters())

    if config.save_merged:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(run_dir / "merged")
        tokenizer.save_pretrained(run_dir / "merged")

    summary = {
        "train_examples": len(train_dataset),
        "val_examples": len(eval_dataset),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "final_eval": {key: value for key, value in evaluation.items() if isinstance(value, (int, float))},
        "base_model": config.model_name,
        "train_seconds": round(time.time() - start, 1),
    }
    write_json(run_dir / "summary.json", summary)
    return summary
