import math
import os
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from alignment.collators import DPOCollator
from alignment.config import save_config
from alignment.files import MetricsLogger, read_jsonl, write_json
from alignment.logprobs import paired_logprobs
from alignment.losses import dpo_terms, preference_metrics
from alignment.modeling import DTYPES, freeze, load_causal_lm, load_tokenizer, save_checkpoint
from alignment.optim import build_optimizer, build_scheduler
from alignment.runtime import RunningMetrics, accumulation_group_size, autocast_context, format_postfix, resolve_device, resolve_reference_device, set_seed


class DPORunner:
    def __init__(self, config, resume=False):
        set_seed(config.seed)
        self.config = config
        self.run_dir = Path(config.run_dir)
        self.resume_dir = self.run_dir / "resume"
        self.resume_state_path = self.resume_dir / "training_state.pt"
        self.resuming = bool(resume and self.resume_state_path.exists() and (self.resume_dir / "model" / "config.json").exists())
        save_config(config, self.run_dir / "config.yaml")

        self.policy_device = resolve_device(config.policy_device)
        self.reference_device = resolve_reference_device(self.policy_device, config.reference_device)

        policy_path = self.resume_dir / "model" if self.resuming else config.sft_checkpoint
        self.tokenizer = load_tokenizer(policy_path)
        self.policy = load_causal_lm(policy_path, torch.float32, self.policy_device)
        self.reference = freeze(load_causal_lm(config.sft_checkpoint, torch.float16, self.reference_device))
        if config.gradient_checkpointing:
            self.policy.gradient_checkpointing_enable()
            self.policy.config.use_cache = False

        self.collator = DPOCollator(self.tokenizer, config.max_length, config.max_prompt_length)
        self.train_rows = read_jsonl(config.train_path)[: config.train_size]
        self.val_rows = read_jsonl(config.val_path)[: config.val_size]

        self.batches_per_epoch = math.ceil(len(self.train_rows) / config.per_device_batch_size)
        self.val_loader = DataLoader(self.val_rows, batch_size=config.eval_batch_size, shuffle=False, collate_fn=self.collator, num_workers=config.num_workers)

        steps_per_epoch = math.ceil(self.batches_per_epoch / config.gradient_accumulation_steps)
        self.total_steps = steps_per_epoch * config.epochs
        self.optimizer = build_optimizer(self.policy, config)
        self.scheduler = build_scheduler(self.optimizer, self.total_steps, config.warmup_ratio, config.lr_scheduler)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.policy_device.type == "cuda")
        self.logger = MetricsLogger(self.run_dir / "metrics.jsonl", resume=self.resuming)

        self.reference_cache = None
        self.best_key = (-1.0, float("-inf"))
        self.best_step = 0
        self.best_metrics = None
        self.final_metrics = None
        self.start_epoch = 0
        self.start_batch = 0
        self.start_step = 0
        if self.resuming:
            self.load_training_state()

    def train_loader_for_epoch(self, epoch):
        generator = torch.Generator().manual_seed(self.config.seed + epoch)
        return DataLoader(self.train_rows, batch_size=self.config.per_device_batch_size, shuffle=True, generator=generator, collate_fn=self.collator, num_workers=self.config.num_workers)

    def load_training_state(self):
        state = torch.load(self.resume_state_path, map_location=self.policy_device, weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.start_epoch = int(state["epoch"])
        self.start_batch = int(state["batch"])
        self.start_step = int(state["step"])
        self.best_key = tuple(state["best_key"])
        self.best_step = int(state["best_step"])
        self.best_metrics = state.get("best_metrics")
        self.final_metrics = state.get("final_metrics")

    def save_training_state(self, epoch, batch, step):
        temporary = self.run_dir / "resume.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        save_checkpoint(self.policy, self.tokenizer, temporary / "model", DTYPES[self.config.save_dtype])
        torch.save(
            {
                "epoch": epoch,
                "batch": batch,
                "step": step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_key": self.best_key,
                "best_step": self.best_step,
                "best_metrics": self.best_metrics,
                "final_metrics": self.final_metrics,
            },
            temporary / "training_state.pt",
        )
        if self.resume_dir.exists():
            shutil.rmtree(self.resume_dir)
        os.replace(temporary, self.resume_dir)

    def reference_logps(self, batch):
        with torch.no_grad(), autocast_context(self.reference_device):
            return paired_logprobs(self.reference, batch["chosen"], batch["rejected"])

    def policy_logps(self, batch):
        with autocast_context(self.policy_device):
            return paired_logprobs(self.policy, batch["chosen"], batch["rejected"])

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm)

    @torch.no_grad()
    def evaluate(self):
        self.policy.eval()
        use_cache = self.reference_cache is not None
        cache = []
        losses, chosen_rewards, rejected_rewards = [], [], []

        for index, batch in enumerate(tqdm(self.val_loader, desc="DPO val", leave=False)):
            policy_chosen, policy_rejected = self.policy_logps(batch)
            if use_cache:
                reference_chosen, reference_rejected = self.reference_cache[index]
            else:
                reference_chosen, reference_rejected = self.reference_logps(batch)
                cache.append((reference_chosen.cpu(), reference_rejected.cpu()))

            batch_losses, batch_chosen, batch_rejected = dpo_terms(policy_chosen, policy_rejected, reference_chosen, reference_rejected, self.config.beta)
            losses.append(batch_losses.cpu())
            chosen_rewards.append(batch_chosen.cpu())
            rejected_rewards.append(batch_rejected.cpu())

        if not use_cache:
            self.reference_cache = cache

        self.policy.train()
        losses = torch.cat(losses)
        chosen_rewards = torch.cat(chosen_rewards)
        rejected_rewards = torch.cat(rejected_rewards)
        metrics = preference_metrics(chosen_rewards, rejected_rewards)
        metrics["loss"] = losses.mean().item()
        return metrics

    def save(self, name):
        save_checkpoint(self.policy, self.tokenizer, self.run_dir / name, DTYPES[self.config.save_dtype])

    def validate_and_checkpoint(self, step, epoch):
        metrics = self.evaluate()
        self.final_metrics = metrics
        self.logger.log(split="val", step=step, epoch=epoch, **metrics)

        key = (metrics["accuracy"], metrics["margin"])
        if step > 0 and key > self.best_key:
            self.best_key = key
            self.best_step = step
            self.best_metrics = metrics
            self.save("best")
        return metrics

    def train(self):
        start = time.time()
        accumulation = self.config.gradient_accumulation_steps
        batches_per_epoch = self.batches_per_epoch
        window = RunningMetrics()
        postfix = {}
        step = self.start_step

        if self.config.eval_at_start and step == 0:
            self.validate_and_checkpoint(0, 0.0)

        completed_batches = self.start_epoch * batches_per_epoch + self.start_batch
        bar = tqdm(total=batches_per_epoch * self.config.epochs, initial=completed_batches, desc=f"DPO {Path(self.config.run_dir).name}")
        self.policy.train()

        for epoch in range(self.start_epoch, self.config.epochs):
            train_loader = self.train_loader_for_epoch(epoch)
            for index, batch in enumerate(train_loader, start=1):
                if epoch == self.start_epoch and index <= self.start_batch:
                    continue
                policy_chosen, policy_rejected = self.policy_logps(batch)
                reference_chosen, reference_rejected = self.reference_logps(batch)

                losses, chosen_rewards, rejected_rewards = dpo_terms(policy_chosen, policy_rejected, reference_chosen, reference_rejected, self.config.beta)
                loss = losses.mean()
                group_size = accumulation_group_size(index, batches_per_epoch, accumulation)
                self.scaler.scale(loss / group_size).backward()

                metrics = preference_metrics(chosen_rewards.detach(), rejected_rewards.detach())
                window.update(loss=loss.item(), **metrics)
                bar.update(1)

                if index % accumulation != 0 and index != batches_per_epoch:
                    continue

                grad_norm = self.optimizer_step()
                step += 1
                window.update(grad_norm=grad_norm)

                if step % self.config.log_every_steps == 0:
                    averages = window.compute()
                    window.reset()
                    lr = self.scheduler.get_last_lr()[0]
                    self.logger.log(split="train", step=step, epoch=epoch + index / batches_per_epoch, lr=lr, scale=self.scaler.get_scale(), **averages)
                    postfix.update(averages)
                    postfix.pop("chosen_reward", None)
                    postfix.pop("rejected_reward", None)
                    bar.set_postfix({**format_postfix(postfix), "step": f"{step}/{self.total_steps}"})

                if step % self.config.eval_every_steps == 0 or step == self.total_steps:
                    val = self.validate_and_checkpoint(step, epoch + index / batches_per_epoch)
                    self.save_training_state(epoch, index, step)
                    postfix["val_loss"] = val["loss"]
                    postfix["val_acc"] = val["accuracy"]
                    postfix["val_margin"] = val["margin"]
                    bar.set_postfix({**format_postfix(postfix), "step": f"{step}/{self.total_steps}"})

        bar.close()
        self.save("final")

        summary = {
            "total_steps": step,
            "best_step": self.best_step,
            "best_val": self.best_metrics,
            "final_val": self.final_metrics,
            "train_examples": len(self.train_rows),
            "val_examples": len(self.val_rows),
            "effective_batch_size": self.config.per_device_batch_size * accumulation,
            "beta": self.config.beta,
            "reference_checkpoint": self.config.sft_checkpoint,
            "train_seconds": round(time.time() - start, 1),
        }
        write_json(self.run_dir / "summary.json", summary)
        if self.resume_dir.exists():
            shutil.rmtree(self.resume_dir)
        return summary
