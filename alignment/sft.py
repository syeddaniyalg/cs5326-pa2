import math
import os
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from alignment.collators import SFTCollator
from alignment.config import save_config
from alignment.files import MetricsLogger, read_jsonl, write_json
from alignment.losses import sft_loss_from_batch, sft_nll_and_tokens
from alignment.modeling import DTYPES, load_causal_lm, load_tokenizer, save_checkpoint
from alignment.optim import build_optimizer, build_scheduler
from alignment.runtime import RunningMetrics, accumulation_group_size, autocast_context, format_postfix, resolve_device, set_seed


class SFTRunner:
    def __init__(self, config, resume=False):
        set_seed(config.seed)
        self.config = config
        self.run_dir = Path(config.run_dir)
        self.resume_dir = self.run_dir / "resume"
        self.resume_state_path = self.resume_dir / "training_state.pt"
        self.resuming = bool(resume and self.resume_state_path.exists() and (self.resume_dir / "model" / "config.json").exists())
        save_config(config, self.run_dir / "config.yaml")

        self.device = resolve_device(config.device)
        model_path = self.resume_dir / "model" if self.resuming else config.model_name
        self.tokenizer = load_tokenizer(model_path)
        self.model = load_causal_lm(model_path, torch.float32, self.device)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

        self.collator = SFTCollator(self.tokenizer, config.max_length)
        self.train_rows = read_jsonl(config.train_path)[: config.train_size]
        self.val_rows = read_jsonl(config.val_path)[: config.val_size]

        self.batches_per_epoch = math.ceil(len(self.train_rows) / config.per_device_batch_size)
        self.val_loader = DataLoader(self.val_rows, batch_size=config.eval_batch_size, shuffle=False, collate_fn=self.collator, num_workers=config.num_workers)

        steps_per_epoch = math.ceil(self.batches_per_epoch / config.gradient_accumulation_steps)
        self.total_steps = steps_per_epoch * config.epochs
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, self.total_steps, config.warmup_ratio, config.lr_scheduler)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        self.logger = MetricsLogger(self.run_dir / "metrics.jsonl", resume=self.resuming)

        self.best_val_loss = float("inf")
        self.best_step = 0
        self.final_val_loss = None
        self.start_epoch = 0
        self.start_batch = 0
        self.start_step = 0
        if self.resuming:
            self.load_training_state()

    def train_loader_for_epoch(self, epoch):
        generator = torch.Generator().manual_seed(self.config.seed + epoch)
        return DataLoader(self.train_rows, batch_size=self.config.per_device_batch_size, shuffle=True, generator=generator, collate_fn=self.collator, num_workers=self.config.num_workers)

    def load_training_state(self):
        state = torch.load(self.resume_state_path, map_location=self.device, weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.start_epoch = int(state["epoch"])
        self.start_batch = int(state["batch"])
        self.start_step = int(state["step"])
        self.best_val_loss = float(state["best_val_loss"])
        self.best_step = int(state["best_step"])
        self.final_val_loss = state.get("final_val_loss")

    def save_training_state(self, epoch, batch, step):
        temporary = self.run_dir / "resume.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        save_checkpoint(self.model, self.tokenizer, temporary / "model", DTYPES[self.config.save_dtype])
        torch.save(
            {
                "epoch": epoch,
                "batch": batch,
                "step": step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "best_step": self.best_step,
                "final_val_loss": self.final_val_loss,
            },
            temporary / "training_state.pt",
        )
        if self.resume_dir.exists():
            shutil.rmtree(self.resume_dir)
        os.replace(temporary, self.resume_dir)

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_nll = 0.0
        total_tokens = 0

        for batch in tqdm(self.val_loader, desc="SFT val", leave=False):
            with autocast_context(self.device):
                nll, tokens = sft_nll_and_tokens(self.model, batch)
            total_nll += nll.item()
            total_tokens += tokens.item()

        self.model.train()
        loss = total_nll / max(1, total_tokens)
        return {"loss": loss, "perplexity": math.exp(min(loss, 20.0))}

    def save(self, name):
        save_checkpoint(self.model, self.tokenizer, self.run_dir / name, DTYPES[self.config.save_dtype])

    def validate_and_checkpoint(self, step, epoch):
        metrics = self.evaluate()
        self.final_val_loss = metrics["loss"]
        self.logger.log(split="val", step=step, epoch=epoch, **metrics)

        if metrics["loss"] < self.best_val_loss:
            self.best_val_loss = metrics["loss"]
            self.best_step = step
            self.save("best")
        return metrics

    def train(self):
        start = time.time()
        accumulation = self.config.gradient_accumulation_steps
        batches_per_epoch = self.batches_per_epoch
        window = RunningMetrics()
        step = self.start_step
        postfix = {}

        completed_batches = self.start_epoch * batches_per_epoch + self.start_batch
        bar = tqdm(total=batches_per_epoch * self.config.epochs, initial=completed_batches, desc="SFT train")
        self.model.train()

        for epoch in range(self.start_epoch, self.config.epochs):
            train_loader = self.train_loader_for_epoch(epoch)
            for index, batch in enumerate(train_loader, start=1):
                if epoch == self.start_epoch and index <= self.start_batch:
                    continue
                with autocast_context(self.device):
                    loss = sft_loss_from_batch(self.model, batch)
                group_size = accumulation_group_size(index, batches_per_epoch, accumulation)
                self.scaler.scale(loss / group_size).backward()
                window.update(loss=loss.item())
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
                    postfix["lr"] = lr
                    postfix["step"] = step
                    bar.set_postfix({**format_postfix(postfix), "step": f"{step}/{self.total_steps}"})

                if step % self.config.eval_every_steps == 0 or step == self.total_steps:
                    metrics = self.validate_and_checkpoint(step, epoch + index / batches_per_epoch)
                    self.save_training_state(epoch, index, step)
                    postfix["val_loss"] = metrics["loss"]
                    postfix["val_ppl"] = metrics["perplexity"]
                    bar.set_postfix({**format_postfix(postfix), "step": f"{step}/{self.total_steps}"})

        bar.close()
        self.save("final")

        summary = {
            "total_steps": step,
            "best_step": self.best_step,
            "best_val_loss": self.best_val_loss,
            "final_val_loss": self.final_val_loss,
            "train_examples": len(self.train_rows),
            "val_examples": len(self.val_rows),
            "effective_batch_size": self.config.per_device_batch_size * accumulation,
            "train_seconds": round(time.time() - start, 1),
        }
        write_json(self.run_dir / "summary.json", summary)
        if self.resume_dir.exists():
            shutil.rmtree(self.resume_dir)
        return summary
