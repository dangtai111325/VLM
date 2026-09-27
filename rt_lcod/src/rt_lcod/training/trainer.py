from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import time
from typing import Iterable

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from rt_lcod.config import ExperimentConfig, dump_config
from rt_lcod.training.checkpoint import load_checkpoint, save_checkpoint
from rt_lcod.training.losses import compute_loss
from rt_lcod.training.metrics import batch_metrics
from rt_lcod.training.tracker import MetricTracker
from rt_lcod.utils.logging import RunLogger, environment_snapshot
from rt_lcod.utils.timing import TimerBook


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


class Trainer:
    def __init__(self, model: torch.nn.Module, config: ExperimentConfig, run_logger: RunLogger):
        self.model = model
        self.config = config
        self.run_logger = run_logger
        requested = config.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(config.training.epochs, 1))
        amp_enabled = config.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self.global_step = 0
        self.start_epoch = 0
        self.best_metric = float("-inf")
        self.timer = TimerBook()
        self.run_logger.write_json("environment.json", environment_snapshot())
        dump_config(config, self.run_logger.run_dir / "config.yaml")

    def resume(self, checkpoint: str | Path) -> None:
        state = load_checkpoint(
            checkpoint, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            scaler=self.scaler, map_location=self.device,
        )
        self.start_epoch = int(state["epoch"]) + 1
        self.global_step = int(state["global_step"])
        self.best_metric = float(state["best_metric"])

    def _autocast(self):
        if self.amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _run_epoch(self, loader: Iterable[dict], training: bool) -> dict[str, float]:
        self.model.train(training)
        tracker = MetricTracker()
        mode = "train" if training else "val"
        iterator = tqdm(loader, desc=mode, leave=False)
        for batch in iterator:
            with self.timer.measure(f"{mode}_data_to_device"):
                batch = move_batch(batch, self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with self.timer.measure(f"{mode}_forward"):
                with self._autocast():
                    output = self.model(batch)
                    losses = compute_loss(output, batch, self.config.loss, self.config.training.kd_temperature)
            if training:
                with self.timer.measure("train_backward"):
                    self.scaler.scale(losses.total).backward()
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.global_step += 1
            metrics = {**losses.scalar_dict(), **batch_metrics(output.logits.detach(), batch["target_index"])}
            tracker.update(metrics, n=int(batch["target_index"].shape[0]))
            iterator.set_postfix(loss=f"{metrics['loss_total']:.3f}")
        return tracker.averages()

    def _primary_score(self, val: dict[str, float]) -> float:
        target_acc = val.get("target_top1_accuracy", 0.0)
        no_target_acc = val.get("no_target_accuracy", 0.0)
        return 0.75 * target_acc + 0.25 * no_target_acc

    def fit(self, train_loader, val_loader) -> Path:
        no_improvement = 0
        last_path = self.run_logger.checkpoint_dir / "last.pt"
        started = time.perf_counter()
        epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, self.config.training.epochs):
            epoch_started = time.perf_counter()
            train_metrics = self._run_epoch(train_loader, training=True)
            with torch.inference_mode():
                val_metrics = self._run_epoch(val_loader, training=False)
            self.scheduler.step()
            score = self._primary_score(val_metrics)
            improved = score > self.best_metric
            if improved:
                self.best_metric = score
                no_improvement = 0
            else:
                no_improvement += 1

            payload = {
                "epoch": epoch, "global_step": self.global_step, "lr": self.optimizer.param_groups[0]["lr"],
                "primary_score": score, "best_metric": self.best_metric,
                "epoch_s": time.perf_counter() - epoch_started,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.run_logger.log_metrics(payload)
            common = dict(
                model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, scaler=self.scaler,
                epoch=epoch, global_step=self.global_step, best_metric=self.best_metric, config=asdict(self.config),
            )
            save_checkpoint(last_path, **common)
            if (epoch + 1) % self.config.training.save_every_epochs == 0:
                save_checkpoint(self.run_logger.checkpoint_dir / f"epoch_{epoch:04d}.pt", **common)
            if improved:
                save_checkpoint(self.run_logger.checkpoint_dir / "best.pt", **common)
            self.run_logger.write_json("run_state.json", {
                "epoch": epoch, "global_step": self.global_step, "best_metric": self.best_metric,
                "elapsed_s": time.perf_counter() - started, "status": "running",
            })
            if no_improvement >= self.config.training.early_stopping_patience:
                break

        self.run_logger.write_json("timing.json", self.timer.summary())
        self.run_logger.write_json("run_state.json", {
            "epoch": epoch, "global_step": self.global_step, "best_metric": self.best_metric,
            "elapsed_s": time.perf_counter() - started, "status": "complete",
        })
        return self.run_logger.checkpoint_dir / "best.pt"
