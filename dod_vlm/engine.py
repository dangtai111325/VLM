from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .config import Config, Paths
from .evaluation import collect_eval_records, score_threshold_metrics
from .loader import compute_loss, make_loader, model_forward, move_batch
from .model import DODVLM
from .utils import RunState, atomic_torch_save, gpu_report, seed_everything, stage_timer


def choose_batch_size(
    cfg: Config,
    manifest: pd.DataFrame,
    feature_dir: Path,
    tokenizer: Any,
    model: DODVLM,
    device: torch.device,
) -> tuple[int, int]:
    candidates: list[int] = []
    current = cfg.preferred_batch_size
    while current >= cfg.minimum_batch_size:
        if current not in candidates:
            candidates.append(current)
        if current == cfg.minimum_batch_size:
            break
        current = max(cfg.minimum_batch_size, current - 2)

    print("Memory probe batch candidates:", candidates)
    for batch_size in candidates:
        try:
            loader = make_loader(
                cfg, manifest, feature_dir, tokenizer, "train", True, False,
                batch_size, seed=cfg.seed
            )
            raw = next(iter(loader))
            batch = move_batch(raw, device)
            model.train(); model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.amp):
                output = model_forward(model, batch)
                loss, _ = compute_loss(cfg, output, batch)
            loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError("loss không finite trong memory probe")
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            grad_accum = max(1, math.ceil(cfg.target_effective_batch / batch_size))
            print(
                f"✓ batch_size={batch_size} pass; grad_accum={grad_accum}; "
                f"effective_batch≈{batch_size * grad_accum}"
            )
            seed_everything(cfg.seed)
            return batch_size, grad_accum
        except torch.cuda.OutOfMemoryError:
            print(f"⚠ OOM với batch_size={batch_size}; thử nhỏ hơn.")
            model.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
    raise RuntimeError(
        "Không tìm được batch size an toàn. Hãy giảm top_k/d_model hoặc kiểm tra VRAM."
    )


def _save_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def training_signature(cfg: Config, cache_signature: str, batch_size: int, grad_accum: int) -> str:
    payload = {
        "config": asdict(cfg),
        "cache_signature": cache_signature,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "code_version": "dod_vlm_training_v4",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def train_model(
    cfg: Config,
    paths: Paths,
    state: RunState,
    manifest: pd.DataFrame,
    feature_dir: Path,
    cache_signature: str,
    device: torch.device,
) -> tuple[DODVLM, Any, int, int, Path, Path]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model)
    model = DODVLM(cfg, pretrained_text=True).to(device)

    with stage_timer("Session 5A — Preflight forward/backward + adaptive batch"):
        batch_size, grad_accum = choose_batch_size(
            cfg, manifest, feature_dir, tokenizer, model, device
        )
        gpu_report("Memory probe:")

    train_sig = training_signature(cfg, cache_signature, batch_size, grad_accum)
    last_ckpt = paths.checkpoints / "last.pt"
    best_ckpt = paths.checkpoints / "best.pt"
    recovery_ckpt = paths.checkpoints / "recovery.pt"

    text_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
    other_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("text_encoder.") and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": cfg.lr_fusion},
            {"params": text_params, "lr": cfg.lr_text},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    def load_ckpt(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("training_signature") != train_sig:
            print(f"⚠ Bỏ qua {path.name}: training signature không khớp.")
            return None
        return checkpoint

    recovery = load_ckpt(recovery_ckpt)
    latest = recovery if recovery is not None else load_ckpt(last_ckpt)
    resume_source = recovery_ckpt if recovery is not None else last_ckpt

    start_epoch = 0
    resume_batch = 0
    best_metric = -1.0
    patience_left = cfg.early_stopping_patience
    resume_rng: dict[str, Any] | None = None

    if latest is not None:
        print("↻ Resume từ:", resume_source)
        model.load_state_dict(latest["model"])
        optimizer.load_state_dict(latest["optimizer"])
        scheduler.load_state_dict(latest["scheduler"])
        scaler.load_state_dict(latest["scaler"])
        best_metric = float(latest.get("best_metric", -1.0))
        patience_left = int(latest.get("patience_left", cfg.early_stopping_patience))
        resume_rng = latest.get("rng_state")
        if latest.get("kind") == "recovery":
            start_epoch = int(latest["epoch"])
            resume_batch = int(latest["batch_step"]) + 1
        else:
            start_epoch = int(latest["epoch"]) + 1

    history_path = paths.run / "training_history.csv"
    history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    val_loader = make_loader(
        cfg, manifest, feature_dir, tokenizer, "val", False, False, batch_size
    )

    def checkpoint_payload(
        kind: str,
        epoch: int,
        batch_step: int,
        val_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return {
            "kind": kind,
            "training_signature": train_sig,
            "cache_signature": cache_signature,
            "epoch": epoch,
            "batch_step": batch_step,
            "best_metric": best_metric,
            "patience_left": patience_left,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": _save_rng_state(),
            "config": asdict(cfg),
            "val_metrics": val_metrics,
        }

    with stage_timer("Session 6 — Training DOD-VLM"):
        for epoch in range(start_epoch, cfg.epochs):
            train_loader = make_loader(
                cfg, manifest, feature_dir, tokenizer, "train", True, True,
                batch_size, seed=cfg.seed + epoch
            )
            model.train(); optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0; seen_batches = 0
            last_recovery = time.perf_counter(); epoch_start = time.perf_counter()
            progress = tqdm(
                enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch + 1:02d}/{cfg.epochs:02d}", unit="batch"
            )
            rng_restored = False
            for step, raw in progress:
                if epoch == start_epoch and step < resume_batch:
                    continue
                if epoch == start_epoch and resume_batch > 0 and not rng_restored:
                    _restore_rng_state(resume_rng); rng_restored = True

                batch = move_batch(raw, device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.amp):
                    output = model_forward(model, batch)
                    loss, parts = compute_loss(cfg, output, batch)
                    scaled_loss = loss / grad_accum
                scaler.scale(scaled_loss).backward()

                should_step = (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

                seen_batches += 1; running_loss += float(loss.detach())
                progress.set_postfix(
                    loss=f"{running_loss / max(seen_batches, 1):.4f}",
                    cand=f"{parts['candidate']:.3f}", null=f"{parts['null']:.3f}",
                    inj=f"{float(batch['injected'].float().mean()):.2f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

                minutes_since_save = (time.perf_counter() - last_recovery) / 60.0
                if should_step and minutes_since_save >= cfg.save_every_minutes:
                    atomic_torch_save(checkpoint_payload("recovery", epoch, step), recovery_ckpt)
                    last_recovery = time.perf_counter()
                    progress.write(f"💾 Recovery checkpoint @ epoch={epoch + 1}, batch={step + 1}")

            resume_batch = 0; resume_rng = None; recovery_ckpt.unlink(missing_ok=True)
            scheduler.step()

            val_records = collect_eval_records(cfg, model, val_loader, device, show_progress=False)
            val_metrics = score_threshold_metrics(
                val_records, cfg.default_candidate_threshold, cfg.default_null_threshold,
                1.0, 1.0, cfg.ece_bins
            )
            metric = val_metrics["set_f1@0.5"]
            elapsed = time.perf_counter() - epoch_start
            history.append({
                "epoch": epoch,
                "train_loss": running_loss / max(seen_batches, 1),
                "minutes": elapsed / 60.0,
                **val_metrics,
            })
            pd.DataFrame(history).to_csv(history_path, index=False)
            print("Validation:", json.dumps(val_metrics, indent=2)); gpu_report("Training:")

            improved = metric > best_metric
            if improved:
                best_metric = metric; patience_left = cfg.early_stopping_patience
            else:
                patience_left -= 1

            atomic_torch_save(
                checkpoint_payload("epoch", epoch, len(train_loader) - 1, val_metrics),
                last_ckpt,
            )
            if improved:
                shutil.copy2(last_ckpt, best_ckpt)
                print(f"★ New best checkpoint: F1={best_metric:.4f}")
            if patience_left <= 0:
                print("Early stopping."); break

        if not best_ckpt.exists():
            raise RuntimeError("Training kết thúc nhưng best.pt chưa được tạo.")
        state.mark(
            "training_complete", best_metric=best_metric, batch_size=batch_size,
            grad_accum=grad_accum, checkpoint=str(best_ckpt)
        )

    return model, tokenizer, batch_size, grad_accum, best_ckpt, last_ckpt
