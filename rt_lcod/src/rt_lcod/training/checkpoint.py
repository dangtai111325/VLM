from __future__ import annotations

from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(
    ), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path,
                    *,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Any,
                    scaler: Any,
                    epoch: int,
                    global_step: int,
                    best_metric: float,
                    config: dict[str,
                                 Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch), "global_step": int(global_step), "best_metric": float(best_metric),
        "rng_state": capture_rng_state(), "config": config,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    # Windows antivirus/indexing can hold either checkpoint for a few milliseconds.
    # Keep the atomic replace semantics, but tolerate transient sharing violations so
    # a long local training run is not lost at the checkpoint boundary.
    for attempt in range(5):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))


def load_checkpoint(path: str | Path,
                    *,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler: Any = None,
                    scaler: Any = None,
                    map_location: str | torch.device = "cpu",
                    restore_rng: bool = True) -> dict[str,
                                                      Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])
    return payload
