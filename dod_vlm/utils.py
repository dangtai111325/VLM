from __future__ import annotations

import contextlib
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import random
import time
from typing import Any, Iterator

import numpy as np
import torch

from .config import Config, Paths


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


class RunState:
    def __init__(self, cfg: Config, paths: Paths):
        self.path = paths.run / "run_state.json"
        self.config_hash = cfg.signature()
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "config_hash": self.config_hash,
                "created_at": now_iso(),
                "stages": {},
            }

    def mark(self, stage: str, **meta: Any) -> None:
        self.data["config_hash"] = self.config_hash
        self.data.setdefault("stages", {})[stage] = {
            "completed_at": now_iso(),
            **meta,
        }
        atomic_json_dump(self.data, self.path)

    def done(self, stage: str) -> bool:
        return stage in self.data.get("stages", {})


@contextlib.contextmanager
def stage_timer(name: str) -> Iterator[None]:
    start = time.perf_counter()
    print("\n" + "=" * 92)
    print(f"▶ {name}")
    print(f"Bắt đầu: {now_iso()}")
    print("=" * 92)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"✓ {name} — {elapsed / 60:.2f} phút")


def gpu_report(prefix: str = "") -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    result = {
        "allocated_gb": torch.cuda.memory_allocated() / 2**30,
        "reserved_gb": torch.cuda.memory_reserved() / 2**30,
        "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    print(
        f"{prefix} GPU: allocated={result['allocated_gb']:.2f} GB | "
        f"reserved={result['reserved_gb']:.2f} GB | peak={result['peak_gb']:.2f} GB"
    )
    return result


def stable_probability(key: str, seed: int) -> float:
    import hashlib

    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def config_dict(cfg: Config) -> dict[str, Any]:
    return asdict(cfg)
