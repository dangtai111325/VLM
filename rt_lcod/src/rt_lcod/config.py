from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    visual_dim: int
    text_dim: int
    hidden_dim: int
    relation_feature_dim: int = 11
    dropout: float = 0.1
    top_k: int = 32
    region_encoder: str = "tiny"
    detector_name: str = "yoloe-26m.pt"
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    num_workers: int = 2
    save_every_epochs: int = 1
    early_stopping_patience: int = 8
    kd_temperature: float = 2.0


@dataclass(frozen=True)
class LossConfig:
    target: float = 1.0
    attribute: float = 0.35
    relation: float = 0.35
    no_target: float = 0.5
    distillation: float = 0.5


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    run_name: str
    device: str
    amp: bool
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    validation: dict[str, Any]
    runtime: dict[str, Any]


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        seed=int(raw["seed"]),
        run_name=str(raw.get("run_name", "rt_lcod")),
        device=str(raw.get("device", "cuda")),
        amp=bool(raw.get("amp", True)),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        loss=LossConfig(**raw["loss"]),
        validation=dict(raw.get("validation", {})),
        runtime=dict(raw.get("runtime", {})),
    )


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": config.seed,
        "run_name": config.run_name,
        "device": config.device,
        "amp": config.amp,
        "model": vars(config.model),
        "training": vars(config.training),
        "loss": vars(config.loss),
        "validation": config.validation,
        "runtime": config.runtime,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
