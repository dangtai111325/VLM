from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass
class Config:
    workspace: str = "./dod_vlm_workspace"
    run_name: str = "dod_vlm_grefcoco_v2"
    seed: int = 1337

    hf_dataset_repo: str = "FudanCVL/gRefCOCO"
    split_file: str = "grefs(unc).json"
    instances_file: str = "instances.json"
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None

    proposal_model: str = "yoloe-26s-seg-pf.pt"
    proposal_imgsz: int = 640
    proposal_conf: float = 0.03
    top_k: int = 64

    roi_output_size: int = 3
    visual_dim: int = 256

    text_model: str = "distilbert-base-uncased"
    max_text_len: int = 48
    unfreeze_last_text_layers: int = 2

    d_model: int = 256
    num_heads: int = 8
    object_layers: int = 2
    cross_layers: int = 2
    ff_dim: int = 768
    dropout: float = 0.10
    relation_geom_dim: int = 12

    epochs: int = 12
    preferred_batch_size: int = 8
    minimum_batch_size: int = 2
    target_effective_batch: int = 24
    lr_fusion: float = 3e-4
    lr_text: float = 2e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 0
    amp: bool = True
    early_stopping_patience: int = 4
    save_every_minutes: int = 15

    positive_iou: float = 0.50
    gt_injection: bool = True
    gt_injection_prob: float = 0.70

    focal_gamma: float = 2.0
    candidate_loss_weight: float = 1.0
    null_loss_weight: float = 0.50
    target_entity_aux_weight: float = 0.20

    threshold_grid_size: int = 31
    default_candidate_threshold: float = 0.50
    default_null_threshold: float = 0.50
    ece_bins: int = 15

    request_timeout: int = 60
    runtime_warmup_runs: int = 3
    runtime_benchmark_runs: int = 20

    def signature(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    annotation: Path
    images: Path
    cache: Path
    feature_cache: Path
    models: Path
    run: Path
    checkpoints: Path
    artifact: Path

    @classmethod
    def from_config(cls, cfg: Config) -> "Paths":
        root = Path(cfg.workspace).resolve()
        data = root / "data"
        run = root / "runs" / cfg.run_name
        paths = cls(
            root=root,
            data=data,
            annotation=data / "grefcoco",
            images=data / "images" / "train2014",
            cache=root / "cache",
            feature_cache=root / "cache" / "image_features",
            models=root / "models",
            run=run,
            checkpoints=run / "checkpoints",
            artifact=run / "final_model",
        )
        for path in [
            paths.annotation,
            paths.images,
            paths.feature_cache,
            paths.models,
            paths.checkpoints,
            paths.artifact,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return paths
