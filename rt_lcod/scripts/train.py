from __future__ import annotations

import argparse
from pathlib import Path
import re

from torch.utils.data import DataLoader, Dataset
import torch

from rt_lcod.config import load_config
from rt_lcod.data.cached_dataset import CachedGroundingDataset, collate_cached
from rt_lcod.data.synthetic import SyntheticGroundingDataset
from rt_lcod.models.student import RTLCODStudent
from rt_lcod.training.trainer import Trainer
from rt_lcod.utils.logging import RunLogger
from rt_lcod.utils.seed import seed_everything


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


class TeacherMergedDataset(Dataset):
    """Attach teacher targets where available; uncovered samples still train supervised."""

    def __init__(self, base: Dataset, teacher_root: str | Path):
        self.base = base
        self.teacher_root = Path(teacher_root)
        teacher_files = {p.stem for p in self.teacher_root.glob("*.pt") if not p.name.startswith("_")}
        sample_names = []
        for index in range(len(base)):
            item = base[index]
            sample_names.append(safe_name(str(item["sample_id"])))
        covered = sum(name in teacher_files for name in sample_names)
        self.coverage = covered / max(len(sample_names), 1)
        print(
            f"[TRAIN] teacher cache={self.teacher_root.resolve()} covered={covered}/{len(sample_names)} "
            f"({self.coverage:.1%})"
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        path = self.teacher_root / f"{safe_name(str(item['sample_id']))}.pt"
        if path.exists():
            teacher = torch.load(path, map_location="cpu", weights_only=False)
            item["teacher_distribution"] = teacher["teacher_distribution"].float()
        return item


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_cached,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-cache")
    parser.add_argument("--val-cache")
    parser.add_argument("--teacher-cache")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.seed)
    print("[TRAIN] ============================================================")
    print(f"[TRAIN] config={Path(args.config).resolve()}")
    print(f"[TRAIN] device_requested={config.device} cuda_available={torch.cuda.is_available()} amp={config.amp}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"[TRAIN] gpu={props.name} total_vram_gib={props.total_memory / 2**30:.2f}")
    print(
        f"[TRAIN] epochs={config.training.epochs} batch={config.training.batch_size} "
        f"lr={config.training.learning_rate} weight_decay={config.training.weight_decay}"
    )
    print(f"[TRAIN] loss_weights={config.loss}")

    if args.synthetic:
        train_ds = SyntheticGroundingDataset(
            size=40,
            visual_dim=config.model.visual_dim,
            text_dim=config.model.text_dim,
            max_candidates=min(config.model.top_k, 8),
            seed=config.seed,
            include_teacher=True,
        )
        val_ds = SyntheticGroundingDataset(
            size=16,
            visual_dim=config.model.visual_dim,
            text_dim=config.model.text_dim,
            max_candidates=min(config.model.top_k, 8),
            seed=config.seed + 1000,
            include_teacher=True,
        )
    else:
        if not args.train_cache or not args.val_cache:
            raise SystemExit("--train-cache and --val-cache are required unless --synthetic is used")
        train_ds = CachedGroundingDataset(args.train_cache)
        val_ds = CachedGroundingDataset(args.val_cache)
        if args.teacher_cache:
            train_ds = TeacherMergedDataset(train_ds, args.teacher_cache)

    print(f"[TRAIN] train_samples={len(train_ds):,} val_samples={len(val_ds):,}")
    train_loader = make_loader(train_ds, config.training.batch_size, config.training.num_workers, shuffle=True)
    val_loader = make_loader(val_ds, config.training.batch_size, config.training.num_workers, shuffle=False)
    model = RTLCODStudent(config.model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN] student_params={total:,} trainable={trainable:,}")

    logger = RunLogger(args.runs_root, args.run_name or config.run_name)
    trainer = Trainer(model, config, logger)
    if args.resume:
        print(f"[TRAIN] resuming={Path(args.resume).resolve()}")
        trainer.resume(args.resume)
    best = trainer.fit(train_loader, val_loader)
    print(f"[TRAIN] run_dir={logger.run_dir}")
    print(f"[TRAIN] best_checkpoint={best}")
    print("[TRAIN] ============================================================")


if __name__ == "__main__":
    main()
