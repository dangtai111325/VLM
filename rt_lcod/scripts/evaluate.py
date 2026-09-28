from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rt_lcod.config import load_config
from rt_lcod.data.cached_dataset import CachedGroundingDataset, collate_cached
from rt_lcod.data.synthetic import SyntheticGroundingDataset
from rt_lcod.models.student import RTLCODStudent
from rt_lcod.training.checkpoint import load_checkpoint
from rt_lcod.training.metrics import batch_metrics
from rt_lcod.training.tracker import MetricTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    requested = config.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    device = torch.device(requested)
    if args.synthetic:
        dataset = SyntheticGroundingDataset(
            size=24,
            visual_dim=config.model.visual_dim,
            text_dim=config.model.text_dim,
            seed=config.seed + 2000)
    else:
        if not args.cache:
            raise SystemExit("--cache required unless --synthetic")
        dataset = CachedGroundingDataset(args.cache)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_cached)
    model = RTLCODStudent(config.model).to(device).eval()
    load_checkpoint(args.checkpoint, model=model, map_location=device, restore_rng=False)
    tracker = MetricTracker()
    with torch.inference_mode():
        progress = tqdm(loader, desc="evaluate", unit="batch", dynamic_ncols=True)
        for batch in progress:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            output = model(batch)
            metrics = batch_metrics(output.logits, batch["target_index"])
            tracker.update(metrics, len(batch["target_index"]))
            progress.set_postfix(top1=f"{tracker.averages()['target_top1_accuracy']:.3f}")
    print(json.dumps(tracker.averages(), indent=2))


if __name__ == "__main__":
    main()
