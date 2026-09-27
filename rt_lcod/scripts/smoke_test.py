from __future__ import annotations

from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader

from rt_lcod.config import load_config
from rt_lcod.data.cached_dataset import collate_cached
from rt_lcod.data.synthetic import SyntheticGroundingDataset
from rt_lcod.geometry import relation_geometry
from rt_lcod.inference.temporal import SimpleTargetTracker
from rt_lcod.models.student import RTLCODStudent
from rt_lcod.prompt import RuleSlotParser
from rt_lcod.training.checkpoint import load_checkpoint
from rt_lcod.training.trainer import Trainer
from rt_lcod.utils.logging import RunLogger
from rt_lcod.utils.seed import seed_everything


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "smoke.yaml")
    seed_everything(config.seed, deterministic=True)

    parsed = RuleSlotParser().parse("the red cup next to the pillow")
    assert parsed.target_class == "cup"
    assert parsed.attribute == "red"
    assert parsed.relation == "next_to"
    assert parsed.reference_class == "pillow"

    geom = relation_geometry(
        torch.tensor([[0.1, 0.1, 0.3, 0.3]]),
        torch.tensor([[0.4, 0.1, 0.6, 0.3]]),
    )
    assert geom.shape == (1, 11)
    assert float(geom[0, 0]) < 0

    train_ds = SyntheticGroundingDataset(
        size=16, visual_dim=config.model.visual_dim, text_dim=config.model.text_dim, max_candidates=6, seed=config.seed,
    )
    val_ds = SyntheticGroundingDataset(
        size=8, visual_dim=config.model.visual_dim, text_dim=config.model.text_dim, max_candidates=6, seed=config.seed + 100,
    )
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_cached)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_cached)

    with tempfile.TemporaryDirectory(prefix="rtlcod-smoke-") as temp:
        model = RTLCODStudent(config.model)
        logger = RunLogger(temp, "smoke")
        trainer = Trainer(model, config, logger)
        best = trainer.fit(train_loader, val_loader)
        assert best.exists()
        restored = RTLCODStudent(config.model)
        state = load_checkpoint(best, model=restored, restore_rng=False)
        assert state["global_step"] > 0
        batch = collate_cached([val_ds[0], val_ds[1]])
        with torch.inference_mode():
            out = restored(batch)
        assert out.logits.shape[0] == 2
        assert out.logits.shape[1] == batch["candidate_mask"].shape[1] + 1

    tracker = SimpleTargetTracker(iou_gate=0.1, max_missed=1)
    state = tracker.update([((10, 10, 20, 20), 0.9)])
    assert state is not None
    same_id = state.track_id
    state = tracker.update([((11, 10, 21, 20), 0.8)])
    assert state is not None and state.track_id == same_id
    print("RT-LCOD smoke test: PASS")


if __name__ == "__main__":
    main()
