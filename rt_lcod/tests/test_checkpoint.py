from pathlib import Path

import torch

from rt_lcod.config import load_config
from rt_lcod.models.student import RTLCODStudent
from rt_lcod.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path):
    config = load_config("configs/smoke.yaml")
    model = RTLCODStudent(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=3,
        global_step=12,
        best_metric=0.7,
        config={"name": "test"},
    )
    restored = RTLCODStudent(config.model)
    optimizer2 = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1)
    state = load_checkpoint(path, model=restored, optimizer=optimizer2, scheduler=scheduler2, restore_rng=False)
    assert state["epoch"] == 3
    assert state["global_step"] == 12
    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.allclose(a, b)
