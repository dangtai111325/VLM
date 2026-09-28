from pathlib import Path

from torch.utils.data import DataLoader

from rt_lcod.config import load_config
from rt_lcod.data.cached_dataset import collate_cached
from rt_lcod.data.synthetic import SyntheticGroundingDataset
from rt_lcod.models.student import RTLCODStudent
from rt_lcod.training.trainer import Trainer
from rt_lcod.utils.logging import RunLogger


def test_trainer_emits_structured_progress_events(tmp_path: Path):
    config = load_config("configs/smoke.yaml")
    dataset = SyntheticGroundingDataset(
        size=4,
        visual_dim=config.model.visual_dim,
        text_dim=config.model.text_dim,
        seed=config.seed,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_cached)
    events = []
    trainer = Trainer(
        RTLCODStudent(config.model),
        config,
        RunLogger(tmp_path, "progress"),
        progress_callback=events.append,
    )

    checkpoint = trainer.fit(loader, loader)

    assert checkpoint.exists()
    assert events[0]["event"] == "started"
    assert any(event["event"] == "batch_completed" for event in events)
    assert any(event["event"] == "epoch_completed" for event in events)
    assert events[-1]["event"] == "completed"
    assert {"device", "global_step", "gpu_memory_peak_mib"} <= set(events[-1])
