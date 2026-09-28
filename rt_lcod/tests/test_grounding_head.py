import torch

from rt_lcod.config import load_config
from rt_lcod.data.cached_dataset import collate_cached
from rt_lcod.data.synthetic import SyntheticGroundingDataset
from rt_lcod.models.student import RTLCODStudent
from rt_lcod.training.losses import compute_loss


def test_forward_and_backward():
    config = load_config("configs/smoke.yaml")
    ds = SyntheticGroundingDataset(
        size=4,
        visual_dim=config.model.visual_dim,
        text_dim=config.model.text_dim,
        seed=13)
    batch = collate_cached([ds[0], ds[1], ds[2]])
    model = RTLCODStudent(config.model)
    output = model(batch)
    assert output.logits.shape == (3, batch["candidate_mask"].shape[1] + 1)
    losses = compute_loss(output, batch, config.loss, config.training.kd_temperature)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
