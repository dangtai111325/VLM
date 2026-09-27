import torch

from rt_lcod.geometry import box_iou_xyxy, relation_geometry


def test_iou_identity():
    box = torch.tensor([[0.1, 0.1, 0.3, 0.4]])
    assert torch.allclose(box_iou_xyxy(box, box), torch.ones(1))


def test_relation_geometry_direction():
    left = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    right = torch.tensor([[0.6, 0.2, 0.8, 0.4]])
    feat = relation_geometry(left, right)
    assert feat.shape == (1, 11)
    assert feat[0, 0] < 0
    assert abs(float(feat[0, 1])) < 1e-5
