from __future__ import annotations

import torch


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for tensors shaped [..., 4] with broadcastable leading dims."""
    x1 = torch.maximum(a[..., 0], b[..., 0])
    y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2])
    y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area_a = (a[..., 2] - a[..., 0]).clamp_min(0) * (a[..., 3] - a[..., 1]).clamp_min(0)
    area_b = (b[..., 2] - b[..., 0]).clamp_min(0) * (b[..., 3] - b[..., 1]).clamp_min(0)
    union = area_a + area_b - inter
    return inter / union.clamp_min(1e-6)


def relation_geometry(target_boxes: torch.Tensor, reference_boxes: torch.Tensor) -> torch.Tensor:
    """Create 11 normalized geometry features for pairwise target/reference boxes."""
    tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)
    rx1, ry1, rx2, ry2 = reference_boxes.unbind(-1)
    tw = (tx2 - tx1).clamp_min(1e-6)
    th = (ty2 - ty1).clamp_min(1e-6)
    rw = (rx2 - rx1).clamp_min(1e-6)
    rh = (ry2 - ry1).clamp_min(1e-6)
    tcx, tcy = (tx1 + tx2) * 0.5, (ty1 + ty2) * 0.5
    rcx, rcy = (rx1 + rx2) * 0.5, (ry1 + ry2) * 0.5
    dx = tcx - rcx
    dy = tcy - rcy
    distance = torch.sqrt(dx.square() + dy.square() + 1e-9)
    iou = box_iou_xyxy(target_boxes, reference_boxes)
    target_area = tw * th
    reference_area = rw * rh
    inter_w = (torch.minimum(tx2, rx2) - torch.maximum(tx1, rx1)).clamp_min(0)
    inter_h = (torch.minimum(ty2, ry2) - torch.maximum(ty1, ry1)).clamp_min(0)
    inter = inter_w * inter_h
    t_contained = inter / target_area.clamp_min(1e-6)
    r_contained = inter / reference_area.clamp_min(1e-6)
    return torch.stack([
        dx, dy, distance,
        torch.log(tw / rw), torch.log(th / rh), torch.log(target_area / reference_area),
        iou, t_contained, r_contained, tcx, tcy,
    ], dim=-1)
