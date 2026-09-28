from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset


class SyntheticGroundingDataset(Dataset):
    """Small deterministic dataset used only by smoke tests and CPU CI."""

    def __init__(
        self,
        size: int = 32,
        visual_dim: int = 32,
        text_dim: int = 24,
        max_candidates: int = 6,
        seed: int = 7,
        include_teacher: bool = True,
    ):
        self.size = int(size)
        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.max_candidates = int(max_candidates)
        self.seed = int(seed)
        self.include_teacher = bool(include_teacher)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(self.seed + index)
        rng = random.Random(self.seed + index)
        n = rng.randint(3, self.max_candidates)

        features = torch.randn(n, self.visual_dim, generator=generator) * 0.2
        raw_boxes = torch.rand(n, 4, generator=generator)
        x1y1 = raw_boxes[:, :2] * 0.65
        wh = 0.12 + raw_boxes[:, 2:] * 0.22
        boxes = torch.cat([x1y1, (x1y1 + wh).clamp(max=0.98)], dim=-1)
        scores = 0.4 + 0.5 * torch.rand(n, generator=generator)
        candidate_mask = torch.ones(n, dtype=torch.bool)

        no_target = (index % 5) == 0
        target_index = -1 if no_target else rng.randrange(n)
        target_mask = torch.zeros(n, dtype=torch.bool)
        target_candidates = {0, min(1, n - 1)}
        if target_index >= 0:
            target_candidates.add(target_index)
        for candidate_index in target_candidates:
            target_mask[candidate_index] = True

        reference_mask = torch.zeros(n, dtype=torch.bool)
        reference_index = (target_index + 1) % n if target_index >= 0 else n - 1
        reference_mask[reference_index] = True

        has_attribute = (index % 2) == 0
        has_relation = (index % 3) != 0
        attribute = torch.randn(self.text_dim, generator=generator)
        relation = torch.randn(self.text_dim, generator=generator)
        if target_index >= 0:
            signal_dim = min(self.visual_dim, self.text_dim)
            if has_attribute:
                features[target_index, :signal_dim] += attribute[:signal_dim] * 0.6
            if has_relation:
                features[target_index, :signal_dim] += relation[:signal_dim] * 0.25

        relation_reference_target_mask = torch.zeros(n, dtype=torch.bool)
        if has_relation:
            relation_reference_target_mask[reference_index] = True

        item = {
            "sample_id": f"synthetic_{index:04d}",
            "features": features.float(),
            "boxes": boxes.float(),
            "detector_scores": scores.float(),
            "candidate_mask": candidate_mask,
            "target_index": target_index,
            "attribute_embedding": attribute.float(),
            "relation_embedding": relation.float(),
            "has_attribute": has_attribute,
            "has_relation": has_relation,
            "reference_mask": reference_mask,
            "target_mask": target_mask,
            "relation_reference_target_mask": relation_reference_target_mask,
        }
        if self.include_teacher:
            teacher = torch.zeros(n + 1)
            teacher[:-1][target_mask] = 0.02
            teacher[target_index if target_index >= 0 else n] = 0.90
            teacher /= teacher.sum().clamp_min(1e-8)
            item["teacher_distribution"] = teacher
        return item
