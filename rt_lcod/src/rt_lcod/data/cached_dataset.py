from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = {
    "sample_id", "features", "boxes", "detector_scores", "candidate_mask", "target_index",
    "attribute_embedding", "relation_embedding", "has_attribute", "has_relation", "reference_mask", "target_mask",
}


class CachedGroundingDataset(Dataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.files = sorted(p for p in self.root.glob("*.pt") if not p.name.startswith("_"))
        if not self.files:
            raise ValueError(f"no .pt cache files in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = torch.load(self.files[index], map_location="cpu", weights_only=False)
        missing = REQUIRED_KEYS - set(item)
        if missing:
            raise KeyError(f"{self.files[index]} missing keys: {sorted(missing)}")
        return item


def collate_cached(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_n = max(int(item["features"].shape[0]) for item in batch)
    feat_dim = int(batch[0]["features"].shape[-1])
    device = batch[0]["features"].device

    def pad_tensor(value: torch.Tensor, trailing: tuple[int, ...], fill: float = 0.0):
        out = torch.full((max_n, *trailing), fill, dtype=value.dtype, device=device)
        out[: value.shape[0]] = value
        return out

    features, boxes, scores, masks, refs, targets = [], [], [], [], [], []
    target_indices: list[int] = []
    teacher_distributions: list[torch.Tensor] = []
    teacher_mask: list[bool] = []
    for item in batch:
        n = item["features"].shape[0]
        features.append(pad_tensor(item["features"], (feat_dim,)))
        boxes.append(pad_tensor(item["boxes"], (4,)))
        scores.append(pad_tensor(item["detector_scores"].reshape(n, 1), (1,)).squeeze(-1))
        masks.append(pad_tensor(item["candidate_mask"].float().reshape(n, 1), (1,)).squeeze(-1).bool())
        refs.append(pad_tensor(item["reference_mask"].float().reshape(n, 1), (1,)).squeeze(-1).bool())
        targets.append(pad_tensor(item["target_mask"].float().reshape(n, 1), (1,)).squeeze(-1).bool())
        ti = int(item["target_index"])
        target_indices.append(max_n if ti < 0 else ti)

        has_teacher = "teacher_distribution" in item
        teacher_mask.append(has_teacher)
        padded_teacher = torch.zeros(max_n + 1, dtype=torch.float32)
        if has_teacher:
            dist = item["teacher_distribution"].float()
            candidate_count = min(n, dist.numel() - 1)
            padded_teacher[:candidate_count] = dist[:candidate_count]
            padded_teacher[-1] = dist[-1]
            padded_teacher /= padded_teacher.sum().clamp_min(1e-8)
        else:
            # Placeholder; ignored by teacher_mask in the KD loss.
            padded_teacher[-1] = 1.0
        teacher_distributions.append(padded_teacher)

    result = {
        "sample_id": [str(item["sample_id"]) for item in batch],
        "features": torch.stack(features),
        "boxes": torch.stack(boxes),
        "detector_scores": torch.stack(scores),
        "candidate_mask": torch.stack(masks),
        "target_index": torch.tensor(target_indices, dtype=torch.long),
        "attribute_embedding": torch.stack([item["attribute_embedding"].float() for item in batch]),
        "relation_embedding": torch.stack([item["relation_embedding"].float() for item in batch]),
        "has_attribute": torch.tensor([bool(item["has_attribute"]) for item in batch]),
        "has_relation": torch.tensor([bool(item["has_relation"]) for item in batch]),
        "reference_mask": torch.stack(refs),
        "target_mask": torch.stack(targets),
        "teacher_distribution": torch.stack(teacher_distributions),
        "teacher_mask": torch.tensor(teacher_mask, dtype=torch.bool),
    }
    if all("relation_reference_target_mask" in item for item in batch):
        relation_refs = []
        for item in batch:
            n = item["features"].shape[0]
            relation_refs.append(
                pad_tensor(item["relation_reference_target_mask"].float().reshape(n, 1), (1,)).squeeze(-1).bool()
            )
        result["relation_reference_target_mask"] = torch.stack(relation_refs)
    return result
