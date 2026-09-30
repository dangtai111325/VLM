from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou

from .config import Config
from .model import DODVLM
from .query import parse_query
from .utils import stable_probability
from .vision import cache_path_for_image


def candidate_labels(boxes: torch.Tensor, gt_boxes: torch.Tensor, threshold: float) -> torch.Tensor:
    if boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.zeros((boxes.shape[0],), dtype=torch.float32)
    return (box_iou(boxes, gt_boxes).max(dim=1).values >= threshold).float()


class CachedDODDataset(Dataset):
    def __init__(self, cfg: Config, manifest: pd.DataFrame, feature_dir: Path, split: str, training: bool) -> None:
        self.cfg = cfg
        self.df = manifest[manifest["split"] == split].reset_index(drop=True)
        self.feature_dir = feature_dir
        self.training = training

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        cache = torch.load(
            cache_path_for_image(self.feature_dir, int(row.image_id)),
            map_location="cpu",
            weights_only=False,
        )
        boxes = cache["boxes"].clone()
        scores = cache["scores"].clone()
        features = cache["features"].clone()
        fallback_mask = cache["fallback_mask"].clone().bool()
        gt_boxes = torch.tensor(row.gt_boxes, dtype=torch.float32).reshape(-1, 4)
        labels = candidate_labels(boxes, gt_boxes, self.cfg.positive_iou)
        output_mask = ~fallback_mask

        injected = False
        should_inject = (
            self.training
            and self.cfg.gt_injection
            and gt_boxes.numel() > 0
            and labels[output_mask].sum() == 0
            and stable_probability(str(row.sample_id), self.cfg.seed) < self.cfg.gt_injection_prob
        )
        if should_inject:
            train_gt_boxes = cache["train_gt_boxes"]
            train_gt_features = cache["train_gt_features"]
            if train_gt_boxes.numel() > 0:
                iou = box_iou(train_gt_boxes, gt_boxes)
                keep_gt = torch.where(iou.max(dim=1).values >= 0.95)[0]
                if len(keep_gt) > 0:
                    boxes = torch.cat([boxes, train_gt_boxes[keep_gt]], dim=0)
                    features = torch.cat([features, train_gt_features[keep_gt]], dim=0)
                    scores = torch.cat([scores, torch.zeros(len(keep_gt))], dim=0)
                    fallback_mask = torch.cat(
                        [fallback_mask, torch.zeros(len(keep_gt), dtype=torch.bool)], dim=0
                    )
                    labels = candidate_labels(boxes, gt_boxes, self.cfg.positive_iou)
                    output_mask = ~fallback_mask
                    injected = True

        if len(boxes) > self.cfg.top_k:
            priority = labels * 10.0 + scores - fallback_mask.float() * 10.0
            keep = torch.topk(priority, k=self.cfg.top_k).indices
            boxes = boxes[keep]
            scores = scores[keep]
            features = features[keep]
            fallback_mask = fallback_mask[keep]
            labels = labels[keep]
            output_mask = output_mask[keep]

        parsed = parse_query(str(row.query))
        return {
            "sample_id": str(row.sample_id),
            "query": str(row.query),
            "target_phrase": parsed["target_phrase"],
            "anchor_phrase": parsed["anchor_phrase"] or "[UNK]",
            "relation_id": int(parsed["relation_id"]),
            "has_relation": bool(parsed["has_relation"]),
            "boxes": boxes,
            "scores": scores,
            "features": features,
            "candidate_y": labels,
            "output_mask": output_mask,
            "gt_boxes": gt_boxes,
            "null_y": float(bool(row.no_target)),
            "image_size": torch.tensor([row.height, row.width], dtype=torch.float32),
            "injected": injected,
            "target_categories": list(row.target_categories),
        }


def _pad_1d(x: torch.Tensor, length: int, value: float = 0.0) -> torch.Tensor:
    out = x.new_full((length,), value)
    out[: min(length, len(x))] = x[:length]
    return out


def _pad_2d(x: torch.Tensor, length: int, width: int, value: float = 0.0) -> torch.Tensor:
    out = x.new_full((length, width), value)
    count = min(length, x.shape[0])
    out[:count] = x[:count]
    return out


def make_collate(cfg: Config, tokenizer: Any):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        k = cfg.top_k
        query_tok = tokenizer(
            [x["query"] for x in batch], padding=True, truncation=True,
            max_length=cfg.max_text_len, return_tensors="pt"
        )
        target_tok = tokenizer(
            [x["target_phrase"] for x in batch], padding=True, truncation=True,
            max_length=16, return_tensors="pt"
        )
        anchor_tok = tokenizer(
            [x["anchor_phrase"] for x in batch], padding=True, truncation=True,
            max_length=16, return_tensors="pt"
        )

        counts = [min(k, len(x["boxes"])) for x in batch]
        candidate_mask = torch.zeros((len(batch), k), dtype=torch.bool)
        output_mask = torch.zeros((len(batch), k), dtype=torch.bool)
        for i, (item, count) in enumerate(zip(batch, counts)):
            candidate_mask[i, :count] = True
            output_mask[i, :count] = item["output_mask"][:count]

        return {
            "sample_id": [x["sample_id"] for x in batch],
            "query": [x["query"] for x in batch],
            "target_categories": [x["target_categories"] for x in batch],
            "visual_features": torch.stack([_pad_2d(x["features"], k, cfg.visual_dim) for x in batch]),
            "boxes": torch.stack([_pad_2d(x["boxes"], k, 4) for x in batch]),
            "detector_scores": torch.stack([_pad_1d(x["scores"], k) for x in batch]),
            "candidate_mask": candidate_mask,
            "output_mask": output_mask,
            "candidate_y": torch.stack([_pad_1d(x["candidate_y"], k) for x in batch]),
            "null_y": torch.tensor([x["null_y"] for x in batch], dtype=torch.float32),
            "image_sizes": torch.stack([x["image_size"] for x in batch]),
            "gt_boxes": [x["gt_boxes"] for x in batch],
            "query_ids": query_tok["input_ids"],
            "query_mask": query_tok["attention_mask"],
            "target_ids": target_tok["input_ids"],
            "target_text_mask": target_tok["attention_mask"],
            "anchor_ids": anchor_tok["input_ids"],
            "anchor_text_mask": anchor_tok["attention_mask"],
            "relation_ids": torch.tensor([x["relation_id"] for x in batch], dtype=torch.long),
            "has_relation": torch.tensor([x["has_relation"] for x in batch], dtype=torch.bool),
            "injected": torch.tensor([x["injected"] for x in batch], dtype=torch.bool),
        }
    return collate


def make_loader(
    cfg: Config,
    manifest: pd.DataFrame,
    feature_dir: Path,
    tokenizer: Any,
    split: str,
    training: bool,
    shuffle: bool,
    batch_size: int,
    seed: int | None = None,
) -> DataLoader:
    dataset = CachedDODDataset(cfg, manifest, feature_dir, split, training)
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=cfg.num_workers,
        collate_fn=make_collate(cfg, tokenizer),
        pin_memory=True,
        drop_last=training and len(dataset) >= batch_size,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def model_forward(model: DODVLM, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return model(
        batch["visual_features"], batch["boxes"], batch["detector_scores"],
        batch["candidate_mask"], batch["output_mask"], batch["image_sizes"],
        batch["query_ids"], batch["query_mask"], batch["target_ids"],
        batch["target_text_mask"], batch["anchor_ids"], batch["anchor_text_mask"],
        batch["relation_ids"], batch["has_relation"],
    )


def focal_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, gamma: float
) -> torch.Tensor:
    if not mask.any():
        return logits.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    pt = probability * targets + (1.0 - probability) * (1.0 - targets)
    loss = ((1.0 - pt).pow(gamma) * bce) * mask.float()
    return loss.sum() / mask.float().sum().clamp_min(1.0)


def compute_loss(
    cfg: Config, output: dict[str, torch.Tensor], batch: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, float]]:
    candidate = focal_bce_with_logits(
        output["candidate_logits"], batch["candidate_y"], batch["output_mask"], cfg.focal_gamma
    )
    null = F.binary_cross_entropy_with_logits(output["null_logit"], batch["null_y"])
    entity_mask = batch["output_mask"]
    if entity_mask.any():
        entity_targets = (batch["candidate_y"][entity_mask] > 0.5).float()
        target_aux = F.binary_cross_entropy_with_logits(
            output["target_match"][entity_mask], entity_targets
        )
    else:
        target_aux = output["target_match"].sum() * 0.0

    total = (
        cfg.candidate_loss_weight * candidate
        + cfg.null_loss_weight * null
        + cfg.target_entity_aux_weight * target_aux
    )
    return total, {
        "candidate": float(candidate.detach()),
        "null": float(null.detach()),
        "target_aux": float(target_aux.detach()),
    }
