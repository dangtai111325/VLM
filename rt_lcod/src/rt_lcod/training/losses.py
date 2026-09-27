from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rt_lcod.config import LossConfig
from rt_lcod.models.grounding_head import GroundingOutput


@dataclass
class LossBreakdown:
    total: torch.Tensor
    target: torch.Tensor
    attribute: torch.Tensor
    relation: torch.Tensor
    no_target: torch.Tensor
    distillation: torch.Tensor

    def scalar_dict(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach()), "loss_target": float(self.target.detach()),
            "loss_attribute": float(self.attribute.detach()), "loss_relation": float(self.relation.detach()),
            "loss_no_target": float(self.no_target.detach()), "loss_distillation": float(self.distillation.detach()),
        }


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def compute_loss(output: GroundingOutput, batch: dict[str, torch.Tensor], weights: LossConfig, kd_temperature: float = 2.0) -> LossBreakdown:
    target = F.cross_entropy(output.logits, batch["target_index"])

    attribute = _zero_like(output.logits)
    if batch["has_attribute"].any():
        attr_targets = torch.zeros_like(output.attribute_logits)
        rows = torch.arange(output.logits.shape[0], device=output.logits.device)
        valid = batch["has_attribute"] & (batch["target_index"] < output.candidate_logits.shape[1])
        attr_targets[rows[valid], batch["target_index"][valid]] = 1.0
        attr_mask = batch["candidate_mask"] & batch["target_mask"] & batch["has_attribute"][:, None]
        if attr_mask.any():
            attribute = F.binary_cross_entropy_with_logits(output.attribute_logits[attr_mask], attr_targets[attr_mask])

    relation = _zero_like(output.logits)
    if batch["has_relation"].any() and "relation_reference_target_mask" in batch:
        relation_targets = torch.zeros_like(output.relation_logits)
        rows = torch.arange(output.logits.shape[0], device=output.logits.device)
        valid = batch["has_relation"] & (batch["target_index"] < output.candidate_logits.shape[1])
        for row in rows[valid].tolist():
            relation_targets[row, batch["target_index"][row], batch["relation_reference_target_mask"][row]] = 1.0
        pair_mask = (
            batch["candidate_mask"][:, :, None] & batch["candidate_mask"][:, None, :]
            & batch["target_mask"][:, :, None] & batch["reference_mask"][:, None, :]
            & batch["has_relation"][:, None, None]
        )
        if pair_mask.any():
            relation = F.binary_cross_entropy_with_logits(output.relation_logits[pair_mask], relation_targets[pair_mask])

    no_target_targets = (batch["target_index"] == output.candidate_logits.shape[1]).float()
    no_target = F.binary_cross_entropy_with_logits(output.no_target_logit, no_target_targets)

    distillation = _zero_like(output.logits)
    if "teacher_distribution" in batch:
        teacher = batch["teacher_distribution"].to(output.logits.device).clone()
        allowed = torch.cat([
            batch["candidate_mask"] & batch["target_mask"],
            torch.ones((teacher.shape[0], 1), dtype=torch.bool, device=teacher.device),
        ], dim=1)
        teacher = teacher.masked_fill(~allowed, 0.0)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        t = float(kd_temperature)
        teacher_soft = F.softmax(torch.log(teacher.clamp_min(1e-8)) / t, dim=-1)
        student_log = F.log_softmax(output.logits / t, dim=-1)
        distillation = F.kl_div(student_log, teacher_soft, reduction="batchmean") * (t * t)

    total = (
        weights.target * target + weights.attribute * attribute + weights.relation * relation
        + weights.no_target * no_target + weights.distillation * distillation
    )
    return LossBreakdown(total, target, attribute, relation, no_target, distillation)
