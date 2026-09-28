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
            "loss_total": float(self.total.detach()),
            "loss_target": float(self.target.detach()),
            "loss_attribute": float(self.attribute.detach()),
            "loss_relation": float(self.relation.detach()),
            "loss_no_target": float(self.no_target.detach()),
            "loss_distillation": float(self.distillation.detach()),
        }


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def compute_loss(
    output: GroundingOutput,
    batch: dict[str, torch.Tensor],
    weights: LossConfig,
    kd_temperature: float = 2.0,
) -> LossBreakdown:
    # Main objective: choose one candidate or the explicit NO_TARGET class.
    target = F.cross_entropy(output.logits, batch["target_index"])

    attribute = _zero_like(output.logits)
    if batch["has_attribute"].any():
        attr_targets = torch.zeros_like(output.attribute_logits)
        rows = torch.arange(output.logits.shape[0], device=output.logits.device)
        valid = batch["has_attribute"] & (batch["target_index"] < output.candidate_logits.shape[1])
        attr_targets[rows[valid], batch["target_index"][valid]] = 1.0
        attr_mask = batch["candidate_mask"] & batch["target_mask"] & batch["has_attribute"][:, None]
        if attr_mask.any():
            attribute = F.binary_cross_entropy_with_logits(
                output.attribute_logits[attr_mask], attr_targets[attr_mask]
            )

    # Only apply direct relation BCE when the dataset/cache contains a VERIFIED
    # reference-object target. gRefCOCO expressions often provide the target box but
    # not a separate GT box for the referenced object; those rows still train relation
    # reasoning indirectly through L_target, but must not be mislabeled as all-negative pairs.
    relation = _zero_like(output.logits)
    relation_reference = batch.get("relation_reference_target_mask")
    if relation_reference is not None:
        verified_rows = (
            batch["has_relation"]
            & relation_reference.any(dim=1)
            & (batch["target_index"] < output.candidate_logits.shape[1])
        )
        if verified_rows.any():
            relation_targets = torch.zeros_like(output.relation_logits)
            rows = torch.where(verified_rows)[0]
            for row in rows.tolist():
                relation_targets[
                    row,
                    batch["target_index"][row],
                    relation_reference[row],
                ] = 1.0
            pair_mask = (
                batch["candidate_mask"][:, :, None]
                & batch["candidate_mask"][:, None, :]
                & batch["target_mask"][:, :, None]
                & batch["reference_mask"][:, None, :]
                & verified_rows[:, None, None]
            )
            if pair_mask.any():
                relation = F.binary_cross_entropy_with_logits(
                    output.relation_logits[pair_mask], relation_targets[pair_mask]
                )

    # Auxiliary calibration objective. NO_TARGET is already present in the main CE;
    # this smaller-weight BCE makes rejection easier to tune without dominating it.
    no_target_targets = (batch["target_index"] == output.candidate_logits.shape[1]).float()
    no_target = F.binary_cross_entropy_with_logits(output.no_target_logit, no_target_targets)

    distillation = _zero_like(output.logits)
    teacher_mask = batch.get("teacher_mask")
    if "teacher_distribution" in batch and teacher_mask is not None and teacher_mask.any():
        selected = teacher_mask.bool()
        teacher = batch["teacher_distribution"][selected].to(output.logits.device).clone()
        candidate_allowed = (batch["candidate_mask"] & batch["target_mask"])[selected]
        allowed = torch.cat(
            [
                candidate_allowed,
                torch.ones((teacher.shape[0], 1), dtype=torch.bool, device=teacher.device),
            ],
            dim=1,
        )
        teacher = teacher.masked_fill(~allowed, 0.0)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        temperature = float(kd_temperature)
        teacher_soft = F.softmax(
            torch.log(teacher.clamp_min(1e-8)) / temperature,
            dim=-1,
        )
        student_log = F.log_softmax(output.logits[selected] / temperature, dim=-1)
        distillation = (
            F.kl_div(student_log, teacher_soft, reduction="batchmean")
            * (temperature * temperature)
        )

    total = (
        weights.target * target
        + weights.attribute * attribute
        + weights.relation * relation
        + weights.no_target * no_target
        + weights.distillation * distillation
    )
    return LossBreakdown(total, target, attribute, relation, no_target, distillation)
