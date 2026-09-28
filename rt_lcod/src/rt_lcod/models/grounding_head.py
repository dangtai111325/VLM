from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rt_lcod.geometry import relation_geometry


@dataclass
class GroundingOutput:
    logits: torch.Tensor
    candidate_logits: torch.Tensor
    attribute_logits: torch.Tensor
    relation_logits: torch.Tensor
    relation_best: torch.Tensor
    no_target_logit: torch.Tensor


class AttributeRelationGroundingHead(nn.Module):
    """Lightweight compositional target selector over detector candidates."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int = 192,
        relation_feature_dim: int = 11,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.attribute_visual = nn.Linear(hidden_dim, hidden_dim)
        self.attribute_text = nn.Linear(text_dim, hidden_dim)
        self.relation_text = nn.Linear(text_dim, hidden_dim)
        self.geometry_proj = nn.Sequential(
            nn.Linear(relation_feature_dim, hidden_dim // 2),
            nn.SiLU(),
        )

        relation_input = hidden_dim * 3 + hidden_dim // 2
        self.relation_mlp = nn.Sequential(
            nn.Linear(relation_input, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.candidate_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # pooled visual + attribute context + relation context + three scalar signals:
        # best candidate score, target-class proposal exists, reference proposal exists.
        self.no_target_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        detector_scores: torch.Tensor,
        candidate_mask: torch.Tensor,
        target_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        attribute_embedding: torch.Tensor,
        relation_embedding: torch.Tensor,
        has_attribute: torch.Tensor,
        has_relation: torch.Tensor,
    ) -> GroundingOutput:
        batch_size, candidate_count, _ = features.shape
        visual = self.visual_proj(features)

        attr_visual = F.normalize(self.attribute_visual(visual), dim=-1)
        attr_text = F.normalize(self.attribute_text(attribute_embedding), dim=-1)
        attribute_logits = (attr_visual * attr_text[:, None, :]).sum(-1)
        attribute_logits = torch.where(
            has_attribute[:, None],
            attribute_logits,
            torch.zeros_like(attribute_logits),
        )

        target_visual = visual[:, :, None, :].expand(
            batch_size, candidate_count, candidate_count, -1
        )
        reference_visual = visual[:, None, :, :].expand(
            batch_size, candidate_count, candidate_count, -1
        )
        target_boxes = boxes[:, :, None, :].expand(
            batch_size, candidate_count, candidate_count, 4
        )
        reference_boxes = boxes[:, None, :, :].expand(
            batch_size, candidate_count, candidate_count, 4
        )
        geometry_hidden = self.geometry_proj(
            relation_geometry(target_boxes, reference_boxes)
        )
        relation_hidden = self.relation_text(relation_embedding)[:, None, None, :].expand(
            batch_size, candidate_count, candidate_count, -1
        )
        relation_logits = self.relation_mlp(
            torch.cat(
                [target_visual, reference_visual, relation_hidden, geometry_hidden],
                dim=-1,
            )
        ).squeeze(-1)

        pair_mask = (
            candidate_mask[:, :, None]
            & candidate_mask[:, None, :]
            & target_mask[:, :, None]
            & reference_mask[:, None, :]
        )
        pair_mask &= ~torch.eye(
            candidate_count,
            dtype=torch.bool,
            device=features.device,
        )[None, :, :]
        masked_relation = relation_logits.masked_fill(~pair_mask, -1e4)
        raw_relation_best = masked_relation.max(dim=-1).values
        valid_relation_for_target = pair_mask.any(dim=-1) & has_relation[:, None]
        relation_best = torch.where(
            valid_relation_for_target,
            raw_relation_best,
            torch.zeros_like(raw_relation_best),
        )

        fusion_input = torch.cat(
            [
                visual,
                detector_scores.unsqueeze(-1),
                target_mask.float().unsqueeze(-1),
                attribute_logits.unsqueeze(-1),
                relation_best.unsqueeze(-1),
            ],
            dim=-1,
        )
        candidate_logits = self.candidate_fusion(fusion_input).squeeze(-1)
        valid_target = candidate_mask & target_mask
        candidate_logits = candidate_logits.masked_fill(~valid_target, -1e4)

        candidate_float = candidate_mask.float().unsqueeze(-1)
        pooled_visual = (visual * candidate_float).sum(1) / candidate_float.sum(1).clamp_min(1.0)

        attribute_context = self.attribute_text(attribute_embedding)
        attribute_context = torch.where(
            has_attribute[:, None],
            attribute_context,
            torch.zeros_like(attribute_context),
        )
        relation_context = self.relation_text(relation_embedding)
        relation_context = torch.where(
            has_relation[:, None],
            relation_context,
            torch.zeros_like(relation_context),
        )

        target_exists_bool = valid_target.any(dim=1)
        target_exists = target_exists_bool.float().unsqueeze(-1)
        reference_exists = reference_mask.any(dim=1).float().unsqueeze(-1)
        raw_best_candidate = candidate_logits.max(dim=1).values.unsqueeze(-1)
        best_candidate = torch.where(
            target_exists_bool[:, None],
            raw_best_candidate,
            torch.zeros_like(raw_best_candidate),
        )

        no_target_input = torch.cat(
            [
                pooled_visual,
                attribute_context,
                relation_context,
                best_candidate,
                target_exists,
                reference_exists,
            ],
            dim=-1,
        )
        no_target_logit = self.no_target_head(no_target_input).squeeze(-1)
        logits = torch.cat([candidate_logits, no_target_logit[:, None]], dim=1)

        return GroundingOutput(
            logits=logits,
            candidate_logits=candidate_logits,
            attribute_logits=attribute_logits,
            relation_logits=relation_logits,
            relation_best=relation_best,
            no_target_logit=no_target_logit,
        )
