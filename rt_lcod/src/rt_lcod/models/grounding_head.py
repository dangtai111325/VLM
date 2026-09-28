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
    def __init__(
            self,
            visual_dim: int,
            text_dim: int,
            hidden_dim: int = 192,
            relation_feature_dim: int = 11,
            dropout: float = 0.1):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(
                visual_dim,
                hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout))
        self.attribute_visual = nn.Linear(hidden_dim, hidden_dim)
        self.attribute_text = nn.Linear(text_dim, hidden_dim)
        self.relation_text = nn.Linear(text_dim, hidden_dim)
        self.geometry_proj = nn.Sequential(
            nn.Linear(
                relation_feature_dim,
                hidden_dim // 2),
            nn.SiLU())
        relation_input = hidden_dim * 3 + hidden_dim // 2
        self.relation_mlp = nn.Sequential(
            nn.Linear(
                relation_input,
                hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                1))
        self.candidate_fusion = nn.Sequential(
            nn.Linear(
                hidden_dim + 4,
                hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                1))
        self.no_target_head = nn.Sequential(
            nn.Linear(
                hidden_dim * 3 + 2,
                hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                1))

    def forward(
            self,
            features,
            boxes,
            detector_scores,
            candidate_mask,
            target_mask,
            reference_mask,
            attribute_embedding,
            relation_embedding,
            has_attribute,
            has_relation) -> GroundingOutput:
        bsz, n, _ = features.shape
        visual = self.visual_proj(features)
        attr_v = F.normalize(self.attribute_visual(visual), dim=-1)
        attr_t = F.normalize(self.attribute_text(attribute_embedding), dim=-1)
        attribute_logits = (attr_v * attr_t[:, None, :]).sum(-1)
        attribute_logits = torch.where(
            has_attribute[:, None], attribute_logits, torch.zeros_like(attribute_logits))

        target_visual = visual[:, :, None, :].expand(bsz, n, n, -1)
        reference_visual = visual[:, None, :, :].expand(bsz, n, n, -1)
        target_boxes = boxes[:, :, None, :].expand(bsz, n, n, 4)
        reference_boxes = boxes[:, None, :, :].expand(bsz, n, n, 4)
        geometry_h = self.geometry_proj(relation_geometry(target_boxes, reference_boxes))
        relation_h = self.relation_text(relation_embedding)[:, None, None, :].expand(bsz, n, n, -1)
        relation_logits = self.relation_mlp(torch.cat(
            [target_visual, reference_visual, relation_h, geometry_h], dim=-1)).squeeze(-1)

        pair_mask = candidate_mask[:, :, None] & candidate_mask[:, None,
                                                                :] & target_mask[:, :, None] & reference_mask[:, None, :]
        pair_mask &= ~torch.eye(n, dtype=torch.bool, device=features.device)[None, :, :]
        masked_relation = relation_logits.masked_fill(~pair_mask, -1e4)
        relation_best = masked_relation.max(dim=-1).values
        relation_best = torch.where(has_relation[:, None],
                                    relation_best, torch.zeros_like(relation_best))

        fusion_input = torch.cat([
            visual,
            detector_scores.unsqueeze(-1),
            target_mask.float().unsqueeze(-1),
            attribute_logits.unsqueeze(-1),
            relation_best.unsqueeze(-1),
        ], dim=-1)
        candidate_logits = self.candidate_fusion(fusion_input).squeeze(-1)
        valid_target = candidate_mask & target_mask
        candidate_logits = candidate_logits.masked_fill(~valid_target, -1e4)

        mask_f = candidate_mask.float().unsqueeze(-1)
        pooled = (visual * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)
        attr_context = self.attribute_text(attribute_embedding)
        rel_context = self.relation_text(relation_embedding)
        best_candidate = candidate_logits.max(dim=1).values.unsqueeze(-1)
        reference_exists = reference_mask.any(dim=1).float().unsqueeze(-1)
        no_target_logit = self.no_target_head(torch.cat(
            [pooled, attr_context, rel_context, best_candidate, reference_exists], dim=-1)).squeeze(-1)
        logits = torch.cat([candidate_logits, no_target_logit[:, None]], dim=1)
        return GroundingOutput(
            logits,
            candidate_logits,
            attribute_logits,
            relation_logits,
            relation_best,
            no_target_logit)
