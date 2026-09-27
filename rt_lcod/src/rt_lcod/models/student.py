from __future__ import annotations

import torch
import torch.nn as nn

from rt_lcod.config import ModelConfig
from rt_lcod.models.grounding_head import AttributeRelationGroundingHead, GroundingOutput


class RTLCODStudent(nn.Module):
    """Trainable grounding student operating on frozen-detector candidate features."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.grounding = AttributeRelationGroundingHead(
            visual_dim=config.visual_dim,
            text_dim=config.text_dim,
            hidden_dim=config.hidden_dim,
            relation_feature_dim=config.relation_feature_dim,
            dropout=config.dropout,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> GroundingOutput:
        return self.grounding(
            features=batch["features"],
            boxes=batch["boxes"],
            detector_scores=batch["detector_scores"],
            candidate_mask=batch["candidate_mask"],
            target_mask=batch["target_mask"],
            reference_mask=batch["reference_mask"],
            attribute_embedding=batch["attribute_embedding"],
            relation_embedding=batch["relation_embedding"],
            has_attribute=batch["has_attribute"],
            has_relation=batch["has_relation"],
        )
