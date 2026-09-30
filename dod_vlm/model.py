from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertConfig, DistilBertModel

from .config import Config
from .query import RELATION_NAMES


def normalize_boxes(boxes: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
    h = image_sizes[:, 0]
    w = image_sizes[:, 1]
    scale = torch.stack([w, h, w, h], dim=-1)
    return boxes / scale[:, None, :].clamp_min(1.0)


def box_geometry(boxes_norm: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_norm.unbind(-1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw = (x2 - x1).clamp_min(1e-6)
    bh = (y2 - y1).clamp_min(1e-6)
    area = bw * bh
    return torch.stack([cx, cy, bw, bh, area, x1, y1, x2, y2], dim=-1)


def pair_geometry(boxes_norm: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_norm.unbind(-1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = (x2 - x1).clamp_min(1e-6)
    h = (y2 - y1).clamp_min(1e-6)

    dxc = cx[:, :, None] - cx[:, None, :]
    dyc = cy[:, :, None] - cy[:, None, :]
    log_wr = torch.log(w[:, :, None] / w[:, None, :])
    log_hr = torch.log(h[:, :, None] / h[:, None, :])

    left_margin = x1[:, None, :] - x2[:, :, None]
    right_margin = x1[:, :, None] - x2[:, None, :]
    above_margin = y1[:, None, :] - y2[:, :, None]
    below_margin = y1[:, :, None] - y2[:, None, :]

    inter_x1 = torch.maximum(x1[:, :, None], x1[:, None, :])
    inter_y1 = torch.maximum(y1[:, :, None], y1[:, None, :])
    inter_x2 = torch.minimum(x2[:, :, None], x2[:, None, :])
    inter_y2 = torch.minimum(y2[:, :, None], y2[:, None, :])
    overlap_x = (inter_x2 - inter_x1).clamp_min(0)
    overlap_y = (inter_y2 - inter_y1).clamp_min(0)
    inter = overlap_x * overlap_y
    area_i = w[:, :, None] * h[:, :, None]
    area_j = w[:, None, :] * h[:, None, :]
    iou = inter / (area_i + area_j - inter).clamp_min(1e-6)
    distance = torch.sqrt(dxc.square() + dyc.square() + 1e-8)

    return torch.stack(
        [
            dxc,
            dyc,
            log_wr,
            log_hr,
            left_margin,
            right_margin,
            above_margin,
            below_margin,
            iou,
            distance,
            overlap_x,
            overlap_y,
        ],
        dim=-1,
    )


class DODVLM(nn.Module):
    def __init__(
        self,
        cfg: Config,
        *,
        pretrained_text: bool = True,
        text_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        if pretrained_text:
            self.text_encoder = DistilBertModel.from_pretrained(cfg.text_model)
        else:
            if text_config is None:
                raise ValueError("text_config là bắt buộc khi pretrained_text=False")
            self.text_encoder = DistilBertModel(DistilBertConfig(**text_config))

        text_dim = int(self.text_encoder.config.dim)
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(False)
        if cfg.unfreeze_last_text_layers > 0:
            for layer in self.text_encoder.transformer.layer[-cfg.unfreeze_last_text_layers :]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)

        self.visual_proj = nn.Sequential(
            nn.Linear(cfg.visual_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )
        self.geom_proj = nn.Sequential(
            nn.Linear(9, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.text_proj = nn.Linear(text_dim, cfg.d_model)

        object_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.object_encoder = nn.TransformerEncoder(object_layer, num_layers=cfg.object_layers)

        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    cfg.d_model,
                    cfg.num_heads,
                    dropout=cfg.dropout,
                    batch_first=True,
                )
                for _ in range(cfg.cross_layers)
            ]
        )
        self.cross_norm = nn.ModuleList(
            [nn.LayerNorm(cfg.d_model) for _ in range(cfg.cross_layers)]
        )

        self.generic_head = nn.Sequential(
            nn.Linear(cfg.d_model * 2 + 1, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

        self.entity_visual = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.entity_text = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.entity_logit_scale = nn.Parameter(torch.tensor(2.3025851))
        self.relation_embed = nn.Embedding(len(RELATION_NAMES), cfg.d_model)
        self.pair_geom_mlp = nn.Sequential(
            nn.Linear(cfg.relation_geom_dim, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.d_model // 2),
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(cfg.d_model * 3 + cfg.d_model // 2, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )
        self.structured_head = nn.Sequential(
            nn.Linear(cfg.d_model + 3, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )
        self.null_head = nn.Sequential(
            nn.Linear(cfg.d_model * 2 + 3, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        tokens = self.text_proj(output.last_hidden_state)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return tokens, pooled

    def forward(
        self,
        visual_features: torch.Tensor,
        boxes: torch.Tensor,
        detector_scores: torch.Tensor,
        candidate_mask: torch.Tensor,
        output_mask: torch.Tensor,
        image_sizes: torch.Tensor,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_text_mask: torch.Tensor,
        anchor_ids: torch.Tensor,
        anchor_text_mask: torch.Tensor,
        relation_ids: torch.Tensor,
        has_relation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _, candidate_count, _ = visual_features.shape

        query_tokens, query_pooled = self.encode_text(query_ids, query_mask)
        _, target_pooled = self.encode_text(target_ids, target_text_mask)
        _, anchor_pooled = self.encode_text(anchor_ids, anchor_text_mask)

        boxes_norm = normalize_boxes(boxes, image_sizes)
        objects = self.visual_proj(visual_features) + self.geom_proj(box_geometry(boxes_norm))
        objects = self.object_encoder(objects, src_key_padding_mask=~candidate_mask)

        generic_objects = objects
        text_padding_mask = ~query_mask.bool()
        for attention, norm in zip(self.cross_attn, self.cross_norm):
            delta, _ = attention(
                generic_objects,
                query_tokens,
                query_tokens,
                key_padding_mask=text_padding_mask,
                need_weights=False,
            )
            generic_objects = norm(generic_objects + delta)
        query_expand = query_pooled[:, None, :].expand(-1, candidate_count, -1)
        generic_logits = self.generic_head(
            torch.cat(
                [generic_objects, query_expand, detector_scores.unsqueeze(-1)],
                dim=-1,
            )
        ).squeeze(-1)

        visual_entity = F.normalize(self.entity_visual(objects), dim=-1)
        target_entity = F.normalize(self.entity_text(target_pooled), dim=-1)
        anchor_entity = F.normalize(self.entity_text(anchor_pooled), dim=-1)
        entity_scale = self.entity_logit_scale.exp().clamp(max=100.0)
        target_match = entity_scale * (visual_entity * target_entity[:, None, :]).sum(-1)
        anchor_match = entity_scale * (visual_entity * anchor_entity[:, None, :]).sum(-1)

        pair_features = self.pair_geom_mlp(pair_geometry(boxes_norm))
        target_object = objects[:, :, None, :].expand(-1, candidate_count, candidate_count, -1)
        anchor_object = objects[:, None, :, :].expand(-1, candidate_count, candidate_count, -1)
        relation = self.relation_embed(relation_ids)[:, None, None, :].expand(
            -1, candidate_count, candidate_count, -1
        )
        relation_logits = self.relation_mlp(
            torch.cat([target_object, anchor_object, relation, pair_features], dim=-1)
        ).squeeze(-1)

        pair_valid = output_mask[:, :, None] & output_mask[:, None, :]
        diagonal = torch.eye(candidate_count, dtype=torch.bool, device=objects.device)[None]
        pair_valid &= ~diagonal
        masked_relation = relation_logits.masked_fill(~pair_valid, -1e4)

        anchor_scores = anchor_match.masked_fill(~output_mask, -1e4)
        anchor_log_weights = torch.log_softmax(anchor_scores, dim=-1)
        relation_support = torch.logsumexp(
            masked_relation + anchor_log_weights[:, None, :],
            dim=-1,
        )
        has_pair = pair_valid.any(dim=-1)
        relation_support = torch.where(
            has_relation[:, None] & has_pair,
            relation_support,
            torch.zeros_like(relation_support),
        )

        structured_logits = self.structured_head(
            torch.cat(
                [
                    objects,
                    detector_scores.unsqueeze(-1),
                    target_match.unsqueeze(-1),
                    relation_support.unsqueeze(-1),
                ],
                dim=-1,
            )
        ).squeeze(-1)

        gate = torch.sigmoid(self.fusion_gate(query_pooled))
        candidate_logits = (1.0 - gate) * generic_logits + gate * structured_logits
        candidate_logits = candidate_logits.masked_fill(~output_mask, -1e4)

        candidate_float = candidate_mask.float().unsqueeze(-1)
        pooled_visual = (objects * candidate_float).sum(1) / candidate_float.sum(1).clamp_min(1.0)
        has_output = output_mask.any(dim=1)
        raw_best = candidate_logits.max(dim=1).values
        best_logit = torch.where(has_output, raw_best, torch.zeros_like(raw_best)).unsqueeze(-1)
        mean_detector_score = (
            detector_scores.masked_fill(~output_mask, 0.0).sum(1)
            / output_mask.sum(1).clamp_min(1)
        ).unsqueeze(-1)
        has_output_float = has_output.float().unsqueeze(-1)
        null_logit = self.null_head(
            torch.cat(
                [
                    pooled_visual,
                    query_pooled,
                    best_logit,
                    mean_detector_score,
                    has_output_float,
                ],
                dim=-1,
            )
        ).squeeze(-1)

        return {
            "candidate_logits": candidate_logits,
            "null_logit": null_logit,
            "generic_logits": generic_logits,
            "structured_logits": structured_logits,
            "target_match": target_match,
            "anchor_match": anchor_match,
            "relation_logits": relation_logits,
            "fusion_gate": gate.squeeze(-1),
        }
