from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import roi_align


class TinyRegionEncoder(nn.Module):
    """Dependency-light shared encoder used by smoke tests."""

    output_dim = 32

    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.output_dim = int(output_dim)
        self.stride = 16
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, self.output_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.output_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.output_dim, self.output_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.output_dim),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        images: torch.Tensor,
        boxes_per_image: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return _roi_features(self.backbone(images), boxes_per_image, self.stride)


class MobileNetV3RegionEncoder(nn.Module):
    """Frozen ImageNet MobileNetV3-Small features + ROIAlign for real experiments."""

    output_dim = 576

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        self.backbone = model.features
        self.stride = 32
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.backbone.eval()
        return self

    @torch.inference_mode()
    def forward(
        self,
        images: torch.Tensor,
        boxes_per_image: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        # Callers provide RGB float tensors in [0, 1]. Match torchvision ImageNet weights.
        normalized = (images - self.image_mean) / self.image_std
        return _roi_features(self.backbone(normalized), boxes_per_image, self.stride)


def _roi_features(
    feature_map: torch.Tensor,
    boxes_per_image: list[torch.Tensor],
    stride: int,
) -> list[torch.Tensor]:
    rois: list[torch.Tensor] = []
    counts: list[int] = []
    for batch_index, boxes in enumerate(boxes_per_image):
        counts.append(int(boxes.shape[0]))
        if boxes.numel() == 0:
            continue
        batch_col = torch.full(
            (boxes.shape[0], 1),
            float(batch_index),
            dtype=boxes.dtype,
            device=boxes.device,
        )
        rois.append(torch.cat([batch_col, boxes], dim=1))

    if not rois:
        return [feature_map.new_zeros((0, feature_map.shape[1])) for _ in boxes_per_image]

    pooled = roi_align(
        feature_map,
        torch.cat(rois, dim=0),
        output_size=(1, 1),
        spatial_scale=1.0 / stride,
        aligned=True,
    ).flatten(1)

    outputs: list[torch.Tensor] = []
    cursor = 0
    for count in counts:
        outputs.append(pooled[cursor : cursor + count])
        cursor += count
    return outputs


def build_region_encoder(name: str, visual_dim: int | None = None) -> nn.Module:
    if name == "tiny":
        return TinyRegionEncoder(output_dim=int(visual_dim or 32))
    if name == "mobilenet_v3_small":
        encoder = MobileNetV3RegionEncoder(pretrained=True)
        if visual_dim is not None and visual_dim != encoder.output_dim:
            raise ValueError(
                f"mobilenet_v3_small produces {encoder.output_dim} dims, "
                f"config requested {visual_dim}"
            )
        return encoder
    raise ValueError(f"unknown region encoder: {name}")
