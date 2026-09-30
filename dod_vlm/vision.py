from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
from PIL import Image
import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import functional as TVF
from tqdm.auto import tqdm
from ultralytics import YOLOE

from .config import Config, Paths
from .utils import RunState, atomic_torch_save, gpu_report, stage_timer


class FrozenRegionEncoder:
    """Detector-independent object encoder with correct Faster-RCNN preprocessing."""

    def __init__(
        self,
        cfg: Config,
        device: torch.device,
        state_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device

        if state_path is None:
            base = fasterrcnn_mobilenet_v3_large_fpn(
                weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            )
        else:
            saved = torch.load(state_path, map_location="cpu", weights_only=False)
            base = fasterrcnn_mobilenet_v3_large_fpn(
                weights=None,
                weights_backbone=None,
                min_size=int(saved["transform"]["min_size"]),
                max_size=int(saved["transform"]["max_size"]),
                image_mean=list(saved["transform"]["image_mean"]),
                image_std=list(saved["transform"]["image_std"]),
            )
            base.backbone.load_state_dict(saved["backbone"])

        self.transform = base.transform
        self.backbone = base.backbone.to(device).eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        self.roi = MultiScaleRoIAlign(
            featmap_names=["0", "1", "pool"],
            output_size=cfg.roi_output_size,
            sampling_ratio=2,
        )
        del base

    @torch.inference_mode()
    def encode(self, image_pil: Image.Image, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        if boxes_xyxy.numel() == 0:
            return torch.zeros((0, self.cfg.visual_dim), dtype=torch.float32)

        image = TVF.pil_to_tensor(image_pil).float().div_(255.0).to(self.device)
        original_h, original_w = image.shape[-2:]
        image_list, _ = self.transform([image], None)
        transformed_h, transformed_w = image_list.image_sizes[0]

        boxes = boxes_xyxy.to(self.device).float().clone()
        scale = torch.tensor(
            [
                transformed_w / max(original_w, 1),
                transformed_h / max(original_h, 1),
                transformed_w / max(original_w, 1),
                transformed_h / max(original_h, 1),
            ],
            dtype=boxes.dtype,
            device=boxes.device,
        )
        boxes = boxes * scale
        boxes[:, 0::2].clamp_(0, transformed_w)
        boxes[:, 1::2].clamp_(0, transformed_h)

        features = self.backbone(image_list.tensors)
        pooled = self.roi(features, [boxes], image_list.image_sizes)
        pooled = pooled.mean(dim=(-1, -2))
        if pooled.shape[-1] != self.cfg.visual_dim:
            raise RuntimeError(
                f"ROI feature dim={pooled.shape[-1]} nhưng cfg.visual_dim={self.cfg.visual_dim}"
            )
        return pooled.float().cpu()

    def save_state(self, path: Path) -> None:
        min_size = self.transform.min_size
        if isinstance(min_size, (list, tuple)):
            min_size = min_size[-1]
        payload = {
            "backbone": {k: v.detach().cpu() for k, v in self.backbone.state_dict().items()},
            "transform": {
                "min_size": int(min_size),
                "max_size": int(self.transform.max_size),
                "image_mean": list(self.transform.image_mean),
                "image_std": list(self.transform.image_std),
            },
            "visual_dim": self.cfg.visual_dim,
            "roi_output_size": self.cfg.roi_output_size,
        }
        atomic_torch_save(payload, path)


def feature_cache_signature(cfg: Config) -> str:
    payload = {
        "proposal_model": cfg.proposal_model,
        "proposal_imgsz": cfg.proposal_imgsz,
        "proposal_conf": cfg.proposal_conf,
        "top_k": cfg.top_k,
        "roi_output_size": cfg.roi_output_size,
        "visual_dim": cfg.visual_dim,
        "region_encoder": "fasterrcnn_mobilenet_v3_large_fpn_correct_transform_v2",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def cache_path_for_image(feature_dir: Path, image_id: int) -> Path:
    return feature_dir / f"{int(image_id):012d}.pt"


def cache_is_valid(path: Path, signature: str) -> bool:
    if not path.exists():
        return False
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        return obj.get("cache_signature") == signature
    except Exception:
        return False


def collect_train_gt_by_image(manifest: pd.DataFrame) -> dict[int, torch.Tensor]:
    train = manifest[manifest["split"] == "train"]
    grouped: dict[int, list[list[float]]] = {}
    for row in train.itertuples():
        grouped.setdefault(int(row.image_id), []).extend(row.gt_boxes)

    result: dict[int, torch.Tensor] = {}
    for image_id, boxes in grouped.items():
        if not boxes:
            result[image_id] = torch.zeros((0, 4), dtype=torch.float32)
            continue
        array = torch.tensor(boxes, dtype=torch.float32)
        rounded = torch.round(array * 10) / 10
        seen: set[tuple[float, ...]] = set()
        keep: list[int] = []
        for index, box in enumerate(rounded.tolist()):
            key = tuple(box)
            if key not in seen:
                seen.add(key)
                keep.append(index)
        result[image_id] = array[keep]
    return result


def _save_proposal_model(model: YOLOE, destination: Path, source_name: str) -> None:
    if destination.exists():
        return

    candidates = [
        Path(source_name),
        Path.cwd() / source_name,
        Path(getattr(model, "ckpt_path", "")) if getattr(model, "ckpt_path", None) else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists() and candidate.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            return

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(destination))
    except Exception as exc:
        raise RuntimeError(
            "YOLOE đã load được nhưng không thể đóng gói checkpoint local. "
            "Final artifact sẽ không self-contained nếu bỏ qua bước này."
        ) from exc
    if not destination.exists():
        raise RuntimeError("Không tạo được local proposal checkpoint sau model.save().")


def cache_visual_features(
    cfg: Config,
    paths: Paths,
    state: RunState,
    manifest: pd.DataFrame,
    device: torch.device,
) -> tuple[Path, Path, Path, str]:
    signature = feature_cache_signature(cfg)
    feature_dir = paths.feature_cache / signature
    feature_dir.mkdir(parents=True, exist_ok=True)
    proposal_local = paths.models / "yoloe_prompt_free.pt"
    region_local = paths.models / "region_encoder.pt"

    with stage_timer("Session 4 — Cache YOLOE proposal + object token"):
        proposal = YOLOE(cfg.proposal_model)
        _save_proposal_model(proposal, proposal_local, cfg.proposal_model)

        if region_local.exists():
            region_encoder = FrozenRegionEncoder(cfg, device, state_path=region_local)
        else:
            region_encoder = FrozenRegionEncoder(cfg, device)
            region_encoder.save_state(region_local)

        train_gt = collect_train_gt_by_image(manifest)
        unique_images = (
            manifest[["image_id", "file_name", "width", "height"]]
            .drop_duplicates("image_id")
            .to_dict("records")
        )
        completed = sum(
            cache_is_valid(cache_path_for_image(feature_dir, item["image_id"]), signature)
            for item in unique_images
        )

        progress = tqdm(
            unique_images,
            desc=f"Feature cache ({completed}/{len(unique_images)} sẵn)",
            unit="img",
        )
        for item in progress:
            image_id = int(item["image_id"])
            output = cache_path_for_image(feature_dir, image_id)
            if cache_is_valid(output, signature):
                continue

            image_path = paths.images / item["file_name"]
            image_pil = Image.open(image_path).convert("RGB")
            result = proposal.predict(
                source=str(image_path),
                imgsz=cfg.proposal_imgsz,
                conf=cfg.proposal_conf,
                max_det=cfg.top_k,
                device=0,
                half=True,
                verbose=False,
            )[0]

            fallback_mask: torch.Tensor
            if result.boxes is None or len(result.boxes) == 0:
                boxes = torch.tensor(
                    [[0.0, 0.0, float(image_pil.width), float(image_pil.height)]],
                    dtype=torch.float32,
                )
                scores = torch.zeros((1,), dtype=torch.float32)
                labels = ["__fallback_context__"]
                fallback_mask = torch.ones((1,), dtype=torch.bool)
            else:
                boxes = result.boxes.xyxy.detach().cpu().float()[: cfg.top_k]
                scores = result.boxes.conf.detach().cpu().float()[: cfg.top_k]
                cls_ids = result.boxes.cls.detach().cpu().long().tolist()[: cfg.top_k]
                labels = [str(result.names[int(index)]) for index in cls_ids]
                fallback_mask = torch.zeros((len(boxes),), dtype=torch.bool)

            proposal_features = region_encoder.encode(image_pil, boxes)
            gt_boxes = train_gt.get(image_id, torch.zeros((0, 4), dtype=torch.float32))
            gt_features = region_encoder.encode(image_pil, gt_boxes)

            atomic_torch_save(
                {
                    "cache_signature": signature,
                    "image_id": image_id,
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                    "fallback_mask": fallback_mask,
                    "features": proposal_features,
                    "train_gt_boxes": gt_boxes,
                    "train_gt_features": gt_features,
                    "image_size": (int(item["height"]), int(item["width"])),
                },
                output,
            )

        del proposal, region_encoder
        gc.collect()
        torch.cuda.empty_cache()
        gpu_report("Sau cache:")
        state.mark(
            "feature_cache_complete",
            signature=signature,
            images=len(unique_images),
            proposal_model=str(proposal_local),
            region_encoder=str(region_local),
        )

    return feature_dir, proposal_local, region_local, signature
