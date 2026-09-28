from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm

from rt_lcod.config import load_config
from rt_lcod.data.schema import load_manifest
from rt_lcod.geometry import box_iou_xyxy
from rt_lcod.models.region_encoder import build_region_encoder
from rt_lcod.models.text_encoder import HFTextEncoder
from rt_lcod.models.yoloe_adapter import YOLOECandidateGenerator


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def best_match(candidate_boxes: torch.Tensor, gt_boxes: list[list[float]], min_iou: float):
    if candidate_boxes.numel() == 0 or not gt_boxes:
        return -1, 0.0
    gt = torch.tensor(gt_boxes, dtype=torch.float32, device=candidate_boxes.device)
    c = candidate_boxes[:, None, :].expand(-1, gt.shape[0], -1)
    g = gt[None, :, :].expand(candidate_boxes.shape[0], -1, -1)
    ious = box_iou_xyxy(c, g)
    value, flat = torch.max(ious.reshape(-1), dim=0)
    candidate_index = int(flat // gt.shape[0])
    return (candidate_index if float(value) >= min_iou else -1), float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--min-match-iou", type=float, default=0.50)
    parser.add_argument("--max-candidates", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)
    device_name = config.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    samples = load_manifest(args.manifest)
    detector = YOLOECandidateGenerator(
        args.model or config.model.detector_name,
        device=device_name,
        imgsz=int(config.runtime.get("input_size", 640)),
    )
    region_encoder = build_region_encoder(
        config.model.region_encoder,
        visual_dim=config.model.visual_dim).to(device).eval()
    text_encoder = HFTextEncoder(config.model.text_model_name, device=device_name)

    skipped = 0
    cached_count = 0
    progress = tqdm(samples, desc="cache candidates", unit="sample", dynamic_ncols=True)
    for index, sample in enumerate(progress, 1):
        image_path = (Path(args.manifest).parent / sample.image).resolve()
        if not image_path.exists():
            image_path = Path(sample.image).resolve()
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image)
        bgr = np.ascontiguousarray(array[..., ::-1])
        classes = [sample.target_class] + \
            ([sample.reference_class] if sample.reference_class else [])
        detections = detector.predict(bgr, classes, conf=args.confidence)
        detections = sorted(detections, key=lambda d: d.score, reverse=True)[: args.max_candidates]
        if not detections:
            skipped += 1
            progress.set_postfix(cached=cached_count, skipped=skipped)
            continue

        boxes_px = torch.tensor([d.box for d in detections], dtype=torch.float32, device=device)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(array)).to(device).float().permute(
            2, 0, 1)[None] / 255.0
        with torch.inference_mode():
            features = region_encoder(image_tensor, [boxes_px])[0].cpu()
        norm = boxes_px.detach().cpu().clone()
        norm[:, [0, 2]] /= sample.width
        norm[:, [1, 3]] /= sample.height
        labels = [d.label.lower().strip() for d in detections]
        target_mask = torch.tensor([x == sample.target_class.lower()
                                   for x in labels], dtype=torch.bool)
        reference_mask = torch.tensor([bool(sample.reference_class) and x ==
                                      sample.reference_class.lower() for x in labels], dtype=torch.bool)

        target_index = -1
        best_iou = 0.0
        if not sample.no_target:
            candidate_indices = torch.where(target_mask)[0]
            if candidate_indices.numel():
                matched_local, best_iou = best_match(
                    boxes_px[candidate_indices], sample.target_boxes, args.min_match_iou)
                if matched_local >= 0:
                    target_index = int(candidate_indices[matched_local])
            if target_index < 0:
                skipped += 1
                progress.set_postfix(cached=cached_count, skipped=skipped)
                continue

        relation_reference_target_mask = torch.zeros(len(detections), dtype=torch.bool)
        if sample.relation and sample.reference_boxes and reference_mask.any():
            ref_indices = torch.where(reference_mask)[0]
            matched_local, _ = best_match(
                boxes_px[ref_indices], sample.reference_boxes, args.min_match_iou)
            if matched_local >= 0:
                relation_reference_target_mask[int(ref_indices[matched_local])] = True

        attr = text_encoder.encode([sample.attribute or ""], device="cpu")[0]
        rel = text_encoder.encode([sample.relation or ""], device="cpu")[0]
        record = {
            "sample_id": sample.sample_id,
            "features": features.float(),
            "boxes": norm.float(),
            "detector_scores": torch.tensor([d.score for d in detections], dtype=torch.float32),
            "candidate_labels": labels,
            "candidate_mask": torch.ones(len(detections), dtype=torch.bool),
            "target_mask": target_mask,
            "reference_mask": reference_mask,
            "relation_reference_target_mask": relation_reference_target_mask,
            "target_index": target_index,
            "attribute_embedding": attr.float(),
            "relation_embedding": rel.float(),
            "has_attribute": sample.attribute is not None,
            "has_relation": sample.relation is not None,
            "proposal_match_iou": best_iou,
            "prompt": sample.prompt,
        }
        torch.save(record, output / f"{safe_name(sample.sample_id)}.pt")
        cached_count += 1
        progress.set_postfix(cached=cached_count, skipped=skipped)
    print(f"done: total={len(samples)} cached={len(list(output.glob('*.pt')))} skipped={skipped}")


if __name__ == "__main__":
    main()
