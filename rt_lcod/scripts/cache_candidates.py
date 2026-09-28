from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

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


def resolve_image_path(manifest: str | Path, image: str) -> Path:
    manifest = Path(manifest).resolve()
    raw = Path(image)
    candidates = [raw, manifest.parent / raw, manifest.parent.parent / raw]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot resolve image={image!r}; tried={[str(x) for x in candidates]}")


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
    parser.add_argument("--config", default="configs/t4_runall.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--min-match-iou", type=float, default=0.50)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true", help="recompute existing cache files")
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_config(args.config)
    confidence = float(args.confidence if args.confidence is not None else config.runtime.get("confidence", 0.15))
    max_candidates = int(args.max_candidates or config.runtime.get("top_k", config.model.top_k))
    device_name = config.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    samples = load_manifest(args.manifest)

    print("[CACHE] ============================================================")
    print(f"[CACHE] manifest={Path(args.manifest).resolve()}")
    print(f"[CACHE] samples={len(samples):,} output={output.resolve()}")
    print(f"[CACHE] device={device_name} model={args.model or config.model.detector_name}")
    print(f"[CACHE] conf={confidence} top_k={max_candidates} min_match_iou={args.min_match_iou}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"[CACHE] gpu={props.name} total_vram_gib={props.total_memory / 2**30:.2f}")

    detector = YOLOECandidateGenerator(
        args.model or config.model.detector_name,
        device=device_name,
        imgsz=int(config.runtime.get("input_size", 512)),
    )
    region_encoder = build_region_encoder(config.model.region_encoder, visual_dim=config.model.visual_dim).to(device).eval()
    text_encoder = HFTextEncoder(config.model.text_model_name, device=device_name)

    stats = {
        "total": len(samples), "cached_new": 0, "cached_existing": 0,
        "skipped_no_detections": 0, "skipped_proposal_miss": 0,
        "positive_total": 0, "positive_matched": 0, "negative_total": 0,
        "candidate_sum": 0, "relation_reference_present": 0,
    }
    proposal_ious: list[float] = []
    progress = tqdm(samples, desc="cache candidates", unit="sample", dynamic_ncols=True)
    for sample in progress:
        cache_path = output / f"{safe_name(sample.sample_id)}.pt"
        if cache_path.exists() and not args.no_resume:
            stats["cached_existing"] += 1
            continue
        if sample.no_target:
            stats["negative_total"] += 1
        else:
            stats["positive_total"] += 1

        image_path = resolve_image_path(args.manifest, sample.image)
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image)
        bgr = np.ascontiguousarray(array[..., ::-1])
        classes = [sample.target_class] + ([sample.reference_class] if sample.reference_class else [])
        detections = detector.predict(bgr, classes, conf=confidence)
        detections = sorted(detections, key=lambda d: d.score, reverse=True)[:max_candidates]
        if not detections:
            stats["skipped_no_detections"] += 1
            progress.set_postfix(new=stats["cached_new"], miss=stats["skipped_proposal_miss"])
            continue

        boxes_px = torch.tensor([d.box for d in detections], dtype=torch.float32, device=device)
        image_tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device).float().permute(2, 0, 1)[None] / 255.0
        with torch.inference_mode():
            features = region_encoder(image_tensor, [boxes_px])[0].cpu()
        norm = boxes_px.detach().cpu().clone()
        norm[:, [0, 2]] /= sample.width
        norm[:, [1, 3]] /= sample.height
        labels = [d.label.lower().strip() for d in detections]
        target_mask = torch.tensor([x == sample.target_class.lower() for x in labels], dtype=torch.bool)
        reference_mask = torch.tensor(
            [bool(sample.reference_class) and x == sample.reference_class.lower() for x in labels], dtype=torch.bool
        )

        target_index = -1
        best_iou = 0.0
        if not sample.no_target:
            candidate_indices = torch.where(target_mask)[0]
            if candidate_indices.numel():
                matched_local, best_iou = best_match(boxes_px[candidate_indices], sample.target_boxes, args.min_match_iou)
                if matched_local >= 0:
                    target_index = int(candidate_indices[matched_local])
            proposal_ious.append(best_iou)
            if target_index < 0:
                stats["skipped_proposal_miss"] += 1
                progress.set_postfix(new=stats["cached_new"], miss=stats["skipped_proposal_miss"])
                continue
            stats["positive_matched"] += 1

        relation_reference_target_mask = torch.zeros(len(detections), dtype=torch.bool)
        if sample.relation and sample.reference_boxes and reference_mask.any():
            ref_indices = torch.where(reference_mask)[0]
            matched_local, _ = best_match(boxes_px[ref_indices], sample.reference_boxes, args.min_match_iou)
            if matched_local >= 0:
                relation_reference_target_mask[int(ref_indices[matched_local])] = True
        if sample.relation and reference_mask.any():
            stats["relation_reference_present"] += 1

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
        torch.save(record, cache_path)
        stats["cached_new"] += 1
        stats["candidate_sum"] += len(detections)
        progress.set_postfix(new=stats["cached_new"], miss=stats["skipped_proposal_miss"])

    cached_files = len(list(output.glob("*.pt")))
    positive_recall = stats["positive_matched"] / max(stats["positive_total"], 1)
    report = {
        **stats,
        "cached_files": cached_files,
        "positive_proposal_recall_at_iou": positive_recall,
        "mean_positive_best_iou": (sum(proposal_ious) / len(proposal_ious)) if proposal_ious else None,
        "mean_candidates_new": stats["candidate_sum"] / max(stats["cached_new"], 1),
        "elapsed_s": time.perf_counter() - started,
        "config": str(Path(args.config).resolve()),
        "model": args.model or config.model.detector_name,
        "confidence": confidence,
        "top_k": max_candidates,
        "min_match_iou": args.min_match_iou,
    }
    (output / "_cache_stats.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[CACHE] ------------------------------------------------------------")
    print(json.dumps(report, indent=2))
    print("[CACHE] ============================================================")


if __name__ == "__main__":
    main()
