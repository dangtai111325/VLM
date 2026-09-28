from __future__ import annotations

import argparse
from pathlib import Path
import re

from PIL import Image
import torch
from tqdm.auto import tqdm

from rt_lcod.data.schema import load_manifest
from rt_lcod.geometry import box_iou_xyxy
from rt_lcod.models.teacher import GroundingDINOTeacher


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--temperature", type=float, default=0.20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = GroundingDINOTeacher(
        args.model,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    candidate_root = Path(args.candidate_cache)
    samples = load_manifest(args.manifest)

    cached_count = 0
    skipped = 0
    progress = tqdm(samples, desc="cache teacher", unit="sample", dynamic_ncols=True)
    for sample in progress:
        candidate_path = candidate_root / f"{safe_name(sample.sample_id)}.pt"
        if not candidate_path.exists():
            skipped += 1
            progress.set_postfix(cached=cached_count, skipped=skipped)
            continue
        cached = torch.load(candidate_path, map_location="cpu", weights_only=False)
        boxes_norm = cached["boxes"].float()
        boxes_px = boxes_norm.clone()
        boxes_px[:, [0, 2]] *= sample.width
        boxes_px[:, [1, 3]] *= sample.height
        image_path = (Path(args.manifest).parent / sample.image).resolve()
        if not image_path.exists():
            image_path = Path(sample.image).resolve()
        detections = teacher.predict(
            Image.open(image_path).convert("RGB"),
            sample.prompt,
            args.threshold)

        n = boxes_px.shape[0]
        candidate_scores = torch.zeros(n, dtype=torch.float32)
        for det in detections:
            teacher_box = torch.tensor(det.box, dtype=torch.float32)[None].expand(n, -1)
            iou = box_iou_xyxy(boxes_px, teacher_box)
            candidate_scores = torch.maximum(candidate_scores, iou * float(det.score))
        none_score = max(0.05, 1.0 - float(candidate_scores.max()) if n else 1.0)
        logits = torch.cat([candidate_scores, torch.tensor([none_score])]) / \
            max(args.temperature, 1e-4)
        distribution = torch.softmax(logits, dim=0)
        torch.save(
            {"sample_id": sample.sample_id, "teacher_distribution": distribution},
            output / f"{safe_name(sample.sample_id)}.pt",
        )
        cached_count += 1
        progress.set_postfix(cached=cached_count, skipped=skipped)
    print(f"done: total={len(samples)} cached={cached_count} skipped={skipped}")


if __name__ == "__main__":
    main()
