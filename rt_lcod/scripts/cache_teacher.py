from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

from PIL import Image
import torch
from tqdm.auto import tqdm

from rt_lcod.data.schema import load_manifest
from rt_lcod.geometry import box_iou_xyxy
from rt_lcod.models.teacher import GroundingDINOTeacher


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def resolve_image_path(manifest: str | Path, image: str) -> Path:
    manifest = Path(manifest).resolve()
    raw = Path(image)
    for candidate in (raw, manifest.parent / raw, manifest.parent.parent / raw):
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot resolve image={image!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all cached train samples")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[TEACHER] ==========================================================")
    print(f"[TEACHER] model={args.model} device={device} threshold={args.threshold} temperature={args.temperature}")
    teacher = GroundingDINOTeacher(
        args.model,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    candidate_root = Path(args.candidate_cache)
    samples = load_manifest(args.manifest)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"[TEACHER] requested_samples={len(samples):,} candidate_cache={candidate_root.resolve()}")

    stats = {"total": len(samples), "cached_new": 0, "cached_existing": 0, "missing_candidate": 0}
    teacher_detection_sum = 0
    progress_every = max(1, len(samples) // 20)
    last_reported = 0

    def report_progress(processed: int) -> None:
        nonlocal last_reported
        if processed - last_reported < progress_every and processed != len(samples):
            return
        elapsed = time.perf_counter() - started
        rate = processed / max(elapsed, 1e-6)
        eta_s = (len(samples) - processed) / max(rate, 1e-6)
        print(
            f"[PROGRESS] teacher={processed}/{len(samples)} "
            f"completed={100 * processed / max(len(samples), 1):.0f}% "
            f"new={stats['cached_new']} existing={stats['cached_existing']} "
            f"missing_candidate={stats['missing_candidate']} "
            f"rate={rate:.2f}_sample/s eta_m={eta_s / 60:.1f}",
            flush=True,
        )
        last_reported = processed

    progress = tqdm(samples, desc="cache teacher", unit="sample", dynamic_ncols=True)
    for index, sample in enumerate(progress, start=1):
        out_path = output / f"{safe_name(sample.sample_id)}.pt"
        if out_path.exists() and not args.no_resume:
            stats["cached_existing"] += 1
            report_progress(index)
            continue
        candidate_path = candidate_root / f"{safe_name(sample.sample_id)}.pt"
        if not candidate_path.exists():
            stats["missing_candidate"] += 1
            report_progress(index)
            continue
        cached = torch.load(candidate_path, map_location="cpu", weights_only=False)
        boxes_norm = cached["boxes"].float()
        boxes_px = boxes_norm.clone()
        boxes_px[:, [0, 2]] *= sample.width
        boxes_px[:, [1, 3]] *= sample.height
        image_path = resolve_image_path(args.manifest, sample.image)
        detections = teacher.predict(Image.open(image_path).convert("RGB"), sample.prompt, args.threshold)
        teacher_detection_sum += len(detections)

        n = boxes_px.shape[0]
        candidate_scores = torch.zeros(n, dtype=torch.float32)
        for det in detections:
            teacher_box = torch.tensor(det.box, dtype=torch.float32)[None].expand(n, -1)
            iou = box_iou_xyxy(boxes_px, teacher_box)
            candidate_scores = torch.maximum(candidate_scores, iou * float(det.score))
        none_score = max(0.05, 1.0 - float(candidate_scores.max()) if n else 1.0)
        logits = torch.cat([candidate_scores, torch.tensor([none_score])]) / max(args.temperature, 1e-4)
        distribution = torch.softmax(logits, dim=0)
        torch.save({"sample_id": sample.sample_id, "teacher_distribution": distribution}, out_path)
        stats["cached_new"] += 1
        progress.set_postfix(new=stats["cached_new"], missing=stats["missing_candidate"])
        report_progress(index)

    report = {
        **stats,
        "cache_files": len(list(output.glob("*.pt"))),
        "mean_teacher_detections_new": teacher_detection_sum / max(stats["cached_new"], 1),
        "elapsed_s": time.perf_counter() - started,
        "model": args.model,
    }
    (output / "_teacher_stats.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("[TEACHER] ==========================================================")


if __name__ == "__main__":
    main()
