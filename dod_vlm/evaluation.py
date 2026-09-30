from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm.auto import tqdm

from .config import Config, Paths
from .loader import make_loader, model_forward, move_batch
from .model import DODVLM
from .utils import RunState, atomic_json_dump, stage_timer
from .vision import cache_path_for_image


def greedy_match_count(pred: torch.Tensor, gt: torch.Tensor, iou_threshold: float) -> tuple[int, int, int]:
    if len(pred) == 0 and len(gt) == 0:
        return 0, 0, 0
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    ious = box_iou(pred.cpu(), gt.cpu())
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    tp = 0
    while True:
        best_value = -1.0
        best_pair = None
        for i in range(ious.shape[0]):
            if i in used_pred:
                continue
            for j in range(ious.shape[1]):
                if j in used_gt:
                    continue
                value = float(ious[i, j])
                if value > best_value:
                    best_value, best_pair = value, (i, j)
        if best_pair is None or best_value < iou_threshold:
            break
        used_pred.add(best_pair[0]); used_gt.add(best_pair[1]); tp += 1
    return tp, len(pred) - tp, len(gt) - tp


@torch.inference_mode()
def collect_eval_records(
    cfg: Config, model: DODVLM, loader: Any, device: torch.device, show_progress: bool = True
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    iterator = tqdm(loader, desc="Collect logits", unit="batch", disable=not show_progress)
    for raw in iterator:
        batch = move_batch(raw, device)
        output = model_forward(model, batch)
        candidate_logits = output["candidate_logits"].detach().cpu()
        null_logits = output["null_logit"].detach().cpu()
        boxes = batch["boxes"].detach().cpu()
        masks = batch["output_mask"].detach().cpu()
        candidate_y = batch["candidate_y"].detach().cpu()
        for i in range(len(raw["sample_id"])):
            valid = masks[i]
            records.append({
                "sample_id": raw["sample_id"][i],
                "candidate_logits": candidate_logits[i][valid].clone(),
                "candidate_y": candidate_y[i][valid].clone(),
                "boxes": boxes[i][valid].clone(),
                "null_logit": float(null_logits[i]),
                "null_y": float(raw["null_y"][i]),
                "gt_boxes": raw["gt_boxes"][i].clone(),
            })
    return records


def _average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    values = []
    for recall_level in np.linspace(0.0, 1.0, 101):
        mask = recalls >= recall_level
        values.append(float(precisions[mask].max()) if mask.any() else 0.0)
    return float(np.mean(values))


def ap_at_iou(
    records: list[dict[str, Any]], iou_threshold: float, null_threshold: float,
    candidate_temperature: float, null_temperature: float
) -> float:
    predictions: list[tuple[float, int, torch.Tensor]] = []
    total_gt = sum(len(record["gt_boxes"]) for record in records)
    if total_gt == 0:
        return 0.0
    for sample_index, record in enumerate(records):
        null_prob = torch.sigmoid(
            torch.tensor(record["null_logit"] / max(null_temperature, 1e-6))
        ).item()
        if null_prob >= null_threshold:
            continue
        probs = torch.sigmoid(record["candidate_logits"] / max(candidate_temperature, 1e-6))
        for score, box in zip(probs.tolist(), record["boxes"]):
            predictions.append((float(score), sample_index, box))
    predictions.sort(key=lambda item: item[0], reverse=True)
    matched: dict[int, set[int]] = {}
    tp, fp = [], []
    for _, sample_index, box in predictions:
        gt = records[sample_index]["gt_boxes"]
        if len(gt) == 0:
            tp.append(0.0); fp.append(1.0); continue
        ious = box_iou(box[None].float(), gt.float())[0]
        used = matched.setdefault(sample_index, set())
        chosen = None
        for gt_index in torch.argsort(ious, descending=True).tolist():
            if gt_index not in used and float(ious[gt_index]) >= iou_threshold:
                chosen = gt_index; break
        if chosen is None:
            tp.append(0.0); fp.append(1.0)
        else:
            used.add(chosen); tp.append(1.0); fp.append(0.0)
    if not tp:
        return 0.0
    tp_cum = np.cumsum(np.asarray(tp)); fp_cum = np.cumsum(np.asarray(fp))
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return _average_precision(recalls, precisions)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int) -> float:
    if len(probabilities) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        include = (probabilities >= low) & (probabilities <= high if high == 1.0 else probabilities < high)
        if include.any():
            ece += include.mean() * abs(probabilities[include].mean() - labels[include].mean())
    return float(ece)


def score_threshold_metrics(
    records: list[dict[str, Any]], candidate_threshold: float, null_threshold: float,
    candidate_temperature: float, null_temperature: float, ece_bins: int
) -> dict[str, float]:
    tp = fp = fn = 0
    negative_total = negative_correct = false_boxes_negative = 0
    null_probs, null_labels = [], []
    for record in records:
        null_prob = float(torch.sigmoid(torch.tensor(record["null_logit"] / max(null_temperature, 1e-6))))
        null_probs.append(null_prob); null_labels.append(float(record["null_y"]))
        if null_prob >= null_threshold:
            pred_boxes = record["boxes"][:0]
        else:
            probs = torch.sigmoid(record["candidate_logits"] / max(candidate_temperature, 1e-6))
            pred_boxes = record["boxes"][probs >= candidate_threshold]
        gt = record["gt_boxes"]
        a, b, c = greedy_match_count(pred_boxes, gt, 0.5)
        tp += a; fp += b; fn += c
        if len(gt) == 0:
            negative_total += 1
            negative_correct += int(len(pred_boxes) == 0)
            false_boxes_negative += len(pred_boxes)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    probs_np = np.asarray(null_probs, dtype=np.float64)
    labels_np = np.asarray(null_labels, dtype=np.float64)
    return {
        "set_precision@0.5": float(precision),
        "set_recall@0.5": float(recall),
        "set_f1@0.5": float(f1),
        "negative_reject_accuracy": negative_correct / max(negative_total, 1),
        "false_boxes_per_negative": false_boxes_negative / max(negative_total, 1),
        "null_brier": float(np.mean((probs_np - labels_np) ** 2)),
        "null_ece": expected_calibration_error(probs_np, labels_np, ece_bins),
        "samples": float(len(records)),
    }


def score_records(
    records: list[dict[str, Any]], candidate_threshold: float, null_threshold: float,
    candidate_temperature: float, null_temperature: float, ece_bins: int
) -> dict[str, float]:
    base = score_threshold_metrics(
        records, candidate_threshold, null_threshold, candidate_temperature, null_temperature, ece_bins
    )
    ap50 = ap_at_iou(records, 0.50, null_threshold, candidate_temperature, null_temperature)
    ap75 = ap_at_iou(records, 0.75, null_threshold, candidate_temperature, null_temperature)
    ap_grid = [
        ap_at_iou(records, float(t), null_threshold, candidate_temperature, null_temperature)
        for t in np.arange(0.50, 0.96, 0.05)
    ]
    return {**base, "AP50": ap50, "AP75": ap75, "mAP@0.5:0.95": float(np.mean(ap_grid))}


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 1.0
    logits = logits.float().cpu().flatten(); labels = labels.float().cpu().flatten()
    if logits.numel() > 500_000:
        indices = torch.randperm(logits.numel(), generator=torch.Generator().manual_seed(1337))[:500_000]
        logits, labels = logits[indices], labels[indices]
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=80)
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward(); return loss
    try:
        optimizer.step(closure)
        return float(log_temperature.exp().clamp(0.05, 20.0).detach())
    except Exception:
        return 1.0


def calibrate(cfg: Config, records: list[dict[str, Any]]) -> tuple[dict[str, float], pd.DataFrame]:
    logit_parts = [r["candidate_logits"] for r in records if r["candidate_logits"].numel()]
    label_parts = [r["candidate_y"] for r in records if r["candidate_y"].numel()]
    candidate_logits = torch.cat(logit_parts) if logit_parts else torch.empty(0)
    candidate_labels = torch.cat(label_parts) if label_parts else torch.empty(0)
    null_logits = torch.tensor([r["null_logit"] for r in records], dtype=torch.float32)
    null_labels = torch.tensor([r["null_y"] for r in records], dtype=torch.float32)
    ct_temp = _fit_temperature(candidate_logits, candidate_labels)
    nt_temp = _fit_temperature(null_logits, null_labels)

    grid = torch.linspace(0.15, 0.85, cfg.threshold_grid_size).tolist()
    rows = []
    best = {
        "candidate_threshold": cfg.default_candidate_threshold,
        "null_threshold": cfg.default_null_threshold,
        "candidate_temperature": ct_temp,
        "null_temperature": nt_temp,
        "score": -1.0,
        "negative_reject_accuracy": -1.0,
        "false_boxes_per_negative": float("inf"),
    }
    for candidate_threshold in tqdm(grid, desc="Calibration", unit="cand_thr"):
        for null_threshold in grid:
            metrics = score_threshold_metrics(
                records, float(candidate_threshold), float(null_threshold), ct_temp, nt_temp, cfg.ece_bins
            )
            rows.append({"candidate_threshold": float(candidate_threshold), "null_threshold": float(null_threshold), **metrics})
            key = (metrics["set_f1@0.5"], metrics["negative_reject_accuracy"], -metrics["false_boxes_per_negative"])
            old = (best["score"], best["negative_reject_accuracy"], -best["false_boxes_per_negative"])
            if key > old:
                best.update({
                    "candidate_threshold": float(candidate_threshold),
                    "null_threshold": float(null_threshold),
                    "score": float(metrics["set_f1@0.5"]),
                    "negative_reject_accuracy": float(metrics["negative_reject_accuracy"]),
                    "false_boxes_per_negative": float(metrics["false_boxes_per_negative"]),
                })
    return best, pd.DataFrame(rows)


def evaluate_and_calibrate(
    cfg: Config, paths: Paths, state: RunState, manifest: pd.DataFrame, feature_dir: Path,
    model: DODVLM, tokenizer: Any, best_ckpt: Path, batch_size: int, device: torch.device
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[dict[str, Any]]]:
    best = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"]); model.to(device).eval()
    val_loader = make_loader(cfg, manifest, feature_dir, tokenizer, "val", False, False, batch_size)
    with stage_timer("Session 7A — Cache validation logits"):
        val_records = collect_eval_records(cfg, model, val_loader, device)
    with stage_timer("Session 7B — Temperature + threshold calibration"):
        calibration, table = calibrate(cfg, val_records)
        atomic_json_dump(calibration, paths.run / "calibration.json")
        table.to_csv(paths.run / "calibration_grid.csv", index=False)
        print(json.dumps(calibration, indent=2))

    test_splits = [name for name in ["test", "testA", "testB"] if name in set(manifest["split"])]
    if not test_splits:
        print("⚠ Không có held-out test split; dùng val chỉ để sanity check.")
        test_splits = ["val"]
    all_metrics: dict[str, dict[str, float]] = {}
    first_records: list[dict[str, Any]] = []
    for split in test_splits:
        loader = make_loader(cfg, manifest, feature_dir, tokenizer, split, False, False, batch_size)
        with stage_timer(f"Session 7C — Final evaluation: {split}"):
            records = collect_eval_records(cfg, model, loader, device)
            metrics = score_records(
                records, calibration["candidate_threshold"], calibration["null_threshold"],
                calibration["candidate_temperature"], calibration["null_temperature"], cfg.ece_bins
            )
            all_metrics[split] = metrics
            print(split, json.dumps(metrics, indent=2))
            if not first_records:
                first_records = records
    atomic_json_dump(all_metrics, paths.run / "final_metrics.json")
    state.mark("evaluation_complete", metrics=all_metrics)
    return calibration, all_metrics, first_records


def proposal_diagnostics(
    cfg: Config, manifest: pd.DataFrame, feature_dir: Path, split: str = "val"
) -> dict[str, float]:
    subset = manifest[(manifest["split"] == split) & (~manifest["no_target"])].reset_index(drop=True)
    if subset.empty:
        return {}
    hits = {0.50: 0, 0.75: 0}; query_hits = {0.50: 0, 0.75: 0}
    total_gt = total_queries = 0; recoverability = []
    for row in tqdm(subset.itertuples(), total=len(subset), desc=f"Proposal diag {split}", unit="query"):
        cache = torch.load(cache_path_for_image(feature_dir, int(row.image_id)), map_location="cpu", weights_only=False)
        boxes = cache["boxes"][~cache["fallback_mask"].bool()]
        gt = torch.tensor(row.gt_boxes, dtype=torch.float32).reshape(-1, 4)
        total_gt += len(gt); total_queries += 1
        if len(boxes) == 0 or len(gt) == 0:
            recoverability.append(0.0); continue
        max_iou = box_iou(gt, boxes).max(dim=1).values
        for threshold in [0.50, 0.75]:
            hits[threshold] += int((max_iou >= threshold).sum())
            query_hits[threshold] += int(bool((max_iou >= threshold).all()))
        recoverability.append(float(torch.stack([
            (max_iou >= t).float().mean() for t in torch.arange(0.5, 0.96, 0.05)
        ]).mean()))
    return {
        "target_recall@0.5": hits[0.50] / max(total_gt, 1),
        "target_recall@0.75": hits[0.75] / max(total_gt, 1),
        "query_all_targets_recall@0.5": query_hits[0.50] / max(total_queries, 1),
        "query_all_targets_recall@0.75": query_hits[0.75] / max(total_queries, 1),
        "mean_recoverability@0.5:0.95": float(np.mean(recoverability)) if recoverability else 0.0,
        "queries": float(total_queries),
    }
