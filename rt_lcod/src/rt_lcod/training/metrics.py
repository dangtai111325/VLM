from __future__ import annotations

import torch


@torch.no_grad()
def batch_metrics(logits: torch.Tensor, target_index: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    no_target_index = logits.shape[1] - 1
    correct = (pred == target_index).float().mean().item()
    negatives = target_index == no_target_index
    positives = ~negatives
    no_target_acc = (pred[negatives] == no_target_index).float(
    ).mean().item() if negatives.any() else float("nan")
    positive_acc = (pred[positives] == target_index[positives]).float(
    ).mean().item() if positives.any() else float("nan")
    false_positive_rate = (pred[negatives] != no_target_index).float(
    ).mean().item() if negatives.any() else float("nan")
    return {
        "target_top1_accuracy": correct,
        "positive_target_accuracy": positive_acc,
        "no_target_accuracy": no_target_acc,
        "negative_false_positive_rate": false_positive_rate,
    }
