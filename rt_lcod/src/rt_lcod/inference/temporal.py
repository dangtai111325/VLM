from __future__ import annotations

from dataclasses import dataclass

import torch

from rt_lcod.geometry import box_iou_xyxy


@dataclass
class TrackState:
    box: tuple[float, float, float, float]
    score: float
    track_id: int
    missed: int = 0


class SimpleTargetTracker:
    """Single-target IoU tracker for practical robot deployment experiments."""

    def __init__(self, iou_gate: float = 0.3, max_missed: int = 5, score_decay: float = 0.95):
        self.iou_gate = float(iou_gate)
        self.max_missed = int(max_missed)
        self.score_decay = float(score_decay)
        self.state: TrackState | None = None
        self._next_id = 1

    def reset(self) -> None:
        self.state = None

    def initialize(self, box, score: float) -> TrackState:
        self.state = TrackState(tuple(map(float, box)), float(score), self._next_id, 0)
        self._next_id += 1
        return self.state

    def update(self, detections: list[tuple[tuple[float, float, float, float], float]]) -> TrackState | None:
        if self.state is None:
            if not detections:
                return None
            box, score = max(detections, key=lambda x: x[1])
            return self.initialize(box, score)
        if not detections:
            self.state.missed += 1
            self.state.score *= self.score_decay
            if self.state.missed > self.max_missed:
                self.reset()
            return self.state
        current = torch.tensor(self.state.box, dtype=torch.float32)
        boxes = torch.tensor([d[0] for d in detections], dtype=torch.float32)
        ious = box_iou_xyxy(current[None, :], boxes)
        best = int(torch.argmax(ious))
        if float(ious[best]) < self.iou_gate:
            self.state.missed += 1
            self.state.score *= self.score_decay
            if self.state.missed > self.max_missed:
                self.reset()
            return self.state
        box, score = detections[best]
        self.state.box = tuple(map(float, box))
        self.state.score = float(score)
        self.state.missed = 0
        return self.state
