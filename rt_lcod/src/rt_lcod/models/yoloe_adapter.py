from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    label: str


class YOLOECandidateGenerator:
    """Lazy Ultralytics YOLOE adapter using stable public detection outputs."""

    def __init__(self, model_name: str = "yoloe-26m.pt", device: str = "cuda", imgsz: int = 640):
        from ultralytics import YOLOE
        self.model = YOLOE(model_name).to(device)
        self.device = device
        self.imgsz = int(imgsz)
        self._classes: tuple[str, ...] | None = None

    def set_classes(self, classes: Iterable[str]) -> None:
        normalized = tuple(dict.fromkeys(str(c).strip() for c in classes if str(c).strip()))
        if not normalized:
            raise ValueError("at least one prompted class is required")
        if normalized != self._classes:
            self.model.set_classes(list(normalized))
            self._classes = normalized

    def predict(self, image: str | Path | np.ndarray, classes: Iterable[str], conf: float = 0.25):
        self.set_classes(classes)
        result = self.model.predict(
            image,
            conf=float(conf),
            imgsz=self.imgsz,
            device=0 if self.device.startswith("cuda") else "cpu",
            verbose=False,
        )[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            xyxy = tuple(float(x) for x in box.xyxy[0].detach().cpu().tolist())
            class_id = int(box.cls[0])
            detections.append(
                Detection(
                    box=xyxy, score=float(
                        box.conf[0]), label=str(
                        result.names[class_id])))
        return detections
