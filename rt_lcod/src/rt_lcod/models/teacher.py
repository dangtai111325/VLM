from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image


@dataclass(frozen=True)
class TeacherDetection:
    box: tuple[float, float, float, float]
    score: float
    label: str


class GroundingDINOTeacher:
    """Offline-only Grounding DINO teacher wrapper for pseudo labels/distillation."""

    def __init__(self, model_name: str = "IDEA-Research/grounding-dino-base",
                 device: str = "cuda", dtype: torch.dtype | None = None):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_name)
        kwargs = {}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name, **kwargs)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def predict(self, image: Image.Image, prompt: str, threshold: float = 0.25):
        text = prompt.strip()
        if not text.endswith("."):
            text += "."
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        raw = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=threshold,
            text_threshold=0.25,
            target_sizes=torch.tensor([image.size[::-1]], device=self.device),
        )[0]
        labels = raw.get("text_labels", raw.get("labels", []))
        detections = []
        for score, label, box in zip(raw["scores"], labels, raw["boxes"]):
            detections.append(TeacherDetection(
                box=tuple(float(x) for x in box.detach().cpu().tolist()),
                score=float(score),
                label=str(label),
            ))
        return detections
