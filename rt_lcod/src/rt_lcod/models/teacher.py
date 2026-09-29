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
            kwargs["dtype"] = dtype
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name, **kwargs)
        self.model = self.model.to(device).eval()
        self.device = device
        self.dtype = next(self.model.parameters()).dtype

    @torch.inference_mode()
    def predict(self, image: Image.Image, prompt: str, threshold: float = 0.25):
        text = prompt.strip()
        if not text.endswith("."):
            text += "."
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        # The processor creates pixel_values as FP32.  On CUDA the teacher is
        # deliberately loaded in FP16 to fit comfortably on the local RTX A3000,
        # so only floating tensors must be converted to the model dtype.  Token
        # IDs and masks must retain their integer/bool dtypes.
        inputs = {
            key: value.to(self.device, dtype=self.dtype)
            if value.is_floating_point()
            else value.to(self.device)
            for key, value in inputs.items()
        }
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
