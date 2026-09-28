from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from rt_lcod.config import ExperimentConfig
from rt_lcod.models.region_encoder import build_region_encoder
from rt_lcod.models.student import RTLCODStudent
from rt_lcod.models.text_encoder import HFTextEncoder
from rt_lcod.models.yoloe_adapter import YOLOECandidateGenerator
from rt_lcod.prompt import ParsedPrompt, RuleSlotParser
from rt_lcod.training.checkpoint import load_checkpoint


@dataclass(frozen=True)
class RuntimeResult:
    box: tuple[float, float, float, float] | None
    confidence: float
    parsed_prompt: ParsedPrompt
    timings_ms: dict[str, float]
    candidate_count: int


class RTLCODRuntime:
    """End-to-end deployment wrapper with prompt/text caching."""

    def __init__(self, config: ExperimentConfig, checkpoint: str, device: str | None = None):
        requested = device or config.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.config = config
        self.detector = YOLOECandidateGenerator(
            config.model.detector_name, device=str(
                self.device), imgsz=int(
                config.runtime.get(
                    "input_size", 640)))
        self.region_encoder = build_region_encoder(
            config.model.region_encoder,
            visual_dim=config.model.visual_dim).to(
            self.device).eval()
        self.student = RTLCODStudent(config.model).to(self.device).eval()
        load_checkpoint(checkpoint, model=self.student, map_location=self.device, restore_rng=False)
        self.text = HFTextEncoder(config.model.text_model_name, device=str(self.device))
        self.parser = RuleSlotParser()
        self._prompt_text: str | None = None
        self._parsed: ParsedPrompt | None = None
        self._attribute_embedding: torch.Tensor | None = None
        self._relation_embedding: torch.Tensor | None = None

    def set_prompt(self, prompt: str) -> ParsedPrompt:
        if prompt == self._prompt_text and self._parsed is not None:
            return self._parsed
        parsed = self.parser.parse(prompt)
        self._attribute_embedding = self.text.encode([parsed.attribute or ""], device=self.device)
        self._relation_embedding = self.text.encode([parsed.relation or ""], device=self.device)
        self._prompt_text, self._parsed = prompt, parsed
        return parsed

    @torch.inference_mode()
    def process_frame(self, frame_bgr: np.ndarray, prompt: str | None = None) -> RuntimeResult:
        if prompt is not None:
            parsed = self.set_prompt(prompt)
        elif self._parsed is not None:
            parsed = self._parsed
        else:
            raise ValueError("set_prompt() or prompt= is required before process_frame")

        timings: dict[str, float] = {}
        classes = [parsed.target_class] + \
            ([parsed.reference_class] if parsed.reference_class else [])
        started = time.perf_counter()
        detections = self.detector.predict(
            frame_bgr, classes=classes, conf=float(
                self.config.runtime.get(
                    "confidence", 0.25)))
        detections = sorted(detections, key=lambda d: d.score, reverse=True)[
            : int(self.config.runtime.get("top_k", self.config.model.top_k))]
        timings["detector"] = (time.perf_counter() - started) * 1000
        if not detections:
            return RuntimeResult(None, 1.0, parsed, timings, 0)

        h, w = frame_bgr.shape[:2]
        boxes_px = torch.tensor([d.box for d in detections],
                                device=self.device, dtype=torch.float32)
        image_rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        image_tensor = torch.from_numpy(image_rgb).to(
            self.device).float().permute(
            2, 0, 1)[None] / 255.0

        started = time.perf_counter()
        features = self.region_encoder(image_tensor, [boxes_px])[0]
        timings["region_encoder"] = (time.perf_counter() - started) * 1000

        norm = boxes_px.clone()
        norm[:, [0, 2]] /= max(w, 1)
        norm[:, [1, 3]] /= max(h, 1)
        labels = [d.label.lower().strip() for d in detections]
        target_mask = torch.tensor([label == parsed.target_class.lower()
                                   for label in labels], dtype=torch.bool, device=self.device)
        reference_mask = torch.tensor([bool(parsed.reference_class) and label == parsed.reference_class.lower(
        ) for label in labels], dtype=torch.bool, device=self.device)
        batch = {
            "features": features[None], "boxes": norm[None],
            "detector_scores": torch.tensor([[d.score for d in detections]], device=self.device),
            "candidate_mask": torch.ones(1, len(detections), dtype=torch.bool, device=self.device),
            "target_mask": target_mask[None], "reference_mask": reference_mask[None],
            "attribute_embedding": self._attribute_embedding, "relation_embedding": self._relation_embedding,
            "has_attribute": torch.tensor([parsed.attribute is not None], device=self.device),
            "has_relation": torch.tensor([parsed.relation is not None], device=self.device),
        }
        started = time.perf_counter()
        output = self.student(batch)
        probabilities = torch.softmax(output.logits, dim=-1)[0]
        timings["grounding_head"] = (time.perf_counter() - started) * 1000
        index = int(torch.argmax(probabilities))
        timings["total_measured"] = sum(timings.values())
        if index == len(detections):
            return RuntimeResult(
                None,
                float(
                    probabilities[index]),
                parsed,
                timings,
                len(detections))
        return RuntimeResult(
            box=detections[index].box, confidence=float(probabilities[index]), parsed_prompt=parsed,
            timings_ms=timings, candidate_count=len(detections),
        )
