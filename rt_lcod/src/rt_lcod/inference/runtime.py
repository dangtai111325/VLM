from __future__ import annotations

from contextlib import nullcontext
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
    """End-to-end deployment wrapper with prompt caching and measured T4 latency."""

    def __init__(
        self,
        config: ExperimentConfig,
        checkpoint: str,
        device: str | None = None,
    ):
        requested = device or config.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.config = config
        self.use_amp = bool(self.device.type == "cuda")

        self.detector = YOLOECandidateGenerator(
            config.model.detector_name,
            device=str(self.device),
            imgsz=int(config.runtime.get("input_size", 512)),
        )
        self.region_encoder = build_region_encoder(
            config.model.region_encoder,
            visual_dim=config.model.visual_dim,
        ).to(self.device).eval()
        self.student = RTLCODStudent(config.model).to(self.device).eval()
        load_checkpoint(
            checkpoint,
            model=self.student,
            map_location=self.device,
            restore_rng=False,
        )
        self.text = HFTextEncoder(config.model.text_model_name, device=str(self.device))
        self.parser = RuleSlotParser()

        self._prompt_text: str | None = None
        self._parsed: ParsedPrompt | None = None
        self._attribute_embedding: torch.Tensor | None = None
        self._relation_embedding: torch.Tensor | None = None

        print(
            f"[RUNTIME] device={self.device} amp_fp16={self.use_amp} "
            f"detector={config.model.detector_name} input={config.runtime.get('input_size', 512)} "
            f"top_k={config.runtime.get('top_k', config.model.top_k)}"
        )

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def set_prompt(self, prompt: str) -> ParsedPrompt:
        if prompt == self._prompt_text and self._parsed is not None:
            return self._parsed
        parsed = self.parser.parse(prompt)
        self._attribute_embedding = self.text.encode(
            [parsed.attribute or ""],
            device=self.device,
        )
        self._relation_embedding = self.text.encode(
            [parsed.relation or ""],
            device=self.device,
        )
        self._prompt_text = prompt
        self._parsed = parsed
        print(
            f"[PROMPT] raw={prompt!r} target={parsed.target_class!r} "
            f"attribute={parsed.attribute!r} relation={parsed.relation!r} "
            f"reference={parsed.reference_class!r}"
        )
        return parsed

    @torch.inference_mode()
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        prompt: str | None = None,
    ) -> RuntimeResult:
        timings: dict[str, float] = {}

        prompt_changed = prompt is not None and prompt != self._prompt_text
        prompt_started = time.perf_counter()
        if prompt is not None:
            parsed = self.set_prompt(prompt)
        elif self._parsed is not None:
            parsed = self._parsed
        else:
            raise ValueError("set_prompt() or prompt= is required before process_frame")
        if prompt_changed:
            self._sync()
            timings["prompt_change"] = (time.perf_counter() - prompt_started) * 1000.0

        classes = [parsed.target_class]
        if parsed.reference_class:
            classes.append(parsed.reference_class)

        self._sync()
        detector_started = time.perf_counter()
        detections = self.detector.predict(
            frame_bgr,
            classes=classes,
            conf=float(self.config.runtime.get("confidence", 0.15)),
        )
        self._sync()
        timings["detector"] = (time.perf_counter() - detector_started) * 1000.0

        detections = sorted(
            detections,
            key=lambda detection: detection.score,
            reverse=True,
        )[: int(self.config.runtime.get("top_k", self.config.model.top_k))]

        if not detections:
            timings["total_measured"] = sum(timings.values())
            return RuntimeResult(
                box=None,
                confidence=1.0,
                parsed_prompt=parsed,
                timings_ms=timings,
                candidate_count=0,
            )

        height, width = frame_bgr.shape[:2]
        boxes_px = torch.tensor(
            [detection.box for detection in detections],
            device=self.device,
            dtype=torch.float32,
        )
        image_rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        image_tensor = (
            torch.from_numpy(image_rgb)
            .to(self.device)
            .float()
            .permute(2, 0, 1)[None]
            / 255.0
        )

        self._sync()
        region_started = time.perf_counter()
        with self._autocast():
            features = self.region_encoder(image_tensor, [boxes_px])[0]
        self._sync()
        timings["region_encoder"] = (time.perf_counter() - region_started) * 1000.0

        normalized_boxes = boxes_px.clone()
        normalized_boxes[:, [0, 2]] /= max(width, 1)
        normalized_boxes[:, [1, 3]] /= max(height, 1)
        labels = [detection.label.lower().strip() for detection in detections]
        target_name = parsed.target_class.lower().strip()
        reference_name = (parsed.reference_class or "").lower().strip()
        target_mask = torch.tensor(
            [label == target_name for label in labels],
            dtype=torch.bool,
            device=self.device,
        )
        reference_mask = torch.tensor(
            [bool(reference_name) and label == reference_name for label in labels],
            dtype=torch.bool,
            device=self.device,
        )
        batch = {
            "features": features[None],
            "boxes": normalized_boxes[None],
            "detector_scores": torch.tensor(
                [[detection.score for detection in detections]],
                device=self.device,
            ),
            "candidate_mask": torch.ones(
                1,
                len(detections),
                dtype=torch.bool,
                device=self.device,
            ),
            "target_mask": target_mask[None],
            "reference_mask": reference_mask[None],
            "attribute_embedding": self._attribute_embedding,
            "relation_embedding": self._relation_embedding,
            "has_attribute": torch.tensor(
                [parsed.attribute is not None],
                device=self.device,
            ),
            "has_relation": torch.tensor(
                [parsed.relation is not None],
                device=self.device,
            ),
        }

        self._sync()
        grounding_started = time.perf_counter()
        with self._autocast():
            output = self.student(batch)
            probabilities = torch.softmax(output.logits.float(), dim=-1)[0]
        self._sync()
        timings["grounding_head"] = (time.perf_counter() - grounding_started) * 1000.0
        timings["total_measured"] = sum(
            value for key, value in timings.items() if key != "prompt_change"
        )

        selected_index = int(torch.argmax(probabilities).item())
        if selected_index == len(detections):
            return RuntimeResult(
                box=None,
                confidence=float(probabilities[selected_index].item()),
                parsed_prompt=parsed,
                timings_ms=timings,
                candidate_count=len(detections),
            )

        return RuntimeResult(
            box=detections[selected_index].box,
            confidence=float(probabilities[selected_index].item()),
            parsed_prompt=parsed,
            timings_ms=timings,
            candidate_count=len(detections),
        )
