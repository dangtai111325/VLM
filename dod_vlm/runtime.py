from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import statistics
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer
from ultralytics import YOLOE

from .config import Config, Paths
from .model import DODVLM
from .query import parse_query
from .utils import RunState, atomic_json_dump, atomic_torch_save, stage_timer
from .vision import FrozenRegionEncoder


def _pad_1d(x: torch.Tensor, length: int, value: float = 0.0) -> torch.Tensor:
    out = x.new_full((length,), value)
    out[: min(length, len(x))] = x[:length]
    return out


def _pad_2d(x: torch.Tensor, length: int, width: int, value: float = 0.0) -> torch.Tensor:
    out = x.new_full((length, width), value)
    count = min(length, x.shape[0])
    out[:count] = x[:count]
    return out


def export_final_artifact(
    cfg: Config,
    paths: Paths,
    state: RunState,
    model: DODVLM,
    tokenizer: Any,
    calibration: dict[str, float],
    metrics: dict[str, dict[str, float]],
    proposal_local: Path,
    region_local: Path,
    cache_signature: str,
) -> Path:
    artifact = paths.artifact
    artifact.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = artifact / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    proposal_dest = artifact / "proposal_model.pt"
    region_dest = artifact / "region_encoder.pt"
    shutil.copy2(proposal_local, proposal_dest)
    shutil.copy2(region_local, region_dest)

    core_path = artifact / "dod_vlm_core.pt"
    payload = {
        "format_version": 2,
        "config": asdict(cfg),
        "config_hash": cfg.signature(),
        "cache_signature": cache_signature,
        "text_config": model.text_encoder.config.to_dict(),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "calibration": calibration,
        "metrics": metrics,
        "assets": {
            "tokenizer": "tokenizer",
            "proposal_model": "proposal_model.pt",
            "region_encoder": "region_encoder.pt",
        },
        "architecture": {
            "proposal": "YOLOE-26s prompt-free",
            "object_encoder": "FasterRCNN MobileNetV3-Large-FPN + MultiScaleRoIAlign",
            "text_encoder": cfg.text_model,
            "fusion": "A0 generic + A1 structured relational + learned gate",
            "output": "independent candidates + explicit null head",
        },
    }
    atomic_torch_save(payload, core_path)
    atomic_json_dump(asdict(cfg), artifact / "config.json")
    atomic_json_dump(calibration, artifact / "calibration.json")
    atomic_json_dump(metrics, artifact / "metrics.json")

    readme = artifact / "README.txt"
    readme.write_text(
        "Self-contained DOD-VLM artifact. Keep this entire directory together.\n"
        "Load with DODVLMRuntime(path_to_this_directory, device).\n",
        encoding="utf-8",
    )
    archive_base = paths.run / "dod_vlm_final_bundle"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", artifact))
    state.mark(
        "final_artifact_saved",
        directory=str(artifact),
        archive=str(archive_path),
    )
    return artifact


class DODVLMRuntime:
    def __init__(self, artifact_dir: str | Path, device: torch.device):
        self.artifact_dir = Path(artifact_dir)
        self.device = device
        core = torch.load(
            self.artifact_dir / "dod_vlm_core.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.cfg = Config(**core["config"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.artifact_dir / core["assets"]["tokenizer"],
            local_files_only=True,
        )
        self.model = DODVLM(
            self.cfg,
            pretrained_text=False,
            text_config=core["text_config"],
        ).to(device)
        self.model.load_state_dict(core["model_state"])
        self.model.eval()

        self.proposal = YOLOE(str(self.artifact_dir / core["assets"]["proposal_model"]))
        self.region_encoder = FrozenRegionEncoder(
            self.cfg,
            device,
            state_path=self.artifact_dir / core["assets"]["region_encoder"],
        )
        self.calibration = core["calibration"]

    @torch.inference_mode()
    def predict(self, image: str | Path | Image.Image, query: str) -> list[dict[str, Any]]:
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            pil = Image.open(image_path).convert("RGB")
            source: Any = str(image_path)
        else:
            pil = image.convert("RGB")
            source = np.asarray(pil)

        result = self.proposal.predict(
            source=source,
            imgsz=self.cfg.proposal_imgsz,
            conf=self.cfg.proposal_conf,
            max_det=self.cfg.top_k,
            device=0,
            half=True,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            boxes = torch.tensor(
                [[0.0, 0.0, float(pil.width), float(pil.height)]],
                dtype=torch.float32,
            )
            scores = torch.zeros((1,), dtype=torch.float32)
            labels = ["__fallback_context__"]
            output_mask_base = torch.zeros((1,), dtype=torch.bool)
        else:
            boxes = result.boxes.xyxy.detach().cpu().float()[: self.cfg.top_k]
            scores = result.boxes.conf.detach().cpu().float()[: self.cfg.top_k]
            class_ids = result.boxes.cls.detach().cpu().long().tolist()[: self.cfg.top_k]
            labels = [str(result.names[int(index)]) for index in class_ids]
            output_mask_base = torch.ones((len(boxes),), dtype=torch.bool)

        features = self.region_encoder.encode(pil, boxes)
        parsed = parse_query(query)
        query_tok = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=self.cfg.max_text_len,
            return_tensors="pt",
        )
        target_tok = self.tokenizer(
            [parsed["target_phrase"]],
            padding=True,
            truncation=True,
            max_length=16,
            return_tensors="pt",
        )
        anchor_tok = self.tokenizer(
            [parsed["anchor_phrase"] or "[UNK]"],
            padding=True,
            truncation=True,
            max_length=16,
            return_tensors="pt",
        )

        k = self.cfg.top_k
        count = min(k, len(boxes))
        visual_features = _pad_2d(features, k, self.cfg.visual_dim)[None].to(self.device)
        boxes_pad = _pad_2d(boxes, k, 4)[None].to(self.device)
        scores_pad = _pad_1d(scores, k)[None].to(self.device)
        candidate_mask = torch.zeros((1, k), dtype=torch.bool, device=self.device)
        candidate_mask[:, :count] = True
        output_mask = torch.zeros((1, k), dtype=torch.bool, device=self.device)
        output_mask[:, :count] = output_mask_base[:count].to(self.device)
        image_sizes = torch.tensor(
            [[pil.height, pil.width]],
            dtype=torch.float32,
            device=self.device,
        )

        output = self.model(
            visual_features,
            boxes_pad,
            scores_pad,
            candidate_mask,
            output_mask,
            image_sizes,
            query_tok["input_ids"].to(self.device),
            query_tok["attention_mask"].to(self.device),
            target_tok["input_ids"].to(self.device),
            target_tok["attention_mask"].to(self.device),
            anchor_tok["input_ids"].to(self.device),
            anchor_tok["attention_mask"].to(self.device),
            torch.tensor([parsed["relation_id"]], dtype=torch.long, device=self.device),
            torch.tensor([parsed["has_relation"]], dtype=torch.bool, device=self.device),
        )

        candidate_temperature = max(float(self.calibration["candidate_temperature"]), 1e-6)
        null_temperature = max(float(self.calibration["null_temperature"]), 1e-6)
        candidate_threshold = float(self.calibration["candidate_threshold"])
        null_threshold = float(self.calibration["null_threshold"])

        null_probability = float(
            torch.sigmoid(output["null_logit"][0] / null_temperature)
        )
        if null_probability >= null_threshold:
            return []

        probabilities = torch.sigmoid(
            output["candidate_logits"][0, :count] / candidate_temperature
        ).cpu()
        keep = torch.where(
            (probabilities >= candidate_threshold) & output_mask_base[:count]
        )[0].tolist()
        return [
            {
                "box": [round(float(value), 2) for value in boxes[index].tolist()],
                "score": round(float(probabilities[index]), 4),
                "proposal_label": labels[index],
                "proposal_score": round(float(scores[index]), 4),
                "null_probability": round(null_probability, 4),
            }
            for index in keep
        ]

    def benchmark(
        self,
        image: str | Path | Image.Image,
        query: str,
        warmup: int,
        runs: int,
    ) -> dict[str, float]:
        for _ in range(warmup):
            self.predict(image, query)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_ms: list[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            self.predict(image, query)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000.0)
        ordered = sorted(times_ms)
        p95_index = min(len(ordered) - 1, max(0, int(np.ceil(0.95 * len(ordered))) - 1))
        p50 = statistics.median(ordered)
        p95 = ordered[p95_index]
        return {
            "latency_p50_ms": float(p50),
            "latency_p95_ms": float(p95),
            "fps_from_p50": float(1000.0 / max(p50, 1e-6)),
            "runs": float(runs),
        }


def reload_and_smoke_test(
    cfg: Config,
    paths: Paths,
    state: RunState,
    artifact_dir: Path,
    manifest: Any,
    device: torch.device,
) -> tuple[DODVLMRuntime, dict[str, Any], dict[str, float]]:
    with stage_timer("Session 9 — Reload artifact + inference + latency benchmark"):
        runtime = DODVLMRuntime(artifact_dir, device)
        test_splits = [
            name
            for name in ["test", "testA", "testB", "val"]
            if name in set(manifest["split"])
        ]
        if not test_splits:
            raise RuntimeError("Không có split nào để chạy inference sanity check.")
        row = manifest[manifest["split"] == test_splits[0]].iloc[0]
        image_path = paths.images / row.file_name
        query = str(row.query)
        prediction = runtime.predict(image_path, query)
        benchmark = runtime.benchmark(
            image_path,
            query,
            warmup=cfg.runtime_warmup_runs,
            runs=cfg.runtime_benchmark_runs,
        )
        result = {
            "image": str(image_path),
            "query": query,
            "prediction": prediction,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(benchmark, indent=2))
        atomic_json_dump(benchmark, paths.run / "runtime_benchmark.json")
        state.mark("reload_inference_passed", sample_id=str(row.sample_id), benchmark=benchmark)
        return runtime, result, benchmark
