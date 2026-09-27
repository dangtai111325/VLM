from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


class Adapter:
    def __init__(self, key, config, ctx):
        self.key, self.c, self.ctx, self.model = key, config[key], ctx, None

    def unload(self):
        self.model = None
        if self.ctx["DEVICE"] == "cuda":
            torch.cuda.empty_cache()

    def predict(self, frame, conf, prompt=""):
        raise NotImplementedError


class YOLOAdapter(Adapter):
    def prepare(self, progress):
        from ultralytics import YOLO
        weight = Path(self.c["weight"])
        weight.parent.mkdir(parents=True, exist_ok=True)
        if not weight.exists():
            progress.note(f"Downloading {weight.name}")
            cwd = Path.cwd()
            try:
                os.chdir(weight.parent)
                YOLO(weight.name)
            finally:
                os.chdir(cwd)
        self.model = YOLO(str(weight)).to(self.ctx["DEVICE"])

    def predict(self, frame, conf, prompt=""):
        started = time.perf_counter()
        pred = self.model.predict(
            frame,
            conf=conf,
            imgsz=640,
            device=0 if self.ctx["DEVICE"] == "cuda" else "cpu",
            verbose=False,
        )[0]
        dets = []
        if pred.boxes is not None:
            for box in pred.boxes:
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
                cid = int(box.cls[0])
                dets.append(
                    self.ctx["Det"](
                        str(pred.names[cid]),
                        float(box.conf[0]),
                        x1, y1, x2, y2,
                    )
                )
        return self.ctx["Result"](dets, (time.perf_counter() - started) * 1000)


class RFDETRAdapter(Adapter):
    def prepare(self, progress):
        from rfdetr import RFDETRLarge
        from rfdetr.assets.coco_classes import COCO_CLASSES
        progress.note(f"Loading {self.c['name']}")
        if self.c["variant"] == "large":
            self.model = RFDETRLarge()
        else:
            from rfdetr_plus import RFDETRXLarge
            self.model = RFDETRXLarge(accept_platform_model_license=True)
        self.labels = COCO_CLASSES

    def predict(self, frame, conf, prompt=""):
        started = time.perf_counter()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        raw = self.model.predict(image, threshold=conf)
        dets = []
        for box, score, cid in zip(raw.xyxy, raw.confidence, raw.class_id):
            x1, y1, x2, y2 = map(float, box)
            i = int(cid)
            dets.append(
                self.ctx["Det"](
                    self.labels[i] if i < len(self.labels) else str(i),
                    float(score), x1, y1, x2, y2,
                )
            )
        return self.ctx["Result"](dets, (time.perf_counter() - started) * 1000)


class GroundingDinoAdapter(Adapter):
    def prepare(self, progress):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        repo = self.c["repo"]
        progress.note(f"Loading {repo}")
        self.processor = AutoProcessor.from_pretrained(
            repo,
            cache_dir=str(self.ctx["HF"] / "hub"),
        )
        kwargs = {"cache_dir": str(self.ctx["HF"] / "hub")}
        if self.ctx["DEVICE"] == "cuda":
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(repo, **kwargs)
        self.model = self.model.to(self.ctx["DEVICE"]).eval()

    def predict(self, frame, conf, prompt=""):
        if not prompt.strip():
            raise ValueError("Grounding DINO requires a text prompt")
        started = time.perf_counter()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text = prompt.strip()
        if not text.endswith("."):
            text += "."
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.ctx["DEVICE"]) for k, v in inputs.items()}
        if self.ctx["DEVICE"] == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        raw = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            box_threshold=conf,
            text_threshold=0.25,
            target_sizes=torch.tensor([image.size[::-1]], device=self.ctx["DEVICE"]),
        )[0]
        labels = raw.get("text_labels", raw.get("labels", []))
        dets = []
        for score, label, box in zip(raw["scores"], labels, raw["boxes"]):
            x1, y1, x2, y2 = box.detach().cpu().tolist()
            dets.append(self.ctx["Det"](str(label), float(score), x1, y1, x2, y2))
        return self.ctx["Result"](dets, (time.perf_counter() - started) * 1000)


def _classes(prompt):
    return [x.strip() for x in re.split(r"[,;\n]+", prompt) if x.strip()] or [prompt.strip()]


class YOLOWorldAdapter(Adapter):
    def prepare(self, progress):
        from ultralytics import YOLOWorld
        self.model = YOLOWorld(self.c["weight"]).to(self.ctx["DEVICE"])
        self._classes = None

    def predict(self, frame, conf, prompt=""):
        if not prompt.strip():
            raise ValueError("YOLO-World requires a text prompt")
        started = time.perf_counter()
        classes = _classes(prompt)
        if classes != self._classes:
            self.model.set_classes(classes)
            self._classes = classes
        pred = self.model.predict(
            frame,
            conf=conf,
            imgsz=640,
            device=0 if self.ctx["DEVICE"] == "cuda" else "cpu",
            verbose=False,
        )[0]
        dets = []
        if pred.boxes is not None:
            for box in pred.boxes:
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
                cid = int(box.cls[0])
                dets.append(self.ctx["Det"](str(pred.names[cid]), float(box.conf[0]), x1, y1, x2, y2))
        return self.ctx["Result"](dets, (time.perf_counter() - started) * 1000)


class YOLOEAdapter(YOLOWorldAdapter):
    def prepare(self, progress):
        from ultralytics import YOLOE
        self.model = YOLOE(self.c["weight"]).to(self.ctx["DEVICE"])
        self._classes = None


class GroundingDino15EdgeAdapter(Adapter):
    def prepare(self, progress):
        token = self.ctx["env_value"]("DDS_API_TOKEN")
        if not token:
            raise RuntimeError("DDS_API_TOKEN missing (.env.local)")
        from dds_cloudapi_sdk import Client, Config
        from dds_cloudapi_sdk.image_resizer import image_to_base64
        from dds_cloudapi_sdk.tasks.v2_task import V2Task
        self.client = Client(Config(token))
        self.V2Task = V2Task
        self.image_to_base64 = image_to_base64

    def predict(self, frame, conf, prompt=""):
        if not prompt.strip():
            raise ValueError("Grounding DINO 1.5 Edge requires a prompt")
        started = time.perf_counter()
        path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                path = Path(f.name)
            cv2.imwrite(str(path), frame)
            task = self.V2Task(
                api_path="/v2/task/grounding_dino/detection",
                api_body={
                    "model": "GroundingDino-1.5-Edge",
                    "image": self.image_to_base64(str(path)),
                    "prompt": {"type": "text", "text": prompt.strip()},
                    "targets": ["bbox"],
                    "bbox_threshold": float(conf),
                    "iou_threshold": 0.8,
                },
            )
            self.client.run_task(task)
            result = task.result or {}
            objects = result.get("objects", []) if isinstance(result, dict) else getattr(result, "objects", [])
            dets = []
            for obj in objects:
                box = obj.get("bbox") if isinstance(obj, dict) else getattr(obj, "bbox", None)
                if not box:
                    continue
                category = obj.get("category", prompt) if isinstance(obj, dict) else getattr(obj, "category", prompt)
                score = obj.get("score", float("nan")) if isinstance(obj, dict) else getattr(obj, "score", float("nan"))
                x1, y1, x2, y2 = map(float, box)
                dets.append(self.ctx["Det"](str(category), float(score), x1, y1, x2, y2))
            return self.ctx["Result"](dets, (time.perf_counter() - started) * 1000)
        finally:
            if path:
                path.unlink(missing_ok=True)


class LocateAnythingAdapter(Adapter):
    def prepare(self, progress):
        root = self.ctx["ROOT"]
        default = root / ".venv-locateanything" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        command = shlex.split(
            self.ctx["env_value"]("LOCATEANYTHING_PYTHON", str(default)),
            posix=os.name != "nt",
        )
        command += [
            str(root / "locateanything_worker.py"),
            "--model", self.c["model"],
            "--device", self.ctx["DEVICE"],
        ]
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self.proc.stdout.readline().strip() != "READY":
            raise RuntimeError("LocateAnything worker failed to start")

    def predict(self, frame, conf, prompt=""):
        if not prompt.strip():
            raise ValueError("LocateAnything-3B requires a prompt")
        started = time.perf_counter()
        path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                path = Path(f.name)
            cv2.imwrite(str(path), frame)
            request = {
                "image": str(path),
                "prompt": prompt.strip(),
                "mode": self.ctx["env_value"]("LOCATEANYTHING_MODE", "hybrid"),
                "max_new_tokens": int(self.ctx["env_value"]("LOCATEANYTHING_MAX_NEW_TOKENS", "2048")),
            }
            self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("LocateAnything worker stopped")
            payload = json.loads(line)
            if payload.get("error"):
                raise RuntimeError(payload["error"])
            dets = [
                self.ctx["Det"](
                    prompt.strip(),
                    float("nan"),
                    float(box["x1"]),
                    float(box["y1"]),
                    float(box["x2"]),
                    float(box["y2"]),
                )
                for box in payload.get("boxes", [])
            ]
            return self.ctx["Result"](dets, (time.perf_counter() - started) * 1000)
        finally:
            if path:
                path.unlink(missing_ok=True)

    def unload(self):
        if getattr(self, "proc", None):
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None
        super().unload()
