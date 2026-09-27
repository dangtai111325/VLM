from __future__ import annotations

import json
import logging
import os
import queue
import re
import shlex
import subprocess
import threading
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


class RFDETRExpectedBackboneWarningFilter(logging.Filter):
    """Hide two expected backbone-layout notices from RF-DETR startup."""

    prefixes = (
        "Using a different number of positional encodings than DINOv2",
        "Using patch size ",
    )

    def filter(self, record):
        return not str(record.getMessage()).startswith(self.prefixes)


class Adapter:
    def __init__(self, key, config, ctx):
        self.key, self.c, self.ctx, self.model = key, config[key], ctx, None

    def unload(self):
        self.model = None
        if self.ctx["DEVICE"] == "cuda":
            torch.cuda.empty_cache()

    def predict(self, frame, conf, prompt=""):
        raise NotImplementedError

    def temp_dir(self):
        directory = self.ctx["ROOT"] / "cache" / "tmp"
        directory.mkdir(parents=True, exist_ok=True)
        return directory


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

        # RF-DETR's published Large/XL checkpoints intentionally use a
        # non-DINOv2 patch layout. These two messages are expected after the
        # full RF-DETR checkpoint is loaded, not a degraded-load condition.
        rfdetr_log = logging.getLogger("rf-detr")
        if not any(
            isinstance(item, RFDETRExpectedBackboneWarningFilter)
            for item in rfdetr_log.filters
        ):
            rfdetr_log.addFilter(RFDETRExpectedBackboneWarningFilter())

        progress.note(f"Loading {self.c['name']}")
        if self.c["variant"] == "large":
            self.model = RFDETRLarge(num_classes=90)
        else:
            from rfdetr_plus import RFDETRXLarge

            self.model = RFDETRXLarge(
                accept_platform_model_license=True,
                num_classes=90,
            )

        # RF-DETR emits a latency warning until inference() configures the
        # deployment model. Keep a single model in memory and use FP16 only
        # when CUDA is available.
        self.model.inference(
            compile=False,
            dtype=torch.float16 if self.ctx["DEVICE"] == "cuda" else torch.float32,
            inplace=True,
        )
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
            threshold=conf,
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
    def model_directory(self):
        directory = self.ctx["ROOT"] / "cache" / "models"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def prepare(self, progress):
        from ultralytics import YOLOWorld
        self.model = self.load_ultralytics_model(YOLOWorld).to(self.ctx["DEVICE"])
        self._classes = None

    def load_ultralytics_model(self, model_class):
        """Download prompted YOLO weights only into testmodel/cache/models."""
        weight = Path(self.c["weight"])
        if not weight.is_absolute():
            weight = self.model_directory() / weight.name
        weight.parent.mkdir(parents=True, exist_ok=True)
        previous = Path.cwd()
        try:
            os.chdir(weight.parent)
            return model_class(weight.name)
        finally:
            os.chdir(previous)

    def predict(self, frame, conf, prompt=""):
        if not prompt.strip():
            raise ValueError("YOLO-World requires a text prompt")
        started = time.perf_counter()
        classes = _classes(prompt)
        previous = Path.cwd()
        try:
            # YOLOE lazily fetches MobileCLIP during its first prediction.
            # Keep that secondary asset with the main checkpoint.
            os.chdir(self.model_directory())
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
        finally:
            os.chdir(previous)
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
        self.model = self.load_ultralytics_model(YOLOE).to(self.ctx["DEVICE"])
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
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".png",
                dir=self.temp_dir(),
            ) as f:
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
    def _acquire_worker_lock(self, lock_path):
        """Allow only one LocateAnything worker across notebook kernels."""
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                descriptor = os.open(
                    str(lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                try:
                    owner_pid = int(lock_path.read_text(encoding="utf-8").strip())
                    if owner_pid < 1:
                        raise ValueError("invalid PID")
                    os.kill(owner_pid, 0)
                except (OSError, ValueError):
                    # The previous kernel stopped without releasing its lock.
                    lock_path.unlink(missing_ok=True)
                    continue
                raise RuntimeError(
                    "LocateAnything is already loading or active in another notebook "
                    "kernel. Stop it there before starting a second session."
                )
            else:
                os.write(descriptor, str(os.getpid()).encode("ascii"))
                self._worker_lock_descriptor = descriptor
                self._worker_lock_path = lock_path
                return

    def _release_worker_lock(self):
        descriptor = getattr(self, "_worker_lock_descriptor", None)
        if descriptor is not None:
            os.close(descriptor)
            self._worker_lock_descriptor = None
        lock_path = getattr(self, "_worker_lock_path", None)
        if lock_path is not None:
            lock_path.unlink(missing_ok=True)
            self._worker_lock_path = None

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
        worker_root = root / "cache" / "locateanything"
        worker_hf = worker_root / "huggingface"
        worker_tmp = worker_root / "tmp"
        worker_log = root / "logs" / "locateanything_worker.log"
        for directory in (worker_hf, worker_tmp, worker_log.parent):
            directory.mkdir(parents=True, exist_ok=True)
        self._acquire_worker_lock(worker_root / "worker.lock")
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HOME": str(worker_hf),
                "HUGGINGFACE_HUB_CACHE": str(worker_hf / "hub"),
                "TRANSFORMERS_CACHE": str(worker_hf / "hub"),
                "XDG_CACHE_HOME": str(worker_root / "xdg"),
                "TEMP": str(worker_tmp),
                "TMP": str(worker_tmp),
                "TMPDIR": str(worker_tmp),
                "PYTHONUNBUFFERED": "1",
            }
        )
        try:
            self._log_handle = worker_log.open("a", encoding="utf-8")
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log_handle,
                text=True,
                bufsize=1,
                cwd=root,
                env=environment,
            )
        except Exception:
            self.unload()
            raise
        timeout_s = float(self.ctx["env_value"]("LOCATEANYTHING_STARTUP_TIMEOUT_S", "180"))
        ready_lines = queue.Queue(maxsize=1)
        threading.Thread(
            target=lambda: ready_lines.put(self.proc.stdout.readline()),
            daemon=True,
        ).start()
        started_at = time.monotonic()
        while True:
            elapsed_s = time.monotonic() - started_at
            remaining_s = timeout_s - elapsed_s
            if remaining_s <= 0:
                self.unload()
                raise TimeoutError(
                    f"LocateAnything did not become ready within {timeout_s:.0f}s; "
                    "see logs/locateanything_worker.log"
                )
            try:
                ready = ready_lines.get(timeout=min(1.0, remaining_s)).strip()
                break
            except queue.Empty:
                progress.note(
                    "Starting LocateAnything worker "
                    f"({elapsed_s:.0f}s / {timeout_s:.0f}s timeout)"
                )
        if ready != "READY":
            self.unload()
            raise RuntimeError(
                "LocateAnything worker failed to start; see logs/locateanything_worker.log"
            )

    def predict(self, frame, conf, prompt=""):
        if not prompt.strip():
            raise ValueError("LocateAnything-3B requires a prompt")
        started = time.perf_counter()
        path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg",
                dir=self.temp_dir(),
            ) as f:
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
                self.proc.wait(timeout=10)
            except Exception:
                pass
            self.proc = None
        if getattr(self, "_log_handle", None):
            self._log_handle.close()
            self._log_handle = None
        self._release_worker_lock()
        super().unload()
