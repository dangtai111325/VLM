"""Benchmark the configured detection/LCOD backends on a live webcam.

Uses the same adapter cells as benchmark_object_detection.ipynb.
Open-vocabulary/grounding models receive --prompt; closed-set models ignore it.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "benchmark_object_detection.ipynb"
OUTPUTS = ROOT / "outputs" / "camera_benchmark"

# Runtime cells through preflight definitions; UI/launch cells are intentionally excluded.
CELL_INDEXES = (2, 4, 6, 8, 10, 12, 14, 16)


class ConsoleProgress:
    def note(self, text: str) -> None:
        print(text, flush=True)

    def task(self, _percent: int, text: str) -> None:
        self.note(text)


def load_notebook_runtime() -> dict:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    previous_cwd = Path.cwd()
    try:
        import os

        os.chdir(ROOT)
        for index in CELL_INDEXES:
            source = "".join(notebook["cells"][index]["source"])
            exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(previous_cwd)
    return namespace


def open_camera(cv2, index: int):
    backend = getattr(cv2, "CAP_DSHOW", None)
    capture = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    if not capture.isOpened() and backend is not None:
        capture.release()
        capture = cv2.VideoCapture(index)
    return capture


def synchronize(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], percent: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percent))


def main() -> None:
    runtime = load_notebook_runtime()
    cv2 = runtime["cv2"]
    torch = runtime["torch"]
    CONFIG = runtime["CONFIG"]
    MANAGER = runtime["MANAGER"]
    DEVICE = runtime["DEVICE"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--prompt", default="person", help="Prompt for LCOD/open-vocabulary models")
    parser.add_argument("--models", nargs="+", choices=tuple(CONFIG), help="Omit to benchmark all configured models")
    args = parser.parse_args()

    if args.warmup < 0 or args.frames < 1:
        raise SystemExit("--warmup must be >= 0 and --frames must be >= 1")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    capture = open_camera(cv2, args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open webcam {args.camera}")

    first_ok, first_frame = capture.read()
    if not first_ok:
        capture.release()
        raise RuntimeError(f"Webcam {args.camera} opened but did not provide a frame")

    height, width = first_frame.shape[:2]
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    started_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")

    environment = {
        "timestamp_utc": started_at.isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": DEVICE,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "camera_index": args.camera,
        "camera_resolution": f"{width}x{height}",
        "camera_reported_fps": reported_fps if reported_fps > 0 else None,
        "confidence": args.confidence,
        "prompt": args.prompt,
        "warmup_frames_per_model": args.warmup,
        "measured_frames_per_model": args.frames,
    }

    frame_rows = []
    models = {}

    try:
        selected_keys = args.models or list(CONFIG)
        for key in selected_keys:
            config = CONFIG[key]
            prompt = args.prompt if config.get("prompted") else ""
            print(f"\nBenchmarking {config['name']} ({args.frames} frames)...", flush=True)

            MANAGER.unload()
            synchronize(torch)
            load_started = time.perf_counter()
            MANAGER.load(key, ConsoleProgress())
            synchronize(torch)
            load_ms = (time.perf_counter() - load_started) * 1000

            for _ in range(args.warmup):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Webcam stream ended during warm-up")
                MANAGER.predict(key, frame, args.confidence, prompt)
            synchronize(torch)

            local_cuda_memory = config["type"] not in {"locateanything", "gdino15_edge"}
            if torch.cuda.is_available() and local_cuda_memory:
                torch.cuda.reset_peak_memory_stats()

            latencies = []
            adapter_latencies = []
            object_counts = []
            for frame_number in range(1, args.frames + 1):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Webcam stream ended during measurement")

                synchronize(torch)
                measured_started = time.perf_counter()
                result = MANAGER.predict(key, frame, args.confidence, prompt)
                synchronize(torch)
                end_to_end_ms = (time.perf_counter() - measured_started) * 1000

                latencies.append(end_to_end_ms)
                adapter_latencies.append(float(result.ms))
                object_counts.append(len(result.detections))
                frame_rows.append(
                    {
                        "model_key": key,
                        "model": config["name"],
                        "prompt": prompt,
                        "frame": frame_number,
                        "end_to_end_ms": round(end_to_end_ms, 3),
                        "adapter_ms": round(float(result.ms), 3),
                        "objects": len(result.detections),
                    }
                )

            peak_vram_mib = None
            if torch.cuda.is_available() and local_cuda_memory:
                peak_vram_mib = round(torch.cuda.max_memory_allocated() / 1024**2, 2)

            mean_latency = float(np.mean(latencies))
            models[key] = {
                "name": config["name"],
                "prompted": bool(config.get("prompted")),
                "remote": bool(config.get("remote")),
                "prompt": prompt,
                "load_ms": round(load_ms, 3),
                "end_to_end_mean_ms": round(mean_latency, 3),
                "end_to_end_median_ms": round(float(np.median(latencies)), 3),
                "end_to_end_p95_ms": round(percentile(latencies, 95), 3),
                "adapter_mean_ms": round(float(np.mean(adapter_latencies)), 3),
                "throughput_fps": round(1000 / mean_latency, 3),
                "mean_detected_objects_per_frame": round(float(np.mean(object_counts)), 3),
                "peak_allocated_vram_mib_in_main_process": peak_vram_mib,
            }
            print(json.dumps(models[key], ensure_ascii=False), flush=True)
    finally:
        capture.release()
        MANAGER.unload()
        synchronize(torch)

    if frame_rows:
        csv_path = OUTPUTS / f"camera_frames_{stamp}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=frame_rows[0].keys())
            writer.writeheader()
            writer.writerows(frame_rows)
        print(f"\nRaw frames: {csv_path}")

    summary_path = OUTPUTS / f"camera_summary_{stamp}.json"
    summary_path.write_text(
        json.dumps({"environment": environment, "models": models}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
