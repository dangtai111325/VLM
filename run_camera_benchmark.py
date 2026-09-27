"""Measure the notebook's four detection backends on a live local webcam.

The runner executes the adapter cells from benchmark_object_detection.ipynb so
the exact preprocessing and inference path is the same one used by the UI.
It stores per-frame measurements and a machine-readable summary below outputs/.
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
CELL_INDEXES = (2, 4, 6, 8, 13, 15, 17)


class ConsoleProgress:
    """Minimal progress interface expected by the adapters."""

    def note(self, text: str) -> None:
        print(text, flush=True)

    def task(self, _percent: int, text: str) -> None:
        self.note(text)


def load_notebook_runtime() -> dict:
    """Execute the adapter/runtime cells without starting the widget UI."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    # Dataclasses resolve postponed annotations through sys.modules[__name__].
    # Use the real entry module while evaluating notebook cells outside Jupyter.
    namespace = {"__name__": "__main__"}
    previous_cwd = Path.cwd()
    try:
        # The notebook resolves ROOT from the working directory.
        import os

        os.chdir(ROOT)
        for index in CELL_INDEXES:
            source = "".join(notebook["cells"][index]["source"])
            exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(previous_cwd)
    return namespace


def open_camera(cv2, index: int):
    """Prefer the Windows DirectShow backend used by the UI, then fall back."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--warmup", type=int, default=10, help="Unmeasured inference frames per model")
    parser.add_argument("--frames", type=int, default=100, help="Measured inference frames per model")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("yolo26m", "rfdetr_m", "dfine_m", "lwdetr_m"),
        help="Optional model keys; omit to benchmark all four",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.frames < 1:
        raise SystemExit("--warmup must be >= 0 and --frames must be >= 1")

    runtime = load_notebook_runtime()
    cv2 = runtime["cv2"]
    torch = runtime["torch"]
    CONFIG = runtime["CONFIG"]
    MANAGER = runtime["MANAGER"]
    DEVICE = runtime["DEVICE"]

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
    frame_rows: list[dict] = []
    models: dict[str, dict] = {}

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
        "warmup_frames_per_model": args.warmup,
        "measured_frames_per_model": args.frames,
    }

    try:
        selected_keys = args.models or list(CONFIG)
        for key in selected_keys:
            config = CONFIG[key]
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
                MANAGER.predict(key, frame, args.confidence)
            synchronize(torch)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            latencies: list[float] = []
            adapter_latencies: list[float] = []
            object_counts: list[int] = []
            for frame_number in range(1, args.frames + 1):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Webcam stream ended during measurement")

                synchronize(torch)
                measured_started = time.perf_counter()
                result = MANAGER.predict(key, frame, args.confidence)
                synchronize(torch)
                end_to_end_ms = (time.perf_counter() - measured_started) * 1000

                latencies.append(end_to_end_ms)
                adapter_latencies.append(float(result.ms))
                object_counts.append(len(result.detections))
                frame_rows.append(
                    {
                        "model_key": key,
                        "model": config["name"],
                        "frame": frame_number,
                        "end_to_end_ms": round(end_to_end_ms, 3),
                        "adapter_ms": round(float(result.ms), 3),
                        "objects": len(result.detections),
                    }
                )

            peak_vram_mib = (
                round(torch.cuda.max_memory_allocated() / 1024**2, 2)
                if torch.cuda.is_available()
                else None
            )
            mean_latency = float(np.mean(latencies))
            models[key] = {
                "name": config["name"],
                "load_ms": round(load_ms, 3),
                "end_to_end_mean_ms": round(mean_latency, 3),
                "end_to_end_median_ms": round(float(np.median(latencies)), 3),
                "end_to_end_p95_ms": round(percentile(latencies, 95), 3),
                "adapter_mean_ms": round(float(np.mean(adapter_latencies)), 3),
                "throughput_fps": round(1000 / mean_latency, 3),
                "mean_detected_objects_per_frame": round(float(np.mean(object_counts)), 3),
                "peak_allocated_vram_mib_including_model": peak_vram_mib,
            }
            print(json.dumps(models[key], ensure_ascii=False), flush=True)
    finally:
        capture.release()
        MANAGER.unload()
        synchronize(torch)

    csv_path = OUTPUTS / f"camera_frames_{stamp}.csv"
    summary_path = OUTPUTS / f"camera_summary_{stamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=frame_rows[0].keys())
        writer.writeheader()
        writer.writerows(frame_rows)
    summary_path.write_text(
        json.dumps({"environment": environment, "models": models}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nRaw frames: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
