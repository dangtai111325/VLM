"""Measure video-pipeline latency and real-time capability for ready models.

The runner reuses the notebook adapters so its results match the interactive UI.
It writes raw per-frame data and a reproducible JSON summary under
``outputs/video_benchmark``. Grounding DINO 1.5 Edge is excluded unless its
API token is configured and the model is explicitly selected.
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
OUTPUTS = ROOT / "outputs" / "video_benchmark"
RUNTIME_CELLS = (2, 4, 6, 8, 10, 12, 14, 16)


class ConsoleProgress:
    def note(self, text: str) -> None:
        print(text, flush=True)

    def update(self, _completed: int, _total: int, text: str) -> None:
        self.note(text)


def load_notebook_runtime() -> dict:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    previous_cwd = Path.cwd()
    try:
        import os

        os.chdir(ROOT)
        for index in RUNTIME_CELLS:
            source = "".join(notebook["cells"][index]["source"])
            exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(previous_cwd)
    return namespace


def synchronize(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], percent: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percent))


def prompt_for_model(key: str, prompt: str) -> str:
    """Keep one CLI prompt useful for both list- and sentence-based models."""
    if key == "grounding_dino":
        labels = [item.strip() for item in prompt.replace(";", ",").split(",") if item.strip()]
        return ". ".join(labels) + "."
    return prompt


def ready_keys(cache: Path, config: dict) -> list[str]:
    report_path = cache / "model_preflight.json"
    if not report_path.exists():
        return [key for key, model in config.items() if not model.get("remote")]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    states = report.get("models", {})
    return [key for key in config if states.get(key, {}).get("state") == "ready"]


def main() -> None:
    runtime = load_notebook_runtime()
    cv2 = runtime["cv2"]
    torch = runtime["torch"]
    config = runtime["CONFIG"]
    manager = runtime["MANAGER"]
    cache = runtime["CACHE"]
    device = runtime["DEVICE"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "data" / "videos" / "highway_traffic_gpl.mp4",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--prompt", default="person, car, bicycle, truck")
    parser.add_argument("--models", nargs="+", choices=tuple(config))
    args = parser.parse_args()

    if args.warmup < 0 or args.frames < 1:
        raise SystemExit("--warmup must be >= 0 and --frames must be >= 1")
    video = args.video.resolve()
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")

    probe = cv2.VideoCapture(str(video))
    if not probe.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    source_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    probe.release()
    if source_fps <= 0:
        source_fps = 30.0

    started_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    selected_keys = args.models or ready_keys(cache, config)
    if not selected_keys:
        raise SystemExit("No ready models. Run the notebook preflight first.")

    environment = {
        "timestamp_utc": started_at.isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "video": str(video.relative_to(ROOT)),
        "source_resolution": f"{width}x{height}",
        "source_fps": source_fps,
        "source_frame_count": source_frames,
        "source_frame_period_ms": round(1000 / source_fps, 3),
        "warmup_frames_per_model": args.warmup,
        "measured_frames_per_model": args.frames,
        "confidence": args.confidence,
        "prompt": args.prompt,
    }
    rows: list[dict] = []
    models: dict[str, dict] = {}

    try:
        for key in selected_keys:
            model = config[key]
            if model.get("remote") and not runtime["env_value"]("DDS_API_TOKEN"):
                models[key] = {
                    "name": model["name"],
                    "state": "not_configured",
                    "detail": "DDS_API_TOKEN is required for this remote API model",
                }
                continue

            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot reopen video: {video}")
            model_prompt = prompt_for_model(key, args.prompt) if model.get("prompted") else ""
            print(f"\nBenchmarking {model['name']} ({args.frames} measured frames)...", flush=True)
            try:
                manager.unload()
                synchronize(torch)
                load_started = time.perf_counter()
                manager.load(key, ConsoleProgress())
                synchronize(torch)
                load_ms = (time.perf_counter() - load_started) * 1000

                for _ in range(args.warmup):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError("Video ended during warm-up")
                    manager.predict(key, frame, args.confidence, model_prompt)
                synchronize(torch)

                local_cuda_model = model["type"] not in {"locateanything", "gdino15_edge"}
                if torch.cuda.is_available() and local_cuda_model:
                    torch.cuda.reset_peak_memory_stats()

                cycle_ms: list[float] = []
                decode_ms: list[float] = []
                adapter_ms: list[float] = []
                object_counts: list[int] = []
                for frame_number in range(1, args.frames + 1):
                    cycle_started = time.perf_counter()
                    ok, frame = capture.read()
                    decoded_at = time.perf_counter()
                    if not ok:
                        raise RuntimeError("Video ended during measurement")
                    synchronize(torch)
                    result = manager.predict(key, frame, args.confidence, model_prompt)
                    synchronize(torch)
                    ended_at = time.perf_counter()

                    total_ms = (ended_at - cycle_started) * 1000
                    decode_time_ms = (decoded_at - cycle_started) * 1000
                    cycle_ms.append(total_ms)
                    decode_ms.append(decode_time_ms)
                    adapter_ms.append(float(result.ms))
                    object_counts.append(len(result.detections))
                    rows.append(
                        {
                            "model_key": key,
                            "model": model["name"],
                            "prompt": model_prompt,
                            "frame": frame_number,
                            "cycle_ms": round(total_ms, 3),
                            "decode_ms": round(decode_time_ms, 3),
                            "adapter_ms": round(float(result.ms), 3),
                            "objects": len(result.detections),
                            "meets_source_deadline": total_ms <= 1000 / source_fps,
                        }
                    )

                mean_cycle_ms = float(np.mean(cycle_ms))
                frame_period_ms = 1000 / source_fps
                deadline_miss_rate = sum(value > frame_period_ms for value in cycle_ms) / len(cycle_ms)
                peak_vram_mib = None
                if torch.cuda.is_available() and local_cuda_model:
                    peak_vram_mib = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
                models[key] = {
                    "name": model["name"],
                    "state": "complete",
                    "prompt": model_prompt,
                    "load_ms": round(load_ms, 3),
                    "pipeline_mean_ms": round(mean_cycle_ms, 3),
                    "pipeline_median_ms": round(float(np.median(cycle_ms)), 3),
                    "pipeline_p95_ms": round(percentile(cycle_ms, 95), 3),
                    "pipeline_p99_ms": round(percentile(cycle_ms, 99), 3),
                    "pipeline_min_ms": round(float(np.min(cycle_ms)), 3),
                    "pipeline_max_ms": round(float(np.max(cycle_ms)), 3),
                    "decode_mean_ms": round(float(np.mean(decode_ms)), 3),
                    "adapter_mean_ms": round(float(np.mean(adapter_ms)), 3),
                    "processing_fps": round(1000 / mean_cycle_ms, 3),
                    "realtime_factor": round((1000 / mean_cycle_ms) / source_fps, 3),
                    "source_deadline_miss_rate": round(deadline_miss_rate, 4),
                    "source_deadline_miss_percent": round(deadline_miss_rate * 100, 2),
                    "mean_detected_objects_per_frame": round(float(np.mean(object_counts)), 3),
                    "peak_allocated_vram_mib_in_main_process": peak_vram_mib,
                }
                print(json.dumps(models[key], ensure_ascii=False), flush=True)
            except Exception as exc:
                models[key] = {
                    "name": model["name"],
                    "state": "failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
                print(json.dumps(models[key], ensure_ascii=False), flush=True)
            finally:
                capture.release()
                manager.unload()
                synchronize(torch)
    finally:
        manager.unload()
        synchronize(torch)

    csv_path = OUTPUTS / f"video_frames_{stamp}.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    summary_path = OUTPUTS / f"video_summary_{stamp}.json"
    summary_path.write_text(
        json.dumps({"environment": environment, "models": models}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Raw frames: {csv_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
