from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from rt_lcod.config import load_config
from rt_lcod.inference.runtime import RTLCODRuntime


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default="artifacts/runtime_benchmark.json")
    args = parser.parse_args()

    import cv2

    config = load_config(args.config)
    runtime = RTLCODRuntime(config, args.checkpoint)
    prompt_started = time.perf_counter()
    parsed = runtime.set_prompt(args.prompt)
    prompt_ms = (time.perf_counter() - prompt_started) * 1000

    capture = cv2.VideoCapture(str(Path(args.video)))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    deadline_ms = 1000.0 / source_fps
    latencies = []
    component = {}
    no_target_count = 0
    measured = 0
    try:
        for frame_index in range(args.warmup + args.frames):
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            result = runtime.process_frame(frame)
            elapsed = (time.perf_counter() - started) * 1000
            if frame_index < args.warmup:
                continue
            latencies.append(elapsed)
            measured += 1
            no_target_count += int(result.box is None)
            for key, value in result.timings_ms.items():
                component.setdefault(key, []).append(float(value))
    finally:
        capture.release()

    if not latencies:
        raise SystemExit("no measured frames")
    summary = {
        "prompt": args.prompt,
        "parsed_prompt": parsed.__dict__,
        "prompt_change_ms": prompt_ms,
        "source_fps": source_fps,
        "source_deadline_ms": deadline_ms,
        "frames": measured,
        "pipeline_mean_ms": float(np.mean(latencies)),
        "pipeline_p50_ms": percentile(latencies, 50),
        "pipeline_p95_ms": percentile(latencies, 95),
        "pipeline_p99_ms": percentile(latencies, 99),
        "processing_fps": 1000.0 / float(np.mean(latencies)),
        "deadline_miss_percent": 100.0 * float(np.mean(np.asarray(latencies) > deadline_ms)),
        "no_target_percent": 100.0 * no_target_count / measured,
        "components_mean_ms": {k: float(np.mean(v)) for k, v in component.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
