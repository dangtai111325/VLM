# Video benchmark results

## Scope and method

This run measures the same adapter pipeline used by the interactive notebook:
frame decoding from disk, preprocessing, model inference, and postprocessing.
It does **not** measure detection accuracy because the source video has no
ground-truth annotations.

- Run time: 2026-09-27 09:59 UTC
- Input: `data/videos/highway_traffic_gpl.mp4` (1280x720, 30 FPS, 812 frames)
- Protocol: 10 warm-up frames, then 60 measured frames per model
- Confidence threshold: 0.35
- Prompted-model query: `person, car, bicycle, truck`
- Runtime: Python 3.13.14, PyTorch 2.11.0+cu128, CUDA 12.8
- OS: Windows 11, build 26200
- CPU: 12th Gen Intel(R) Core(TM) i7-12850HX; observed clock 2.419 GHz
- Host memory available at inspection: 10,085 MiB (installed total was not
  exposed by the current Windows permission context)
- GPU: NVIDIA RTX A3000 12GB Laptop GPU; 12,288 MiB VRAM; NVIDIA driver
  596.71; CUDA runtime 12.8

## Input configuration

The numerical table below is a **file-video** benchmark, not a webcam test:
the source is 1280x720 at 30 FPS, so its realtime deadline is 33.333 ms/frame.

For the separately captured webcam benchmark on this machine, OpenCV used
camera index `0` with the DirectShow fallback and negotiated 640x480. The
camera driver did not report a reliable FPS value to OpenCV, therefore a camera
source-frame deadline cannot be calculated from that historical run. Its raw
configuration/result artifact remains at
`outputs/camera_benchmark/camera_summary_20260926T200414Z.json`.

The benchmark process ran models one at a time. Existing notebook kernels were
still holding VRAM, so the table describes the realistic shared-GPU state of
this workstation; rerun with all notebook kernels stopped for an isolated
maximum-throughput result.

## Realtime performance

`Pipeline` includes video decode plus the adapter call. `Realtime factor` is
`processing FPS / 30`; a value at least 1.0 can keep up on average. `Miss %`
is the percentage of frames whose complete pipeline exceeded the 33.333 ms
source-frame deadline. `VRAM` is the peak allocation made by the benchmark
process itself, not total device memory across other active kernels.

| Model | Load (ms) | Pipeline mean / p95 / p99 (ms) | Processing FPS | Realtime factor | Miss % | Mean objects/frame | VRAM (MiB) | Realtime assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO26l | 321.1 | 17.946 / 19.816 / 21.042 | 55.723 | 1.857 | 0.00 | 4.317 | 298.39 | Yes — stable at 30 FPS |
| YOLO26x | 270.3 | 28.170 / 30.262 / 30.600 | 35.499 | 1.183 | 0.00 | 6.150 | 659.01 | Yes — stable at 30 FPS |
| RF-DETR-L | 2925.4 | 27.462 / 33.098 / 40.036 | 36.415 | 1.214 | 5.00 | 11.317 | 152.55 | Mostly realtime; occasional tail-latency misses |
| RF-DETR-XL | 3139.0 | 34.470 / 39.561 / 43.069 | 29.011 | 0.967 | 68.33 | 11.033 | 407.19 | No — below the 30 FPS input rate |
| Grounding DINO | 7784.1 | 328.917 / 338.417 / 343.689 | 3.040 | 0.101 | 100.00 | 7.367 | 1136.65 | No — use offline/low-rate processing |
| YOLO-World | 183.4 | 24.644 / 29.710 / 31.475 | 40.577 | 1.353 | 0.00 | 2.617 | 1378.81 | Yes — stable at 30 FPS |
| YOLOE | 209.4 | 29.722 / 39.254 / 42.603 | 33.645 | 1.122 | 11.67 | 4.750 | 413.45 | Borderline — average keeps up, p95 does not |

The fastest measured end-to-end path is YOLO26l. For 30-FPS open-vocabulary
detection in this configuration, YOLO-World is the stable option. RF-DETR-L
has strong object density in this run but needs a small frame queue/drop policy
to avoid occasional latency buildup. Grounding DINO is useful for grounding
quality, but it is not suitable for a 30-FPS live display on this hardware.

## Not measured in this run

- **Grounding DINO 1.5 Edge:** requires a `DDS_API_TOKEN`; no token is
  configured, so no cloud/API measurement was made.
- **LocateAnything-3B:** its worker was already active in the interactive
  notebook and holding GPU memory. It was deliberately excluded to avoid a
  duplicate worker or an out-of-memory result. Its preflight smoke test passed;
  measure it in a separate, idle-GPU run.

## Reproducibility artifacts

- Per-frame measurements:
  `outputs/video_benchmark/video_frames_20260927T095946Z.csv`
- Full environment and summary:
  `outputs/video_benchmark/video_summary_20260927T095946Z.json`
- Runner:
  `run_video_benchmark.py`

To reproduce this protocol after closing notebook kernels:

```powershell
python run_video_benchmark.py --warmup 10 --frames 60
```
