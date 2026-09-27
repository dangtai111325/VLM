# Benchmark status — updated LCOD model matrix

The project model matrix has been replaced with:

**Closed-set:** YOLO26l, YOLO26x, RF-DETR-L, RF-DETR-XL  
**LCOD/open-vocabulary/grounding:** Grounding DINO, YOLO-World, YOLOE, Grounding DINO 1.5 Edge, LocateAnything-3B

The previous webcam report corresponded to the superseded matrix (YOLO26m, RF-DETR-M, D-FINE-M, LW-DETR-M), so its numerical table is no longer presented here as the current project benchmark.

Historical raw files remain under `outputs/camera_benchmark/` for provenance.

## Re-run protocol

```powershell
python run_camera_benchmark.py --camera 0 --warmup 10 --frames 100 --confidence 0.35 --prompt "person"
```

For prompted models, the report records the exact prompt. Closed-set models ignore the prompt.

## Important comparability note

- Local models: end-to-end latency measures adapter preprocessing + inference + postprocessing.
- LocateAnything-3B runs in a dedicated subprocess/environment, so main-process CUDA peak memory does not represent the worker's total GPU footprint.
- Grounding DINO 1.5 Edge is an official remote API model; latency includes image upload, network and service time. Do not compare that value as pure local GPU latency.

A new numeric summary should be committed only after the updated model matrix passes preflight and the benchmark is rerun on the target machine.
