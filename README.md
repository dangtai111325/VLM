# RT-LCOD — Real-Time Language-Conditioned Object Detection

This branch implements one concrete V1 architecture:

**YOLOE-26M open-vocabulary detector + lightweight attribute/relation grounding head + explicit NO_TARGET + Grounding DINO teacher distillation.**

Grounding DINO is training-only. Deployment keeps YOLOE plus the small student head.

## V1 prompt contract

A prompt contains one target class, optionally one simple attribute, and optionally one first-order relation plus one reference class.

Examples:

- `cup`
- `red cup`
- `fire extinguisher`
- `red car next to bus`
- `bottle left of laptop`

Second-order relation chains are intentionally out of scope for V1.

## Local RTX Run All notebook

Open:

`rt_lcod/notebooks/rt_lcod_end_to_end.ipynb`

On the local RTX A3000, **Run All executes the real pipeline** rather than the old synthetic-only notebook:

```text
gRefCOCO subset
  -> train / val / held-out test manifests
  -> YOLOE-26M proposal cache
  -> frozen region/text feature cache
  -> Stage 1 supervised grounding training
  -> Stage 1 validation
  -> Grounding DINO offline teacher cache
  -> Stage 2 knowledge distillation
  -> choose Stage 1 vs Stage 2 on validation
  -> held-out final test
  -> load final checkpoint
  -> interactive video player + prompt textbox
```

Default local baseline sizes are deliberately bounded: 1,000 train, 150 validation, 150 test, and at most 300 teacher samples. Change them only after collecting the first full set of logs.

## Logs kept for optimization

The notebook and scripts print/store environment, GPU/VRAM, data statistics, proposal recall, losses, target/no-target accuracy, checkpoints, elapsed times, cache statistics and video latency. Video inference reports detector/region/head latency plus mean/p50/p95/p99 and processing FPS.

Generated datasets, caches, checkpoints and model files stay local and are gitignored.

## Repository layout

- `plan.md` — broader research plan
- `rt_lcod/configs/local_a3000.yaml` — RTX A3000 baseline profile
- `rt_lcod/scripts/prepare_grefcoco.py` — real dataset preparation
- `rt_lcod/src/rt_lcod/runall.py` — one-click orchestrator
- `rt_lcod/src/rt_lcod/inference/video_ui.py` — video/prompt UI
- `rt_lcod/notebooks/rt_lcod_end_to_end.ipynb` — main entry point
- `.github/workflows/rt_lcod_ci.yml` — CPU smoke/unit CI

## Local development smoke test

```bash
cd rt_lcod
python -m pip install -e .[all]
python scripts/smoke_test.py
pytest -q
```

CPU CI validates software contracts. YOLOE/Grounding-DINO training speed and final accuracy must be measured on the local RTX A3000 run; the notebook is instrumented specifically to capture those values.
