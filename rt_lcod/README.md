# RT-LCOD

This folder contains the trainable student architecture defined in `../plan.md`:

**YOLOE-26M + lightweight region encoder + attribute/relation grounding head + explicit no-target + Grounding DINO distillation.**

The heavy detector/teacher integrations are optional and lazy-loaded. The core package, tests, synthetic training, checkpoint/resume, timing, metrics, and temporal tracker can run without downloading model checkpoints.

## Quick start

```bash
cd rt_lcod
python -m venv .venv
# activate the environment
python -m pip install -U pip
python -m pip install -e .[dev]
python scripts/smoke_test.py
pytest -q
```

For full YOLOE/Grounding DINO runtime support:

```bash
python -m pip install -e .[all]
```

Install the CUDA-compatible PyTorch build for the target machine as appropriate.

## Heavy vs safe mode

`notebooks/rt_lcod_end_to_end.ipynb` defaults to `RUN_HEAVY = False`. In this mode, **Run All** validates the entire local research scaffold without downloading YOLOE or Grounding DINO.

Set `RUN_HEAVY = True` only after the environment is ready and dataset manifests exist.

## Main commands

```bash
python scripts/validate_manifest.py --manifest data/manifests/train.jsonl
python scripts/train.py --config configs/smoke.yaml --synthetic --run-name smoke_train
python scripts/evaluate.py --config configs/smoke.yaml --synthetic --checkpoint runs/<run>/checkpoints/best.pt
```

Full-data commands and stage gates are documented in `../plan.md`.
