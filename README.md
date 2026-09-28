# RT-LCOD — Real-Time Language-Conditioned Object Detection

This branch is dedicated to the RT-LCOD research project for robot perception.

## Architecture

**YOLOE-26M open-vocabulary detector + lightweight attribute/relation grounding head + explicit NO_TARGET rejection + Grounding DINO teacher distillation.**

Runtime is intentionally lightweight: heavy teacher models are used offline during training only.

## Scope V1

Each prompt contains:

- one target class;
- zero or one attribute;
- zero or one first-order relation;
- zero or one reference object class;
- optional no-target outcome when the full condition is not satisfied.

Examples:

- `cup`
- `red cup`
- `red cup next to pillow`
- `bottle left of laptop`

Second-order relation chains are out of scope for V1.

## Repository layout

- `plan.md` — full research and implementation plan
- `rt_lcod/` — complete training/inference project
- `rt_lcod/notebooks/rt_lcod_end_to_end.ipynb` — staged research notebook
- `.github/workflows/rt_lcod_ci.yml` — CI smoke validation

## Quick start

```bash
cd rt_lcod
python -m pip install -e .[dev]
python scripts/smoke_test.py
pytest -q
```

Then open:

`rt_lcod/notebooks/rt_lcod_end_to_end.ipynb`

Keep `RUN_HEAVY=False` first. Enable heavy stages only after the synthetic smoke path passes.

## Validation status

The branch includes unit tests, synthetic forward/backward validation, checkpoint/resume tests, timing instrumentation, and CI. Full YOLOE/Grounding-DINO CUDA validation must be run on the target GPU environment.
