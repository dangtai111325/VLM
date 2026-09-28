# RT-LCOD

Main architecture:

**YOLOE-26M + lightweight region encoder + attribute/relation grounding student + NO_TARGET + Grounding DINO distillation.**

The final video runtime does not contain Grounding DINO.

## Recommended entry point

Use `notebooks/rt_lcod_end_to_end.ipynb` on an NVIDIA T4 and choose **Run All**.

The notebook automatically:

1. checks CUDA/T4 and prints environment/VRAM;
2. downloads a bounded real gRefCOCO subset and required COCO images;
3. builds train/validation/held-out test manifests;
4. runs YOLOE-26M once per sample and caches candidate/region/text features;
5. trains Stage 1 and saves every checkpoint/metric/timing file;
6. evaluates Stage 1 validation;
7. caches Grounding DINO teacher outputs offline;
8. trains Stage 2 KD starting from Stage 1 weights;
9. selects the final checkpoint using validation only;
10. runs held-out test;
11. loads the final video runtime;
12. displays a video path box, prompt box and Play/Pause/Stop controls.

Default T4 profile is `configs/t4_runall.yaml`.

## Why only gRefCOCO in the default Run All?

The broader research plan includes RefCOCO+/RefCOCOg, Visual Genome, OmniLabel and robot-domain data. The first executable baseline intentionally uses one real dataset that already provides referring expressions and true no-target examples. Additional datasets should be introduced only after the first T4 logs identify the actual bottleneck.

## Generated files

Real data, images, candidate caches, teacher caches, runs and checkpoints are local/gitignored. Important summaries are written under `data/*_report` / cache stats, `runs/<run>/`, and `artifacts/`.

## Development tests

```bash
python -m pip install -e .[all]
python scripts/smoke_test.py
pytest -q
```

These validate the software path. They do not substitute for the full T4 experiment.
