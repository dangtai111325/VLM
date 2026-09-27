# Validation status

Validation performed before pushing this branch scaffold:

- `python -m compileall -q src scripts tests` — PASS
- `pytest -q` — **9 passed**
- `python scripts/smoke_test.py` — **PASS**
- notebook code cells executed sequentially with `RUN_HEAVY=False` — **PASS**
- editable install tested with `--no-build-isolation --no-deps` in the offline build environment — PASS

The smoke path validates:

- controlled prompt parser;
- pairwise geometry features;
- variable-size candidate padding;
- grounding-head forward/backward;
- target/no-target/KD losses;
- metric tracking;
- checkpoint save/load;
- run timing/metadata logging;
- simple temporal target tracker;
- notebook structure and smoke-mode execution.

## Heavy path not executed in this build environment

This environment had no external package/model network access, so the following cannot be truthfully claimed as runtime-tested here:

- downloading/running `yoloe-26m.pt`;
- downloading/running Grounding DINO;
- generating real candidate caches from external datasets;
- TensorRT export;
- target-GPU FPS/accuracy.

Those paths are lazy-loaded, compile cleanly, and have explicit commands/preflight in the plan/notebook. They must be validated on the actual CUDA workstation before any accuracy/FPS claim is made.

No software-only validation can guarantee a future research accuracy or real-time target. The branch is structured to measure those results reproducibly and to fail early when proposal recall, grounding quality, negative rejection, or latency gates are not met.
