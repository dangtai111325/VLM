# RT-LCOD Project Plan

## 0. Project decision

This branch implements one concrete architecture for **Language-Conditioned Object Detection (LCOD) for robot perception**:

> **YOLOE-26M open-vocabulary detector + lightweight slot-based attribute/relation grounding head + explicit no-target rejection + Grounding DINO teacher distillation.**

The deployment model is detector-centric. Heavy grounding/VLM models are used **offline during training only** and are not part of the robot runtime.

The first research scope is intentionally constrained so it can be trained, debugged, benchmarked, and deployed:

- exactly one **target class**;
- zero or one **attribute**;
- zero or one **first-order relation**;
- zero or one **reference object class** when a relation exists;
- no second-order relation chains such as `cup next to pillow on the bed`;
- explicit **no-target** output when the complete condition is not satisfied.

Examples in scope: `cup`, `red cup`, `red cup next to pillow`, `bottle left of laptop`, `small box under table`.

Out of scope for V1: relation chains, multi-hop logic, counting constraints, temporal language, and free-form text generation.

---

## 1. Success criteria

The project is successful only if it improves the **joint** trade-off between detection coverage, prompt understanding, rejection reliability, and real-time behavior. The goal is not to beat every foundation model on every independent benchmark.

### 1.1 Detection

Retain as much of YOLOE's open-vocabulary detection capability as possible.

Measure:

- Recall@IoU=0.5;
- mAP50 / mAP50:95 when annotations exist;
- target proposal recall@K;
- reference-object proposal recall@K;
- held-out/novel-class performance after robot-domain adaptation.

### 1.2 Language grounding

The selected object must satisfy the **entire prompt**, not only the class noun.

Measure:

- target grounding accuracy at IoU >= 0.5;
- top-1 target selection accuracy;
- attribute accuracy;
- relation accuracy;
- no-target accuracy;
- false-positive rate on negative prompts.

### 1.3 Real-time performance

For a 30 FPS source, the per-frame budget is approximately 33.3 ms.

Report:

- mean / median / p95 / p99 latency;
- processing FPS;
- source-frame deadline miss rate;
- peak VRAM;
- prompt-change latency separately from frame latency.

### 1.4 Robot perception

Report:

- target acquisition success;
- wrong-object selection rate;
- no-target rejection rate;
- reacquisition after occlusion/tracking loss;
- pick/grasp success conditioned on correct perception;
- end-to-end task success.

---

## 2. Final runtime architecture

```text
Prompt
  |
  v
Lightweight Slot Parser
  |---- target class
  |---- optional attribute
  |---- optional relation
  `---- optional reference class
  |
  v
Frozen/small text encoder -> cached prompt embeddings

Camera frame
  |
  v
YOLOE-26M open-vocabulary detector
  |
  |---- candidate boxes
  |---- detector confidences
  `---- target/reference candidate labels
  |
  +-------------------------------+
                                  |
Frame ----------------> Frozen lightweight region encoder
                                  |
                         ROI-aligned region features
                                  |
                                  v
                    Attribute-Relation Grounding Head
                       |       |         |
                       |       |         `--> no-target logit
                       |       `------------> relation score
                       `--------------------> target score
                                  |
                                  v
                         selected target box
```

### Why YOLOE-26M

YOLOE already solves the open-vocabulary candidate-generation problem. Starting from a closed-set detector would require simultaneously inventing open-vocabulary classification, region-text alignment, compositional grounding, and rejection.

The medium variant is the first student hypothesis because the large model is already close to the real-time budget on the current development machine, while the small model may sacrifice too much proposal recall. Final ablation must compare at least YOLOE-S and YOLOE-M.

### Region features

V1 uses a frozen ImageNet MobileNetV3-Small feature map + ROIAlign. This deliberately avoids undocumented YOLOE internal hooks and makes candidate caching deterministic. A later optimization can reuse a YOLOE internal feature map after the research pipeline is stable.

### Prompt representation

```text
target_class
attribute?          # max 1
relation?           # max 1
reference_class?    # required when relation exists
```

The parser runs only when the prompt changes, so its latency is excluded from steady-state frame latency and reported separately.

---

## 3. Grounding head

For each candidate object `i`:

### 3.1 Class gate

YOLOE is prompted with the target and optional reference classes. V1 hard-gates target candidates by YOLOE's returned label.

### 3.2 Attribute score

```text
A_i = similarity(attribute_projection(region_i), attribute_text_embedding)
```

If the prompt has no attribute, this term is masked out.

### 3.3 Relation score

For target candidate `i` and reference candidate `j`:

```text
R_ij = MLP([
    region_i,
    region_j,
    geometry(box_i, box_j),
    relation_text_embedding,
])
```

Geometry includes center offsets, normalized distance, size/area ratios, IoU, overlap/containment hints, and normalized target center.

The candidate relation score is the maximum compatible reference score.

### 3.4 Final decision

Each candidate receives a fused score from detector confidence, class gate, attribute score, relation score, and region features. A learned `NO_TARGET` logit is appended:

```text
candidate_0 ... candidate_N-1, NO_TARGET
```

This is mandatory for robot safety because the model must be able to refuse a prompt.

---

## 4. Teacher-student strategy

### Student

Deployment student:

```text
YOLOE-26M
+ frozen lightweight region encoder
+ cached text embeddings
+ attribute/relation grounding head
+ no-target head
```

### Teacher

Primary teacher: **Grounding DINO**.

Teacher is used offline to produce:

- boxes;
- region-language relevance;
- candidate distributions;
- pseudo labels for additional robot frames.

The teacher is never required by deployment inference.

### Distillation

Teacher outputs are aligned to student candidates and cached as a distribution over:

```text
student candidates + NO_TARGET
```

Student minimizes supervised losses plus temperature-scaled KL divergence. Teacher computation therefore adds training cost but **zero deployment latency**.

LocateAnything or another heavy VLM may be used only as an optional oracle for analysis or pseudo-label verification.

---

## 5. Canonical data schema

All source datasets are normalized into JSONL before training.

Positive example:

```json
{
  "id": "sample_000001",
  "image": "images/000001.jpg",
  "width": 1280,
  "height": 720,
  "prompt": "the red cup next to the pillow",
  "slots": {
    "target_class": "cup",
    "attribute": "red",
    "relation": "next_to",
    "reference_class": "pillow"
  },
  "target_boxes": [[124, 210, 280, 390]],
  "reference_boxes": [[310, 190, 590, 420]],
  "no_target": false,
  "source": "robot_train"
}
```

Negative example uses the same schema but has `target_boxes: []` and `no_target: true`.

Recommended dataset progression:

1. RefCOCO / RefCOCO+ / RefCOCOg — target grounding;
2. Visual Genome — attributes and pairwise relations;
3. gRefCOCO/GREC — no-target/generalized referring expressions;
4. OmniLabel — language-based/open-vocabulary detection evaluation;
5. robot-domain RGB/RGB-D data — final adaptation.

---

## 6. Candidate cache

Stage 1 keeps YOLOE fixed. For each sample, cache:

```text
sample_id
boxes                 [N,4] normalized xyxy
candidate_labels      list[str]
detector_scores       [N]
region_features       [N,D]
candidate_mask        [N]
target_mask           [N]
reference_mask        [N]
target_index          int, -1 for no-target
attribute_embedding   [T]
relation_embedding    [T]
relation_reference_target_mask [N] optional
teacher_distribution  [N+1] optional
```

Positive examples where YOLOE fails to propose the annotated target are recorded as **proposal failures**, not silently converted into no-target examples.

---

## 7. Hard-negative strategy

Hard negatives are mandatory.

Given `red cup next to pillow`, generate candidate negatives by changing exactly one slot:

- wrong class: `red bottle next to pillow`;
- wrong attribute: `blue cup next to pillow`;
- wrong relation: `red cup under pillow`;
- wrong reference: `red cup next to laptop`.

Generated candidates must be verified against annotations before becoming training negatives. Do not assume a string mutation is automatically false in the image.

Maintain a balance of positive samples, easy negatives, one-slot hard negatives, and scenes containing multiple same-class objects.

---

## 8. Training stages

### Stage 0 — baseline/environment

1. freeze package versions;
2. run CPU synthetic smoke tests;
3. run YOLOE-26M inference on a small validation set;
4. measure target/reference proposal recall;
5. benchmark raw detector latency/p95;
6. save environment metadata.

**Exit gate:** detector works on target GPU and proposal recall/latency baselines exist.

### Stage 1 — supervised grounding with frozen YOLOE

Frozen:

- YOLOE detector;
- MobileNetV3 region encoder;
- pretrained text encoder.

Trainable:

- visual/text projections;
- attribute head;
- relation head;
- target fusion head;
- no-target head.

Workflow:

1. cache train/val YOLOE candidates;
2. cache region/text embeddings;
3. train grounding head;
4. save every epoch;
5. maintain `last.pt` and `best.pt`;
6. evaluate positive/no-target/attribute/relation subsets independently.

**Exit gate:** beats `highest detector confidence` target selection on ambiguous scenes and provides useful no-target behavior.

### Stage 2 — Grounding DINO distillation

1. run teacher offline;
2. align teacher boxes with student candidates;
3. cache teacher distributions;
4. resume from Stage 1 best checkpoint;
5. train with supervised + KD loss;
6. verify accuracy gain without deployment latency change.

### Stage 3 — robot-domain adaptation

1. collect RGB/RGB-D frames across clutter, lighting, camera pose, occlusion;
2. manually annotate a high-quality subset;
3. pseudo-label additional frames with Grounding DINO;
4. human-verify high-value/hard examples;
5. generate robot hard negatives;
6. fine-tune the grounding head using mixed general + robot data;
7. keep general validation active to detect over-specialization.

### Stage 4 — optional YOLOE adaptation

Only after Stages 1–3 are stable. Fine-tune YOLOE only if proposal recall is the measured bottleneck. Any detector-weight change invalidates old candidate caches; regenerate caches and re-train/fine-tune the grounding head.

### Stage 5 — deployment optimization

1. cache prompt embeddings until prompt changes;
2. fixed input size where practical;
3. FP16 first, INT8 only after calibration/accuracy checks;
4. ONNX/TensorRT for supported components;
5. separate capture/inference/render threads;
6. latest-frame policy to avoid stale queues;
7. optional detector-every-N-frames + tracker between detections;
8. report raw inference FPS separately from tracked display FPS.

---

## 9. Losses

```text
L_total =
    L_target
  + lambda_attr * L_attribute
  + lambda_rel  * L_relation
  + lambda_none * L_no_target
  + lambda_kd   * L_distillation
```

- `L_target`: cross-entropy over N candidates + NO_TARGET;
- `L_attribute`: candidate-level attribute supervision;
- `L_relation`: verified target-reference pair supervision when available;
- `L_no_target`: explicit rejection auxiliary loss;
- `L_distillation`: temperature-scaled KL teacher/student distribution loss.

---

## 10. Prompt parser roadmap

### V1

Deterministic relation lexicon + one-hop controlled grammar. This makes grounding experiments reproducible and lets parser errors be separated from grounding errors.

### V2

Train a small token classifier with labels:

```text
O
B/I-TARGET
B/I-ATTRIBUTE
B/I-RELATION
B/I-REFERENCE
```

Evaluate parser slot F1 and exact structured-prompt accuracy independently from LCOD grounding.

---

## 11. Experiment tracking and checkpointing

Every run creates:

```text
runs/<timestamp>_<run_name>/
  config.yaml
  environment.json
  run_state.json
  metrics.jsonl
  metrics.csv
  timing.json
  checkpoints/
    last.pt
    best.pt
    epoch_XXXX.pt
```

Track:

- seed;
- git commit;
- Python/PyTorch/CUDA/GPU;
- optimizer/scheduler parameters;
- global step/epoch;
- elapsed and epoch time;
- data/forward/backward timing;
- validation metrics;
- best metric;
- peak VRAM where available.

`last.pt` stores model, optimizer, scheduler, AMP scaler, epoch/global step, best metric, and RNG states so training can resume without resetting the learning schedule.

---

## 12. Evaluation protocol

### Proposal evaluation

Before blaming grounding, measure target and reference proposal recall@K. A missing target candidate cannot be recovered by the grounding head.

### Prompt buckets

Report separately:

- class only;
- class + attribute;
- class + relation;
- class + attribute + relation;
- no-target.

### Difficulty buckets

Create splits for:

- one target-class candidate;
- multiple same-class candidates;
- attribute-confusable objects;
- relation-confusable objects;
- crowded scenes;
- target absent;
- reference absent.

### Latency modes

1. **raw**: detector + region encoder + grounding every frame;
2. **practical robot**: latest-frame policy + optional tracker.

Prompt parsing/encoding latency is measured separately.

---

## 13. Temporal tracking for robot deployment

Tracking is an optional system optimization, not part of the static LCOD metric.

V1 includes a minimal single-target IoU tracker:

- initialize from grounded target;
- update on overlapping detections;
- keep track ID while IoU/score gates pass;
- re-ground when confidence drops, object disappears, prompt changes, or track expires.

ByteTrack/BoT-SORT can replace it later.

---

## 14. Robot API/state machine

Perception API target:

```text
set_prompt(text)
process_frame(rgb, optional_depth)
-> target_box | NO_TARGET
-> confidence
-> parsed_prompt
-> timing
-> optional track_id
```

Robot state machine:

```text
WAIT_PROMPT
  -> PARSE_PROMPT
  -> ACQUIRE_TARGET
  -> TRACK_TARGET
  -> HAND_OFF_TO_GRASP
  -> VERIFY / REACQUIRE
```

Keep 2D LCOD, 3D projection, grasp planning, and manipulation control separable so perception failures are not confused with grasp/control failures.

---

## 15. Minimum ablation

| ID | Detector | Attribute | Relation | No-target | KD | Robot FT |
|---|---|---|---|---|---|---|
| A0 | YOLOE | no | no | no | no | no |
| A1 | YOLOE | yes | no | yes | no | no |
| A2 | YOLOE | yes | yes | yes | no | no |
| A3 | YOLOE | yes | yes | yes | GDINO | no |
| A4 | YOLOE | yes | yes | yes | GDINO | yes |

Also compare YOLOE-S vs YOLOE-M for final speed/accuracy Pareto selection.

---

## 16. Repository layout

```text
plan.md
rt_lcod/
  README.md
  VALIDATION.md
  pyproject.toml
  requirements.txt
  .env.example
  configs/{base.yaml,smoke.yaml}
  src/rt_lcod/
    config.py
    prompt.py
    geometry.py
    data/{schema.py,cached_dataset.py,synthetic.py}
    models/{region_encoder.py,text_encoder.py,grounding_head.py,student.py,teacher.py,yoloe_adapter.py}
    training/{losses.py,metrics.py,checkpoint.py,tracker.py,trainer.py}
    inference/{runtime.py,temporal.py}
    utils/{seed.py,logging.py,timing.py}
  scripts/
    validate_manifest.py
    cache_candidates.py
    cache_teacher.py
    generate_hard_negatives.py
    train.py
    evaluate.py
    benchmark_runtime.py
    smoke_test.py
  notebooks/rt_lcod_end_to_end.ipynb
  tests/
.github/workflows/rt_lcod_ci.yml
```

---

## 17. Beginner execution order

### Step 1 — smoke path

```bash
cd rt_lcod
python -m venv .venv
# activate environment
python -m pip install -U pip
python -m pip install -e .[dev]
python scripts/smoke_test.py
pytest -q
```

Do not proceed until this passes.

### Step 2 — notebook smoke mode

Open `notebooks/rt_lcod_end_to_end.ipynb`, keep `RUN_HEAVY=False`, and Run All.

### Step 3 — prepare/validate manifests

```bash
python scripts/validate_manifest.py --manifest data/manifests/train.jsonl
```

### Step 4 — cache YOLOE candidates

```bash
python scripts/cache_candidates.py \
  --config configs/base.yaml \
  --manifest data/manifests/train.jsonl \
  --output data/candidate_cache/train \
  --model yoloe-26m.pt
```

Repeat for validation.

### Step 5 — Stage 1 training

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --train-cache data/candidate_cache/train \
  --val-cache data/candidate_cache/val \
  --run-name stage1
```

### Step 6 — cache teacher outputs

```bash
python scripts/cache_teacher.py \
  --manifest data/manifests/train.jsonl \
  --candidate-cache data/candidate_cache/train \
  --output data/teacher_cache/train
```

### Step 7 — Stage 2 KD

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --train-cache data/candidate_cache/train \
  --val-cache data/candidate_cache/val \
  --teacher-cache data/teacher_cache/train \
  --resume runs/<stage1>/checkpoints/best.pt \
  --run-name stage2_kd
```

### Step 8 — evaluate

```bash
python scripts/evaluate.py \
  --config configs/base.yaml \
  --cache data/candidate_cache/val \
  --checkpoint runs/<run>/checkpoints/best.pt
```

### Step 9 — benchmark actual runtime

```bash
python scripts/benchmark_runtime.py \
  --config configs/base.yaml \
  --checkpoint runs/<run>/checkpoints/best.pt \
  --video ../test.mp4 \
  --prompt "red cup next to pillow"
```

### Step 10 — robot adaptation

Only after general validation works: collect robot data, annotate, add verified hard negatives/pseudo labels, fine-tune with mixed general + robot data, and rerun both general and robot benchmarks.

---

## 18. Stage gates

- **Proposal gate:** YOLOE must propose target/reference at sufficient recall.
- **Grounding gate:** learned selector must beat highest detector-confidence baseline on ambiguous scenes.
- **Negative gate:** no-target accuracy/FPR must be measured before robot use.
- **KD gate:** teacher distillation must improve validation without adding deployment cost.
- **Realtime gate:** use p95 latency on the actual target GPU, not only average FPS.
- **Robot gate:** prove perception target acquisition/rejection before interpreting manipulation success.

---

## 19. Main risks and fallback order

### YOLOE misses objects

Compare S/M/L, improve prompt synonyms, then consider mixed-data YOLOE adaptation only after proposal recall proves this is the bottleneck.

### Region encoder costs too much

Reduce top-K/channels/resolution, then replace the auxiliary encoder with a YOLOE internal feature map once the integration is stable.

### Parser dominates errors

Evaluate with oracle structured slots, then train a small slot tagger. Keep controlled-grammar robot mode until parser exact-match is adequate.

### Teacher hallucination

Never overwrite verified labels. Confidence-filter pseudo labels and manually inspect teacher/student disagreements.

### Robot fine-tuning destroys open-vocabulary ability

Keep YOLOE frozen as long as possible; mix general data into robot training and monitor held-out novel classes.

---

## 20. What can and cannot be guaranteed

The branch is engineered so the **core project code, synthetic training, checkpoint/resume, loss/metrics, prompt parsing, geometry, experiment timing, and tracker are testable without heavy model downloads**.

No code scaffold can honestly guarantee a future research accuracy/FPS before training on real datasets and benchmarking the actual robot GPU. The project therefore uses explicit measurable stage gates instead of promising an unsupported result.

The final objective is:

> maximize open-vocabulary proposal coverage, one-hop class/attribute/relation grounding, no-target reliability, and real-time throughput on the robot's actual hardware.
