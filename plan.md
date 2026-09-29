# RT-LCOD / DOD — PLAN V3: End-to-End Final Model

> Target machine: NVIDIA RTX A3000 12 GB (Precision laptop)
>
> Goal: one reproducible pipeline from raw public data to a final Described Object Detection (DOD) model. Input is an RGB image plus a free-form English description; output is 0, 1, or N bounding boxes satisfying the complete description.

## 0. Locked decisions

This branch no longer treats intermediate experiments as user-facing stages. Proposal diagnostics, coverage checks, calibration, and validation remain inside the pipeline as instrumentation, but a normal run proceeds end-to-end and produces a final model artifact.

The final system is intentionally not an end-to-end coordinate-generating VLM. It separates localization from language-conditioned reasoning so that localization can stay fast and the semantic model can be trained on a 12 GB GPU.

Locked design:

1. **Prompt-independent proposal generation**: YOLOE-26s prompt-free is the default proposal source. It must not depend on the current command.
2. **Detector-independent object tokens**: a frozen visual backbone plus RoIAlign converts each proposal into a per-object visual vector. We do not depend on undocumented YOLOE detection embeddings.
3. **Frozen bidirectional text encoder**: DistilBERT is the default research-v1 encoder. It produces full-sentence token features and pooled entity/query features.
4. **A0 generic branch**: object self-attention followed by text cross-attention gives every candidate a full-sentence score.
5. **A1 structured branch**: target/anchor entity matching plus explicit pairwise geometry/relation compatibility gives a second candidate score.
6. **Learned A0/A1 fusion**: the model learns how much structured reasoning to use. Parser failure therefore degrades toward A0 rather than crashing inference.
7. **Explicit no-target/null head**: rejecting an absent target is a learned task, not merely a high candidate threshold.
8. **Independent candidate sigmoid scores**: multi-target output is allowed by construction.
9. **Validation-time calibration**: null and candidate decisions are calibrated on the validation split before final test reporting.
10. **Cache expensive image computation**: detector and ROI features are computed once per unique image, saved to disk, then unloaded before DOD-core training.

Research v1 is English. Vietnamese is a later extension using either translation-to-English or a multilingual encoder after the English pipeline is stable.

---

# 1. Task contract

For image `I` and free-form description `q`, predict a set

`Y = {(b_i, p_i)}; |Y| in {0,1,...,N}`

where `b_i = [x1,y1,x2,y2]` and `p_i` is the calibrated match probability.

The query may contain:

- object class / noun phrase;
- attributes such as color, size, material or state;
- spatial relation, e.g. left/right/above/below/near;
- one or more contextual/anchor objects;
- expressions that match multiple objects;
- expressions for which no valid object exists.

The final model must therefore distinguish three separate questions:

- **Localization**: are all potentially relevant objects present among proposals?
- **Grounding**: which proposal(s) satisfy the query?
- **Existence/rejection**: is there no valid target at all?

These questions are trained/evaluated separately internally, but are exposed as one final model API.

---

# 2. Final architecture

## 2.1 Overall dataflow

```text
QUERY q
  |
  +--> lightweight parser ------------------> QueryGraph
  |                                           target phrase
  |                                           anchor phrase(s)
  |                                           relation id
  |                                           parser-valid mask
  |
  +--> frozen text encoder -----------------> token features T [B,L,Dt]
                                              pooled full-query tq [B,Dt]
                                              target entity et [B,Dt]
                                              anchor entity ea [B,Dt]

IMAGE I
  |
  +--> prompt-independent YOLOE proposal ----> boxes Bx [B,K,4]
  |                                            detector score s [B,K]
  |                                            optional detector label
  |
  +--> frozen visual backbone + RoIAlign ----> visual ROI V [B,K,Dv]
                                               |
                                               + detector-label text prior Elabel [B,K,Dt]
                                               + box geometry Gbox [B,K,Dg]
                                               + detector score
                                               |
                                               v
                                          Object Token X [B,K,D]
                                               |
                      +------------------------+------------------------+
                      |                                                 |
                      v                                                 v
               A0 generic branch                                A1 structured branch
               object self-attention                            entity-object matching
               text cross-attention                             pairwise relation score
                      |                                                 |
                generic logits g_i                              structured logits r_i
                      +-------------------------+-----------------------+
                                                |
                                          learned gate alpha
                                                |
                                      candidate logit z_i
                                                |
                           +--------------------+--------------------+
                           |                                         |
                           v                                         v
                    sigmoid candidates                         explicit null head
                           |                                         |
                           +--------------------+--------------------+
                                                |
                                      calibrated 0 / 1 / N boxes
```

## 2.2 Proposal generator

Default checkpoint: `yoloe-26s-seg-pf.pt`.

Requirements:

- prompt-independent at inference;
- low confidence threshold, default 0.02;
- `max_det=96`, retain top `K=64` after score sort;
- input resolution 640;
- proposal output is `[box_xyxy, detector_score, optional detector_label]`;
- proposal class label is only an optional semantic prior, never a hard target/reference mask.

Reason: relational grounding requires both target and contextual objects to exist in the candidate set. A head-noun-only detector can make reasoning impossible before the fusion module sees the image.

## 2.3 Detector-independent visual object token

Do not use an undocumented hidden YOLOE tensor as the main object representation. Use a separate frozen visual path:

1. Resize RGB image to `512 x 512` while tracking original geometry.
2. Frozen `MobileNetV3-Small` ImageNet backbone.
3. RoIAlign each candidate box on the final feature map.
4. Pool a `3 x 3` crop to one 576-dimensional vector.

For candidate `i`, build:

`x_i = LN(Wv * ROI_i + Wb * geom(box_i) + Ws * score_i + Wl * label_embedding_i)`

Recommended normalized single-box geometry:

- center x/y;
- width/height;
- area;
- log aspect ratio.

`label_embedding_i` is optional. It is useful as a prior but must be ablated later to prove the fusion model does not merely reuse detector labels.

Tensor contract after padding:

- `visual`: `[B,K,576]`
- `boxes`: `[B,K,4]`, normalized to [0,1]
- `scores`: `[B,K]`
- `label_emb`: `[B,K,Dt]`
- `candidate_mask`: `[B,K]` boolean
- object token `X`: `[B,K,D]`, default D=256.

## 2.4 Text representation

Default encoder: `distilbert-base-uncased`, frozen initially.

For the full query:

- tokenize to max length 48;
- obtain token states `T_raw [B,L,Dt]`;
- project to model width: `T = Wt(T_raw) [B,L,D]`;
- mean-pool valid tokens for full-query vector `q_pool [B,Dt]`.

For structured A1, encode the parser-produced target phrase and anchor phrase separately to `e_target` and `e_anchor`.

Text computation is cacheable per command at real-time inference.

## 2.5 Query parser / QueryGraph

The parser is a helper, not the sole semantic path. A0 always receives the complete original sentence.

Minimum QueryGraph fields:

```python
{
  "raw": str,
  "target_phrase": str,
  "anchor_phrase": str | None,
  "relation": one_of[
      "none", "left_of", "right_of", "above", "below",
      "near", "in_front_of", "behind", "inside", "on_top_of"
  ],
  "has_relation": bool,
  "parser_confidence": float | None,
}
```

Research-v1 parser may be deterministic for common relations. Later versions can replace it with an LLM parser executed once per command. Parser failure must set `has_relation=False`, preserving A0 behavior.

## 2.6 A0 — generic full-sentence fusion

Purpose: strong parser-independent baseline and residual semantic path.

Pipeline:

1. `X0 = object_token_builder(...)`
2. object self-attention: `X1 = TransformerEncoder(X0, mask)`
3. repeat `Ncross` blocks:
   - candidate-to-text multi-head cross-attention;
   - residual + LayerNorm;
   - FFN + residual + LayerNorm.
4. generic logit: `g_i = MLP_generic(X_final_i)`.

Default:

- D=256;
- heads=8;
- object layers=2;
- cross layers=2;
- FFN=512;
- dropout=0.1.

Why self-attention before text cross-attention: an object should see other objects before deciding whether relational language applies. This creates an implicit scene context even when the parser is imperfect.

## 2.7 A1 — structured entity/relation reasoning

A1 exists to make relational reasoning explicit and diagnosable.

### Entity-object matching

Project object token and entity text into a normalized shared space:

`u_i = normalize(Wo X_i)`

`v_t = normalize(We e_target)`

`v_a = normalize(We e_anchor)`

Scores:

`S_target(i) = scale * dot(u_i, v_t)`

`S_anchor(j) = scale * dot(u_j, v_a)`.

A1 never receives hard `target_mask/reference_mask` from YOLOE.

### Anchor pruning

Computing every pair is O(K^2). On A3000, rank candidates using `S_anchor` and retain top `M=8` anchors. Then relation computation is O(K*M).

### Pairwise geometry

For target candidate i and anchor j, compute a relative geometry vector including:

- normalized center dx/dy;
- Euclidean center distance;
- log width ratio;
- log height ratio;
- left/right signed margins;
- above/below signed margins;
- pair IoU.

`G_ij` is passed through an MLP to a geometry embedding.

### Relation compatibility

Embed relation id `r` into `e_r`.

For each target-anchor pair:

`R_ij = MLP_rel([X_i, X_j, e_r, MLP_geom(G_ij)]) + S_anchor(j)`.

Aggregate over top anchors using `logsumexp`, not hard argmax:

`rel_i = logsumexp_j(R_ij) - log(M)`.

Structured score:

- no relation: `r_i = S_target(i)`;
- relation present: `r_i = S_target(i) + rel_i`.

This branch directly answers: “candidate i looks like the requested target, and there exists a plausible anchor j satisfying relation r.”

## 2.8 A0/A1 fusion

Learn a sample-level gate from pooled full-query representation and `has_relation`:

`alpha = sigmoid(MLP_gate([q_pool, has_relation]))`.

Final candidate logit:

`z_i = g_i + alpha * r_i`.

This is deliberately residual. A1 adds structured evidence; it does not erase the generic full-sentence path.

## 2.9 Explicit no-target head

A sigmoid per candidate is not sufficient to reject absent targets. Build a separate null head from:

- pooled contextual object representation;
- pooled query representation;
- best candidate logit;
- optionally proposal-count/statistical features.

`z_null = MLP_null([...])`.

Training label:

- `y_null=1` if GT target set is empty;
- `y_null=0` otherwise.

At inference, reject when calibrated null probability exceeds `tau_null`, or when no candidate exceeds `tau_candidate`.

## 2.10 Multi-target behavior

Do **not** softmax candidates against each other. Use independent BCE/focal logits. A query can therefore activate more than one candidate. Apply NMS only after semantic thresholding.

---

# 3. Dataset pipeline

## 3.1 Main training set

Use official gRefCOCO annotations from `FudanCVL/gRefCOCO` plus MS COCO train2014 images.

Normalize each sentence into one row:

```text
sample_id
image_id
file_name
split
query
gt_boxes_xyxy[]
no_target
category_names[]
```

One image may have multiple sentences; proposal/ROI cache is keyed by image, not sentence.

## 3.2 Splits

Respect dataset-provided split names. Do not leak validation/test sentences or images into training transformations that learn parameters.

Training: `train`.
Validation: `val` for early stopping and calibration.
Held-out reporting: test/testA/testB when available.

## 3.3 Cache schema

One file per unique image:

```python
{
  "image_id": int,
  "file_name": str,
  "width": int,
  "height": int,
  "boxes_xyxy": FloatTensor[K,4],
  "scores": FloatTensor[K],
  "labels": list[str],
  "roi_features": Float16Tensor[K,576],
  "gt_bank_boxes": FloatTensor[G,4],
  "gt_bank_features": Float16Tensor[G,576],
}
```

Cache metadata must record proposal checkpoint, input size, confidence threshold, K, ROI backbone, ROI input size, and code/config version. If any of these changes, invalidate the cache.

## 3.4 Training-only GT injection

Proposal recall limits the student. For a training sample, if a GT box has no proposal with IoU >= 0.5, inject that GT box plus its cached ROI feature into the candidate list.

Rules:

- training only;
- GT-injected object has detector score 0;
- no detector label semantic prior;
- never inject at validation/test/inference;
- record injection rate as a diagnostic.

This provides a learning signal for the grounding model while exposing proposal misses rather than silently discarding those examples. High injection rate means proposal generation is the bottleneck and must be improved before claiming final system quality.

## 3.5 Candidate labels

Match GT boxes to candidates by IoU. Greedy one-candidate-per-GT assignment is preferred to avoid many duplicate positives for one object. Unmatched valid candidates are negatives.

No-target samples contain all-zero candidate labels and `null=1`.

---

# 4. Training strategy for 12 GB VRAM

## 4.1 Precompute phase

On each unique image:

1. load image;
2. YOLOE prompt-free inference under inference_mode + FP16;
3. keep low-threshold top proposals;
4. frozen MobileNetV3 forward;
5. RoIAlign proposal features;
6. RoIAlign all unique GT boxes used by train expressions;
7. save cache to CPU/disk.

Then delete YOLOE and MobileNet objects and call CUDA cache cleanup.

This is essential: training should not hold detector + visual backbone + text encoder + fusion gradients simultaneously.

## 4.2 Text phase

Keep DistilBERT frozen. It runs under `torch.inference_mode()`. For further optimization, sentence/entity embeddings may be cached after the first working run.

## 4.3 DOD-core training

Train only A0/A1/null/fusion parameters initially.

Recommended A3000 defaults:

- batch size 16;
- AMP FP16;
- AdamW;
- lr `3e-4`;
- weight decay `1e-4`;
- grad clip 1.0;
- 10 epochs max;
- early stopping patience 3;
- DataLoader workers 0 on Windows/Jupyter for reliability; increase only after validation.

Loss:

`L = lambda_c * L_candidate + lambda_n * L_null + lambda_e * L_entity + lambda_r * L_rank`

Initial weights:

- candidate 1.0;
- null 0.45;
- entity 0.20;
- optional ranking 0.10.

Candidate/entity loss: focal BCE to handle severe positive/negative imbalance.

Null loss: BCEWithLogits.

Structured auxiliary supervision should use candidate-GT matching, not detector labels.

## 4.4 Training checks that do not stop the pipeline

Log every epoch:

- train/val total loss;
- candidate positive recall at selected threshold;
- candidate precision;
- no-target accuracy;
- no-target FPR/FPPC if available;
- GT-injection fraction;
- mean proposals/image;
- GPU max allocated/reserved memory;
- data time and step time.

Keep best checkpoint by validation objective, not the final epoch.

---

# 5. Calibration and final decision rule

After loading the best checkpoint, collect validation logits.

Search candidate threshold `tau_c` and null threshold `tau_n` on a fixed grid. Objective should balance target F1 and no-target accuracy; record the exact objective.

Optional next step: temperature scaling for candidate and null logits. Calibration parameters belong in the final model bundle.

Inference:

1. calculate null probability;
2. if `p_null >= tau_n`: output empty set;
3. otherwise sigmoid each candidate logit;
4. retain `p_i >= tau_c`;
5. semantic NMS, default IoU 0.5;
6. return up to `max_outputs` boxes.

---

# 6. Benchmark contract

The final notebook must at minimum report:

## Proposal diagnostics

- Recall@K at IoU 0.50 and 0.75;
- train GT-injection rate;
- proposal count distribution.

For a relation-annotated diagnostic subset, additionally report target recall, anchor recall, and joint target+anchor recall. Do not fabricate anchor ground truth from a parser.

## DOD/grounding

- precision / recall / F1 at IoU 0.50;
- precision / recall / F1 at IoU 0.75;
- no-target accuracy;
- no-target false positives / FPPC when evaluator supports it;
- multi-target recall;
- per-query-length and relation/non-relation slices.

When official D3/OmniLabel/OVDEval evaluators are added, report their native metrics unchanged. Do not compare Recall@K to mAP numerically; use an oracle prediction generated from the same proposal set and run it through the same official evaluator when estimating headroom.

## Calibration

- selected `tau_candidate`;
- selected `tau_null`;
- optional ECE/Brier score;
- validation objective used to select thresholds.

## Runtime

Measure after accuracy is functional:

- proposal P50/P95;
- ROI object-token extraction P50/P95;
- text encoding latency when command changes;
- DOD core P50/P95;
- end-to-end P50/P95;
- FPS for repeated frames with cached text;
- peak VRAM.

The deployment target remains >=10 FPS on the eventual edge target, but training-machine numbers are reference only.

---

# 7. Final artifact format

A successful full notebook run creates:

```text
rt_lcod/artifacts/dod_final_model/
  final_dod_bundle.pt
  region_encoder.pt
  proposal_model.pt              # if local checkpoint was resolved/copied
  text_encoder/
    config.json
    model.safetensors
    tokenizer files...
  calibration.json
  metrics.json
  training_history.csv
  dataset_metadata.json
  config.json
```

`final_dod_bundle.pt` contains at least:

- architecture version;
- learned DOD core state dict;
- dimensions / relation vocabulary;
- full training configuration;
- candidate/null calibration;
- proposal model identifier;
- text encoder identifier;
- dataset/cache metadata hash where practical.

The reference runtime must be able to start from only the artifact directory plus Python dependencies and execute:

```python
runtime = FinalDODRuntime(artifact_dir)
result = runtime.predict(image, "the cup to the left of the red box")
# result -> list[{box:[x1,y1,x2,y2], score:float}]
```

---

# 8. Notebook execution contract

New notebook path:

`rt_lcod/notebooks/rt_lcod_dod_final_end_to_end.ipynb`

Normal usage:

1. use a CUDA-enabled PyTorch environment;
2. open notebook from `rt_lcod/notebooks`;
3. edit only the single configuration cell if paths/hyperparameters must change;
4. leave `quick=False` for final training;
5. Run All;
6. wait for data download/cache/training/evaluation to finish;
7. inspect `../artifacts/dod_final_model/final_dod_bundle.pt` and metrics;
8. execute the final inference cell on a held-out/local image.

`quick=True` is permitted only to verify mechanics on a small subset. A quick-run checkpoint must never be presented as the final research result.

The pipeline is idempotent where possible: downloaded annotations/images and per-image caches are reused when present.

---

# 9. Failure handling built into the end-to-end run

The notebook should fail early with an actionable error for:

- CPU-only torch when user requested a full run;
- missing/corrupt gRefCOCO annotations;
- missing COCO images after download;
- YOLOE prompt-free checkpoint incompatibility;
- NaN/Inf training loss;
- malformed cache tensor shapes;
- artifact reload failure.

It should not silently continue when data is incomplete.

Operational OOM fallback order for A3000 12 GB:

1. reduce training batch 16 -> 8 -> 4;
2. leave K=64 if possible because K affects proposal ceiling;
3. reduce d_model 256 -> 192 only if needed;
4. cache text features so DistilBERT is not resident during training;
5. never solve OOM by increasing proposal confidence first, because that can destroy recall.

---

# 10. Post-v1 extensions (not required for first final model)

Only after the English gRefCOCO final model is reproducible:

- replace rule parser with one-shot LLM structured parser;
- multilingual comparison: Vietnamese -> English translation vs multilingual encoder;
- add teacher-generated hard negatives and paraphrases;
- add relation-annotated Visual Genome diagnostic training/subsets;
- official D3, OmniLabel, OVDEval evaluation;
- distill/quantize text and fusion modules;
- TensorRT proposal path and ONNX/TensorRT DOD-core export;
- integrate tracking for video/robot perception;
- replace MobileNet ROI backbone with a feature pyramid only if accuracy justifies latency.

The final research question remains:

> Can a prompt-independent object proposal set plus a lightweight object-token language-relation model achieve useful 0/1/N described-object detection while preserving a deployable latency budget?
