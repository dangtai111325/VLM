# DOD-VLM — Final End-to-End Plan

## 1. Task contract

Cho ảnh `I` và free-form description `q`, model dự đoán tập:

`Y = {(b_i, s_i)}`

với `|Y|` có thể bằng 0, 1 hoặc N.

Model phải xử lý được:

- noun/class;
- attribute như màu sắc/kích thước;
- first-order spatial relation;
- multi-target;
- absent/no-target.

Research v1 dùng English. Mục tiêu edge `>=10 FPS` là deployment target sau khi quality được xác nhận; notebook hiện tại đo latency thật nhưng không giả định đã đạt target.

---

## 2. Kiến trúc khóa

### 2.1 Prompt-independent proposal

Default: `yoloe-26s-seg-pf.pt`.

Lý do:

- query robot đổi liên tục;
- không phụ thuộc dynamic text prompt trong detector export;
- proposal space không đổi theo câu;
- language-conditioned reasoning nằm ở student VLM.

Output mỗi ảnh:

- `boxes[K,4]`;
- detector confidence;
- detector label chỉ dùng diagnostic/runtime output metadata;
- `fallback_mask[K]`.

Nếu YOLOE trả 0 box, thêm một full-image **context fallback token**. Token này chỉ giữ attention numerically stable, không tham gia candidate loss và không được xuất prediction.

### 2.2 Detector-independent object visual token

Không dùng undocumented YOLOE detection embedding.

Pipeline:

`image -> FasterRCNN GeneralizedRCNNTransform -> frozen MobileNetV3-FPN -> MultiScaleRoIAlign(YOLOE boxes) -> avg pool -> 256-D vector`

Quan trọng: dùng đúng normalization/resize transform của pretrained Faster-RCNN, đồng thời rescale proposal box sang transformed coordinates trước RoIAlign.

### 2.3 Text stream

Default: `distilbert-base-uncased`.

- pretrained English bidirectional encoder;
- freeze phần lớn model;
- fine-tune 2 layer cuối;
- full token features phục vụ A0;
- target/anchor phrase embeddings phục vụ A1.

### 2.4 Query parsing

Parser chỉ là auxiliary structured path.

Input parser:

- raw query duy nhất;
- không dùng GT category làm model input.

Output:

- target phrase;
- optional anchor phrase;
- optional relation ID.

**Cùng một hàm parser được dùng trong train và inference** để loại train/inference mismatch.

A0 luôn nhận full sentence nên parser không phải single point of failure.

### 2.5 A0 — Generic grounding

Inputs:

- visual token `V[B,K,256]`;
- box geometry;
- full text token `T[B,L,d]`.

Flow:

1. project visual token sang `d_model`;
2. cộng learned box geometry encoding;
3. self-attention giữa object tokens;
4. cross-attention object → full sentence tokens;
5. generic candidate logit.

### 2.6 A1 — Structured relational grounding

A1 học:

- scaled target-object cosine logit `S_target(i)`;
- scaled anchor-object cosine logit `S_anchor(j)`;
- pair geometry `G(i,j)` gồm relative center, width/height ratio, left/right/above/below margin, IoU, distance, overlap;
- relation-conditioned compatibility `R(i,j,r)`;
- latent anchor aggregation bằng log-sum-exp trên anchor distribution;
- structured candidate logit.

Không yêu cầu anchor GT box cho toàn bộ gRefCOCO.

### 2.7 Learned A0/A1 fusion

`p_logit(i) = (1-g(q))*A0(i) + g(q)*A1(i)`

`g(q)` học từ pooled query representation.

### 2.8 No-target

Có explicit global `null_logit` từ:

- pooled object context;
- pooled query context;
- best valid candidate logit;
- detector score statistics;
- flag có real output candidate hay không.

No-target không được giải bằng candidate threshold duy nhất.

---

## 3. Dataset

### Main train/validation/test

- gRefCOCO annotations;
- COCO train2014 images.

Query-level schema:

- sample ID;
- split;
- image ID/path;
- raw query;
- GT boxes;
- target categories **diagnostic only**;
- no-target flag.

Notebook tải chỉ các ảnh thực sự xuất hiện trong manifest.

---

## 4. Feature cache và resume

Expensive vision computation chạy một lần trên mỗi unique image.

Mỗi cache file chứa:

- boxes;
- detector scores;
- labels;
- fallback mask;
- 256-D ROI object tokens;
- train GT boxes + GT ROI features phục vụ optional GT injection;
- image size;
- cache signature.

Cache signature phụ thuộc proposal config + ROI config + encoder recipe.

Atomic save tránh file nửa chừng.

---

## 5. GT injection

Chỉ áp dụng train.

Nếu positive query không có real proposal IoU >= 0.5:

- deterministic probability theo `sample_id`;
- chèn GT ROI đã cache;
- injected detector score = 0;
- validation/test/runtime tuyệt đối không inject.

Mục đích: không mất toàn bộ fusion supervision khi proposal miss, nhưng vẫn giữ test-time proposal ceiling thực tế.

---

## 6. Training trên RTX A3000 12 GB

### Preflight

Trước optimizer training:

- chạy real cached batch;
- full forward + backward;
- nếu CUDA OOM, giảm batch `8 -> 6 -> 4 -> 2`;
- tăng gradient accumulation để effective batch xấp xỉ 24.

### Optimizer

- AdamW;
- fusion LR `3e-4`;
- text LR `2e-5`;
- AMP fp16;
- grad clip 1.0;
- cosine scheduler;
- early stopping.

### Loss

`L = 1.0*L_candidate + 0.5*L_null + 0.2*L_target_entity`

- `L_candidate`: focal BCE trên valid real/injected candidates;
- `L_null`: BCE no-target;
- `L_target_entity`: BCE alignment trên scaled target-object similarity.

### Checkpoint

- `recovery.pt`: giữa epoch, chỉ ghi sau optimizer step;
- `last.pt`: sau mỗi epoch;
- `best.pt`: best validation F1;
- lưu model/optimizer/scheduler/scaler/RNG/config/training signature.

Train DataLoader seed theo epoch và GT injection deterministic để resume reproducible hơn.

---

## 7. Calibration

Validation model inference chỉ chạy **một lần**.

Sau đó CPU:

1. fit candidate temperature;
2. fit null temperature;
3. grid search candidate threshold × null threshold bằng fast set metrics;
4. khóa calibration;
5. evaluate held-out split.

Không recompute AP trong từng grid cell.

---

## 8. Metrics

Final evaluation báo:

- set Precision@IoU0.5;
- set Recall@IoU0.5;
- set F1@IoU0.5;
- AP50;
- AP75;
- mAP@0.5:0.95;
- negative reject accuracy;
- false boxes per negative;
- null Brier score;
- null ECE;
- training loss/history;
- peak VRAM;
- elapsed session/epoch time;
- end-to-end runtime latency P50/P95 và FPS từ P50.

Proposal diagnostics báo:

- target Recall@0.5;
- target Recall@0.75;
- query-all-targets recall;
- mean recoverability 0.5:0.95.

Anchor/joint recall không được giả lập trên gRefCOCO vì benchmark không cung cấp structured anchor GT cho mọi expression; cần relation diagnostic subset riêng.

---

## 9. Final artifact

`final_model/` chứa:

- `dod_vlm_core.pt` — trained DOD-VLM state + text config;
- `tokenizer/` — local tokenizer;
- `proposal_model.pt` — local YOLOE checkpoint;
- `region_encoder.pt` — frozen backbone state + transform recipe;
- config;
- calibration;
- metrics;
- README.

Ngoài ra tạo `dod_vlm_final_bundle.zip`.

Runtime load hoàn toàn từ local model assets, không gọi Hugging Face/Ultralytics để tải weights.

Notebook bắt buộc:

1. export;
2. `del model`, GC, clear CUDA cache;
3. tạo `DODVLMRuntime` mới từ disk;
4. infer ảnh thật;
5. benchmark latency.

---

## 10. Tiêu chí hoàn thành notebook

Notebook được coi là hoàn thành về engineering khi:

- project không phụ thuộc legacy code;
- mọi code cell compile;
- dependency/CUDA preflight rõ ràng;
- real batch forward/backward preflight pass;
- zero-proposal path finite;
- feature cache resumable;
- mid-epoch + epoch checkpoint resumable;
- train/infer parser giống nhau;
- best checkpoint được calibrate;
- held-out metrics được ghi;
- final artifact self-contained về model assets;
- artifact reload + inference pass;
- latency benchmark được lưu.

Không tuyên bố “100% chạy trên mọi máy” trước khi full Run All thực tế hoàn tất trên đúng environment/hardware; notebook được thiết kế để fail-fast và resume thay vì che giấu lỗi môi trường.

---

## 11. Phase sau final research-v1

- D³;
- OmniLabel;
- OVDEval;
- Visual Genome / manually verified relation diagnostic subset;
- teacher-generated hard negatives/paraphrases;
- multilingual Vietnamese;
- YOLO/ROI TensorRT or integrated object-token engine;
- Jetson Orin P50/P95 latency và quality/latency Pareto.
