# DOD-VLM — Kế hoạch triển khai cuối cùng

## 1. Bài toán

Huấn luyện một mô hình Vision-Language cho **Described Object Detection (DOD)**:

- input: ảnh `I` + mô tả tự do `q`;
- output: tập `Y = {(b_i, s_i)}` với số phần tử có thể là 0, 1 hoặc N;
- câu có thể chứa class, thuộc tính, quan hệ và mô tả dài;
- mục tiêu nghiên cứu v1: English;
- mục tiêu triển khai về sau: >=10 FPS trên edge GPU sau khi tối ưu runtime.

## 2. Quyết định kiến trúc

### 2.1 Proposal độc lập prompt

Dùng `YOLOE-26s-seg-pf.pt`.

Lý do:
- lệnh robot có thể đổi liên tục;
- proposal không bị khóa theo prompt export;
- candidate space độc lập với query;
- fusion chịu trách nhiệm semantic matching.

### 2.2 Object token độc lập detector head

Không phụ thuộc embedding nội bộ YOLOE.

Pipeline:
`image -> frozen MobileNetV3-FPN -> MultiScaleRoIAlign(YOLOE boxes) -> 256-D object vectors`.

Điều này tách:
- localization;
- representation;
- language-conditioned reasoning.

### 2.3 Text stream

Dùng `distilbert-base-uncased`.

- toàn bộ model pretrained;
- freeze phần lớn;
- fine-tune 2 transformer layer cuối;
- giữ token-level features để cross-attention;
- encode thêm target phrase / anchor phrase cho structured path.

### 2.4 A0 — Generic fusion

Input:
- `V[B,K,256]`;
- normalized box geometry;
- `T[B,L,d]`.

Pipeline:
1. visual projection;
2. geometry projection;
3. object self-attention;
4. text cross-attention;
5. generic candidate logits.

A0 luôn tồn tại và không phụ thuộc parser.

### 2.5 A1 — Structured relational grounding

Query được parse nhẹ thành:
- target phrase;
- optional anchor phrase;
- optional relation.

A1 học:
- `S_target(i)`;
- `S_anchor(j)`;
- pairwise geometry `G(i,j)`;
- relation-conditioned compatibility `R(i,j,r)`;
- latent anchor aggregation;
- structured target logits.

Không yêu cầu anchor ground-truth box cho mọi sample; anchor là latent variable.

### 2.6 Learned fusion

`candidate_logit = (1-g(q))*A0 + g(q)*A1`

Nếu parser không tốt, gate có thể nghiêng về A0.

### 2.7 No-target

Không dùng threshold candidate làm cơ chế reject duy nhất.

Có explicit:
`null_logit = f(global object context, query context, best candidate score, detector statistics)`.

Training có no-target examples từ gRefCOCO.

### 2.8 Multi-target

Mỗi candidate dùng sigmoid độc lập, không softmax theo candidate.

## 3. Dữ liệu

### Train/val/test chính

gRefCOCO + COCO train2014 images.

gRefCOCO được dùng vì có:
- single target;
- multiple target;
- no-target.

### Schema nội bộ

Mỗi query:
- image_id;
- query;
- gt boxes;
- target categories;
- no-target flag.

## 4. Proposal / feature cache

Tính một lần trên mỗi unique image.

Cache:
- candidate boxes;
- YOLOE confidence;
- YOLOE class label;
- 256-D ROI feature;
- train GT ROI feature phục vụ optional GT injection.

Cache có signature theo:
- proposal model;
- image size;
- confidence;
- K;
- ROI output config.

## 5. GT injection

Chỉ train.

Nếu sample positive không có proposal IoU >= 0.5:
- có xác suất chèn GT ROI đã cache;
- injected candidate detector score = 0;
- validation/test/inference không dùng injection.

Mục tiêu:
- không bỏ toàn bộ supervision khi proposal miss;
- vẫn theo dõi train/test mismatch.

## 6. Loss

`L = L_candidate + 0.5 L_null + 0.2 L_target_entity`

Trong đó:
- candidate: focal BCE masked theo valid candidate;
- null: BCE;
- target entity auxiliary: tăng alignment của positive candidate với target phrase.

## 7. Training trên A3000 12 GB

- proposal + ROI cache chạy trước;
- giải phóng YOLOE + visual backbone;
- training chỉ giữ cached object vectors + text encoder + DOD core;
- AMP fp16;
- batch 12;
- grad accumulation 2;
- AdamW;
- fusion LR 3e-4;
- text LR 2e-5;
- gradient clipping;
- cosine scheduler;
- early stopping.

## 8. Checkpoint / resume

Ba lớp recovery:

1. Feature cache per image: image đã xong không tính lại.
2. `recovery.pt`: lưu định kỳ giữa epoch.
3. `last.pt` / `best.pt`: epoch-level checkpoint.

DataLoader train seed theo epoch để resume giữa epoch gần đúng thứ tự batch.

## 9. Calibration

Sau training:
- load best checkpoint;
- chạy validation một lần và cache logits;
- grid search candidate threshold và null threshold trên CPU;
- khóa threshold;
- evaluate test đúng một lần.

## 10. Metric bắt buộc

- set precision @ IoU 0.5;
- set recall @ IoU 0.5;
- set F1 @ IoU 0.5;
- no-target accuracy;
- false boxes per negative;
- train/val loss;
- peak VRAM;
- elapsed time mỗi session / epoch.

Benchmark nghiên cứu mở rộng về sau:
- D3;
- OmniLabel;
- OVDEval;
- RefCOCO/+/g legacy comparison;
- latency P50/P95 trên Jetson.

## 11. Final artifact

`dod_vlm_final.pt` chứa:
- DODVLM state dict;
- config;
- proposal checkpoint name;
- visual encoder recipe;
- calibration;
- metrics;
- architecture metadata.

Notebook phải reload artifact từ disk và infer một sample thật trước khi kết thúc.

## 12. Điều kiện notebook được coi là hoàn thành

- Run All không phụ thuộc source legacy;
- mọi code cell compile;
- core A0/A1 pass synthetic forward/backward;
- feature cache resumable;
- train resumable giữa epoch;
- best checkpoint + calibration + final bundle được tạo;
- final bundle reload được;
- runtime inference trả `[]` hoặc list boxes.
