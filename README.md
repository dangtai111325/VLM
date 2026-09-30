# DOD-VLM End-to-End

Project độc lập để huấn luyện **Described Object Detection (DOD) VLM** trên NVIDIA RTX A3000 12 GB.

## Bài toán

Input:

- một ảnh RGB;
- một mô tả tự do bằng tiếng Anh.

Output:

- 0, 1 hoặc N bounding boxes thỏa **toàn bộ** mô tả, gồm class, thuộc tính và quan hệ không gian đơn giản.

Ví dụ:

- `find all cars`;
- `find all red cars`;
- `find the car next to the bench`;
- `the cup to the left of the red box`.

## Kiến trúc

1. **YOLOE-26s prompt-free** tạo K proposal độc lập query.
2. **Frozen FasterRCNN MobileNetV3-FPN + MultiScaleRoIAlign** tạo object visual token detector-independent.
3. **DistilBERT** mã hóa full sentence và target/anchor phrase; chỉ fine-tune hai transformer layer cuối.
4. **A0 generic grounding:** object self-attention + full-sentence cross-attention.
5. **A1 structured grounding:** target/anchor matching + pairwise geometry + relation-conditioned reasoning.
6. **Learned fusion gate** trộn A0/A1 theo query.
7. **Independent candidate logits** hỗ trợ multi-target.
8. **Explicit null head** học no-target.
9. **Temperature + threshold calibration** trên validation trước khi test.

## Chạy

Mở:

`notebooks/dod_vlm_end_to_end.ipynb`

rồi chọn **Run All**.

Notebook tự thực hiện:

- kiểm tra version dependency, CUDA, Torch/TorchVision compatibility và disk;
- tải + chuẩn hóa gRefCOCO;
- tải đúng ảnh COCO cần dùng;
- cache YOLOE proposals + object ROI features theo unique image;
- chạy proposal diagnostics;
- chạy preflight forward/backward thật và tự giảm batch nếu A3000 OOM;
- train/resume DOD-VLM;
- lưu recovery / last / best checkpoints;
- calibrate temperature và threshold;
- evaluate test/testA/testB nếu có;
- báo F1, AP50, AP75, mAP@0.5:0.95, no-target metrics và calibration metrics;
- đóng gói artifact self-contained về model weights/assets;
- giải phóng VRAM;
- reload artifact từ disk;
- inference ảnh thật và benchmark P50/P95 latency.

## Resume khi bị ngắt

Có ba lớp resume:

1. image feature cache: ảnh đã cache hợp lệ được skip;
2. `recovery.pt`: lưu định kỳ giữa epoch, chỉ sau optimizer step;
3. `last.pt` và `best.pt`: epoch-level checkpoint.

Chạy lại **Run All** sẽ tái sử dụng dữ liệu/cache/checkpoint tương thích.

## Artifact cuối

Giữ nguyên thư mục:

`dod_vlm_workspace/runs/dod_vlm_grefcoco_v2/final_model/`

hoặc file:

`dod_vlm_workspace/runs/dod_vlm_grefcoco_v2/dod_vlm_final_bundle.zip`

Artifact chứa trained multimodal core, tokenizer, YOLOE local checkpoint, frozen region encoder, config, calibration và metrics. Runtime reference là `dod_vlm.runtime.DODVLMRuntime`.

## Yêu cầu môi trường

- Python >= 3.10;
- NVIDIA RTX A3000 12 GB hoặc GPU CUDA tương đương;
- PyTorch >= 2.3 với CUDA;
- TorchVision release line tương thích với PyTorch;
- Internet trong phase chuẩn bị dữ liệu/model pretrained;
- khuyến nghị >= 30 GB disk trống.

> Notebook **không tự thay PyTorch** để tránh vô tình cài wheel CPU hoặc wheel CUDA không tương thích. Hãy cài PyTorch/TorchVision CUDA chính thức trước khi Run All.

## Phạm vi nghiên cứu

Research v1 dùng English + gRefCOCO. D³, OmniLabel, OVDEval, Visual Genome relation diagnostics, Vietnamese và TensorRT/Jetson là các phase mở rộng sau khi pipeline chính ổn định.
