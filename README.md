# DOD-VLM End-to-End

Project độc lập để huấn luyện **Described Object Detection VLM** trên RTX A3000 12 GB.

## Mục tiêu

Input:
- 1 ảnh RGB
- 1 mô tả tự do bằng tiếng Anh

Output:
- 0, 1 hoặc N bounding boxes thỏa mô tả.

## Kiến trúc

1. **YOLOE-26s prompt-free** tạo proposal độc lập với câu.
2. **Frozen MobileNetV3-FPN + MultiScaleRoIAlign** tạo object visual tokens.
3. **DistilBERT** mã hóa toàn câu và entity phrase; 2 transformer layer cuối được fine-tune.
4. **A0 generic fusion:** object self-attention + text cross-attention.
5. **A1 structured fusion:** target/anchor matching + relation-conditioned pairwise geometry.
6. **Learned fusion gate** trộn A0 và A1.
7. **Candidate sigmoid heads** hỗ trợ 0/1/N.
8. **Explicit null head** học no-target.
9. **Validation calibration** chọn threshold trước khi test.

## Chạy

Mở:

`notebooks/dod_vlm_end_to_end.ipynb`

và chọn **Run All**.

Notebook tự:
- cài dependency còn thiếu;
- tải gRefCOCO annotation;
- tải đúng ảnh COCO cần dùng;
- cache YOLOE proposal + object ROI feature;
- resume cache/checkpoint nếu bị gián đoạn;
- train VLM;
- calibrate;
- evaluate;
- lưu final bundle;
- reload bundle và chạy inference sanity check.

## Artifact cuối

`dod_vlm_workspace/runs/dod_vlm_grefcoco_v1/final_model/dod_vlm_final.pt`

## Yêu cầu máy

- Python >= 3.10
- NVIDIA RTX A3000 12 GB
- PyTorch có CUDA
- đủ disk cho ảnh COCO được dùng + feature cache

Toàn bộ logic nghiên cứu nằm trong notebook; project không phụ thuộc code legacy.
