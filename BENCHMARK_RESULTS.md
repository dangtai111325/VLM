# Kết quả benchmark phát hiện đối tượng — webcam

## Đang test cái gì

Đây là phép đo hiệu năng suy luận object detection trực tiếp từ **webcam 0** bằng đúng adapter, tiền xử lý và hậu xử lý mà UI trong `benchmark_object_detection.ipynb` sử dụng. Bốn phương pháp/model được đo là:

1. YOLO26m
2. RF-DETR-M
3. D-FINE-M
4. LW-DETR-M

Phép đo tập trung vào độ trễ và thông lượng suy luận. Nó **không phải** đánh giá mAP/độ chính xác, vì webcam không có ground truth nhãn.

## Phương pháp test

- Nguồn vào: webcam cục bộ số 0, độ phân giải khung hình thực nhận `640×480`.
- Confidence threshold: `0.35` (cùng mặc định UI).
- Mỗi model được load độc lập từ cache local, chạy **10 khung warm-up**, sau đó đo **100 khung**.
- Các model chạy tuần tự; mỗi khung webcam được đọc trực tiếp, không ghi ảnh hoặc video.
- `End-to-end latency` được đo quanh `Manager.predict`, có CUDA synchronization trước và sau, nên gồm tiền xử lý, suy luận và hậu xử lý; không gồm thời gian đọc frame từ webcam.
- `Adapter mean` là timer nội bộ của adapter, cùng phạm vi suy luận/pre-post-processing nhưng không có synchronization ngoài.
- `Throughput` = `1000 / end-to-end mean latency`.
- `P95` là percentile thứ 95 của 100 latency đo được. `Peak VRAM` là bộ nhớ CUDA allocated cao nhất sau khi model đã được load, nên bao gồm model resident và phần runtime.

## Môi trường test

| Thành phần | Giá trị thực tế |
|---|---|
| Thời điểm bắt đầu | 2026-09-26 20:04:14 UTC (2026-09-27 03:04:14, ICT) |
| Hệ điều hành | Windows 11, build 26200 |
| Python | 3.13.14 |
| PyTorch | 2.11.0 + CUDA 12.8 |
| Thiết bị | CUDA — NVIDIA RTX A3000 12GB Laptop GPU |
| Webcam | Camera index 0, 640×480 |
| FPS webcam | OpenCV backend không trả FPS, nên không ghi số giả định |

## Thông số performance mỗi phương pháp

| Model | Load từ cache (ms) | End-to-end mean (ms) | Median (ms) | P95 (ms) | Throughput (FPS) | Adapter mean (ms) | Peak VRAM (MiB) | Object/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO26m | 958.485 | 57.945 | 54.247 | 84.578 | 17.258 | 57.848 | 272.86 | 0.000 |
| RF-DETR-M | 7,714.615 | 162.863 | 166.424 | 193.132 | 6.140 | 159.622 | 125.46 | 0.000 |
| D-FINE-M | 2,996.764 | 280.666 | 278.886 | 311.692 | 3.563 | 280.428 | 125.52 | 0.000 |
| LW-DETR-M | 1,861.927 | 205.826 | 199.162 | 252.200 | 4.858 | 205.542 | 120.48 | 0.000 |

### Diễn giải ngắn

- **YOLO26m** nhanh nhất trong phép đo này: `17.258 FPS`, latency trung bình `57.945 ms`.
- **RF-DETR-M** đứng thứ hai về throughput (`6.140 FPS`) nhưng load checkpoint chậm nhất (`7.715 s`).
- **LW-DETR-M** nhanh hơn **D-FINE-M** (`4.858` so với `3.563 FPS`) và dùng peak allocated VRAM thấp nhất (`120.48 MiB`).
- `Object/frame = 0` vì cảnh webcam tại thời điểm test không có detection nào vượt confidence `0.35`. Chỉ số này mô tả nội dung cảnh quay, không được dùng để xếp hạng độ chính xác model.

## Dữ liệu thô và tái lập

- Tổng hợp chính xác: `outputs/camera_benchmark/camera_summary_20260926T200414Z.json`
- 400 phép đo frame: `outputs/camera_benchmark/camera_frames_20260926T200414Z.csv`
- Runner: `run_camera_benchmark.py`

Chạy lại cùng protocol:

```powershell
python run_camera_benchmark.py --camera 0 --warmup 10 --frames 100 --confidence 0.35
```

Kết quả có thể thay đổi theo cảnh webcam, nhiệt độ GPU, tác vụ nền và phiên bản driver. File report này chỉ tổng hợp duy nhất lần chạy có timestamp nêu trên.
