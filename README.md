# VLM / LCOD Benchmark Lab

Interactive benchmark for closed-set object detection and Language-Conditioned Object Detection (LCOD).

## Current model matrix

### Closed-set baselines
- YOLO26l
- YOLO26x
- RF-DETR-L
- RF-DETR-XL

### Open-vocabulary / grounding
- Grounding DINO
- YOLO-World
- YOLOE
- Grounding DINO 1.5 Edge
- LocateAnything-3B

The notebook keeps one common adapter interface and the existing workflow:

`Run All → preflight → select model/input/prompt → Play`

## Environment split

The main kernel and LocateAnything intentionally use different Transformers environments.

### Main kernel

Use `.env.main` as the version/spec reference. Typical install:

```powershell
python -m pip install -U "transformers>=5.1,<6" "ultralytics>=8.4.0" "rfdetr[plus]>=1.10" "dds-cloudapi-sdk>=0.5.3" ipywidgets opencv-python pillow numpy
```

Install PyTorch separately for the CUDA version on the machine.

RF-DETR-XL is supplied by the `rfdetr_plus` extension installed through `rfdetr[plus]`; it uses the Platform Model License and requires accepting that license/account entitlement at runtime.

### LocateAnything-3B

Create a dedicated interpreter and point `LOCATEANYTHING_PYTHON` in `.env.locateanything` to it.

The NVIDIA model card currently documents:

```powershell
python -m pip install opencv-python-headless==4.11.0.86 transformers==4.57.1 numpy==1.25.0 Pillow==11.1.0 peft torchvision decord==0.6.0 lmdb==1.7.5
```

PyTorch must be installed separately. LocateAnything's official model card lists Linux as the supported OS; Windows execution is therefore best-effort unless run through a Linux/WSL environment.

### Grounding DINO 1.5 Edge

This model is exposed by the official DeepDataSpace API rather than a downloadable local checkpoint. Install:

```powershell
python -m pip install "dds-cloudapi-sdk>=0.5.3"
```

Create `.env.local` (gitignored):

```dotenv
DDS_API_TOKEN=your_token_here
```

Remote API latency includes upload/network/service time and should not be interpreted as pure local GPU latency.

## Run

Open `benchmark_object_detection.ipynb` and **Run All**.

For LCOD models, enter a prompt such as:

- `bottle`
- `red bottle`
- `red bottle next to the laptop`

For command-line camera benchmarking:

```powershell
python run_camera_benchmark.py --camera 0 --warmup 10 --frames 100 --confidence 0.35 --prompt "person"
```

Benchmark a subset:

```powershell
python run_camera_benchmark.py --models yolo26l grounding_dino yolo_world --prompt "red bottle"
```

## Outputs

- interactive sessions: `outputs/live_sessions/`
- reproducible camera benchmark outputs: `outputs/camera_benchmark/`

Existing benchmark files from the previous four-model matrix are retained as historical artifacts.
