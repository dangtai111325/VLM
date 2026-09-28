from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any

import torch

from rt_lcod.config import ExperimentConfig, load_config


@dataclass
class RunAllState:
    stage1_run: str | None = None
    stage1_checkpoint: str | None = None
    stage2_run: str | None = None
    stage2_checkpoint: str | None = None
    final_checkpoint: str | None = None
    final_test_report: str | None = None


class RunAll:
    """One-kernel orchestration for the local RTX A3000 RT-LCOD notebook.

    Heavy detector/teacher work is launched in child processes. This keeps GPU memory
    predictable and makes every stage independently resumable from files on disk.
    """

    TOTAL_SESSIONS = 11

    def __init__(
        self,
        root: str | Path,
        train_samples: int = 2000,
        val_samples: int = 300,
        test_samples: int = 300,
        teacher_max_samples: int = 800,
        run_teacher: bool = True,
        rebuild_candidate_cache: bool = False,
        config_name: str | Path = "local_a3000",
        data_download_workers: int = 12,
        prefer_http_coco: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        config_path = Path(config_name)
        if config_path.suffix != ".yaml":
            config_path = config_path.with_suffix(".yaml")
        self.config_path = (
            config_path if config_path.is_absolute() else self.root / "configs" / config_path
        )
        if not self.config_path.is_file():
            raise FileNotFoundError(f"RunAll config not found: {self.config_path}")
        self.config: ExperimentConfig = load_config(self.config_path)
        self.data = self.root / "data"
        self.cache_root = self.data / "candidate_cache"
        self.teacher_cache = self.data / "teacher_cache" / "train"
        self.runs = self.root / "runs"
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.train_samples = int(train_samples)
        self.val_samples = int(val_samples)
        self.test_samples = int(test_samples)
        self.teacher_max_samples = int(teacher_max_samples)
        self.run_teacher = bool(run_teacher)
        self.rebuild_candidate_cache = bool(rebuild_candidate_cache)
        self.data_download_workers = max(1, int(data_download_workers))
        self.prefer_http_coco = bool(prefer_http_coco)
        self.state = RunAllState()
        self.runtime = None

    def _banner(self, name: str) -> float:
        match = re.search(r"Session\s+(\d+)", name)
        session = int(match.group(1)) if match else None
        print("\n" + "=" * 96)
        print(f"[RUNALL] {name}")
        if session is not None:
            print(
                f"[PROGRESS] pipeline_session={session}/{self.TOTAL_SESSIONS} "
                f"completed={100 * (session - 1) / self.TOTAL_SESSIONS:.0f}%"
            )
        print(f"[RUNALL] started={datetime.now().isoformat(timespec='seconds')}")
        print("=" * 96)
        return time.perf_counter()

    @staticmethod
    def _finish(name: str, started: float) -> None:
        print(f"[RUNALL] {name} completed elapsed_s={time.perf_counter() - started:.2f}")
        print("=" * 96)

    def _cmd(self, args: list[Any], label: str = "child process") -> None:
        if label == "child process" and len(args) > 1:
            label = Path(str(args[1])).stem
        print("[CMD]", " ".join(map(str, args)), flush=True)
        started = time.perf_counter()
        process = subprocess.Popen([str(x) for x in args], cwd=self.root)
        next_heartbeat = 30.0
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            if elapsed >= next_heartbeat:
                print(
                    f"[PROGRESS] {label}: still_running elapsed_m={elapsed / 60:.1f}; "
                    "use the script's tqdm/EPOCH line above for current percent and ETA.",
                    flush=True,
                )
                next_heartbeat += 30.0
            time.sleep(1.0)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, [str(x) for x in args])
        print(f"[CMD] {label}: rc=0 elapsed_s={time.perf_counter() - started:.2f}")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _gpu(self, tag: str) -> None:
        if not torch.cuda.is_available():
            print(f"[GPU][{tag}] CUDA unavailable")
            return
        props = torch.cuda.get_device_properties(0)
        print(
            f"[GPU][{tag}] name={props.name} total_GiB={props.total_memory / 2**30:.2f} "
            f"alloc_MiB={torch.cuda.memory_allocated() / 2**20:.1f} "
            f"reserved_MiB={torch.cuda.memory_reserved() / 2**20:.1f} "
            f"peak_MiB={torch.cuda.max_memory_allocated() / 2**20:.1f}"
        )

    def _save_state(self) -> None:
        path = self.artifacts / "runall_state.json"
        path.write_text(json.dumps(asdict(self.state), indent=2) + "\n", encoding="utf-8")
        print(f"[RUNALL] state={path}")

    def preflight(self) -> None:
        started = self._banner("Session 1 — GPU/CUDA preflight")
        print(f"[ENV] root={self.root}")
        print(f"[ENV] python={sys.version.replace(chr(10), ' ')}")
        print(f"[ENV] platform={platform.platform()}")
        print(f"[ENV] torch={torch.__version__} cuda_runtime={torch.version.cuda}")
        print(f"[ENV] cuda_available={torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required. Start Jupyter with the prepared Python 3.14 RTX runtime.")
        print(f"[ENV] gpu={torch.cuda.get_device_name(0)}")
        try:
            print(subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip())
        except Exception as exc:
            print(f"[ENV][WARN] nvidia-smi unavailable: {exc}")
        print(f"[ENV] config={self.config}")
        self._gpu("preflight")
        self._finish("preflight", started)

    def prepare_data(self) -> None:
        started = self._banner("Session 2 — download/prepare gRefCOCO")
        command: list[Any] = [
            sys.executable,
            "scripts/prepare_grefcoco.py",
            "--root", self.data,
            "--train", self.train_samples,
            "--val", self.val_samples,
            "--test", self.test_samples,
            "--seed", self.config.seed,
            "--negative-fraction", 0.30,
            "--workers", self.data_download_workers,
            "--allow-http-fallback",
        ]
        if self.prefer_http_coco:
            command.append("--prefer-http-coco")
            print("[DATA][WARN] using public HTTP COCO download due to the configured local network TLS issue")
        self._cmd(command)
        for split in ("train", "val", "test"):
            manifest = self.data / "manifests" / f"{split}.jsonl"
            self._cmd([sys.executable, "scripts/validate_manifest.py", "--manifest", manifest])
            print(f"[DATA] split={split} manifest={manifest} bytes={manifest.stat().st_size:,}")
            with manifest.open("r", encoding="utf-8") as handle:
                for _ in range(2):
                    print("[DATA] sample", handle.readline().strip()[:1000])
        print(json.dumps(self._read_json(self.data / "data_report.json"), indent=2, ensure_ascii=False))
        self._finish("prepare_data", started)

    def cache_candidates(self) -> None:
        started = self._banner("Session 3 — YOLOE + frozen feature caches")
        for split in ("train", "val", "test"):
            output = self.cache_root / split
            cmd: list[Any] = [
                sys.executable,
                "scripts/cache_candidates.py",
                "--config", self.config_path,
                "--manifest", self.data / "manifests" / f"{split}.jsonl",
                "--output", output,
                "--model", self.config.model.detector_name,
                "--confidence", self.config.runtime.get("confidence", 0.15),
                "--max-candidates", self.config.runtime.get("top_k", self.config.model.top_k),
            ]
            if self.rebuild_candidate_cache:
                cmd.append("--no-resume")
            self._cmd(cmd)
            report = self._read_json(output / "_cache_stats.json")
            print(f"[CACHE] split={split}")
            print(json.dumps(report, indent=2))
            if split == "train" and report.get("positive_proposal_recall_at_iou", 0.0) < 0.70:
                print("[CACHE][WARN] target proposal recall < 70%; this limits the grounding ceiling.")
        self._gpu("after-caches")
        self._finish("cache_candidates", started)

    def _latest_new_run(self, pattern: str, before: set[Path]) -> Path:
        after = set(self.runs.glob(pattern))
        candidates = sorted(after - before, key=lambda p: p.stat().st_mtime)
        if not candidates:
            candidates = sorted(after, key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise RuntimeError(f"no run directory matched {pattern}")
        return candidates[-1]

    def _print_tail_metrics(self, run_dir: Path) -> None:
        path = run_dir / "metrics.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        print(f"[METRICS] file={path} epochs={len(lines)}")
        for line in lines[-min(5, len(lines)):]:
            print(json.dumps(json.loads(line), indent=2))
        timing = run_dir / "timing.json"
        if timing.exists():
            print(f"[TIMING] {timing.read_text(encoding='utf-8')}")

    def train_stage1(self) -> None:
        started = self._banner("Session 4 — Stage 1 supervised student training")
        self.runs.mkdir(exist_ok=True)
        before = set(self.runs.glob("*_stage1_local"))
        self._cmd([
            sys.executable,
            "scripts/train.py",
            "--config", self.config_path,
            "--train-cache", self.cache_root / "train",
            "--val-cache", self.cache_root / "val",
            "--runs-root", self.runs,
            "--run-name", "stage1_local",
        ])
        run = self._latest_new_run("*_stage1_local", before)
        checkpoint = run / "checkpoints" / "best.pt"
        self.state.stage1_run = str(run)
        self.state.stage1_checkpoint = str(checkpoint)
        print(f"[STAGE1] run={run}")
        print(f"[STAGE1] best={checkpoint}")
        self._print_tail_metrics(run)
        self._save_state()
        self._gpu("after-stage1")
        self._finish("train_stage1", started)

    def evaluate_checkpoint(self, checkpoint: Path, split: str, name: str) -> dict[str, Any]:
        output = self.artifacts / f"{name}_{split}.json"
        self._cmd([
            sys.executable,
            "scripts/evaluate.py",
            "--config", self.config_path,
            "--cache", self.cache_root / split,
            "--checkpoint", checkpoint,
            "--output", output,
        ])
        report = self._read_json(output)
        print(f"[EVAL] {name}/{split}={json.dumps(report, indent=2)}")
        return report

    def evaluate_stage1_validation(self) -> dict[str, Any]:
        started = self._banner("Session 5 — Stage 1 validation")
        if not self.state.stage1_checkpoint:
            raise RuntimeError("run train_stage1() first")
        report = self.evaluate_checkpoint(Path(self.state.stage1_checkpoint), "val", "stage1")
        self._finish("evaluate_stage1_validation", started)
        return report

    def cache_teacher_outputs(self) -> None:
        started = self._banner("Session 6 — Grounding DINO offline teacher")
        if not self.run_teacher:
            print("[TEACHER] disabled; skipping")
            self._finish("cache_teacher_outputs", started)
            return
        cmd: list[Any] = [
            sys.executable,
            "scripts/cache_teacher.py",
            "--manifest", self.data / "manifests" / "train.jsonl",
            "--candidate-cache", self.cache_root / "train",
            "--output", self.teacher_cache,
            "--model", "IDEA-Research/grounding-dino-base",
            "--threshold", 0.20,
            "--temperature", 0.20,
        ]
        if self.teacher_max_samples > 0:
            cmd += ["--max-samples", self.teacher_max_samples]
        self._cmd(cmd)
        print(json.dumps(self._read_json(self.teacher_cache / "_teacher_stats.json"), indent=2))
        self._gpu("after-teacher")
        self._finish("cache_teacher_outputs", started)

    def train_stage2(self) -> None:
        started = self._banner("Session 7 — Stage 2 knowledge distillation")
        if not self.run_teacher:
            print("[STAGE2] teacher disabled; Stage 2 skipped")
            self._finish("train_stage2", started)
            return
        if not self.state.stage1_checkpoint:
            raise RuntimeError("run train_stage1() first")
        before = set(self.runs.glob("*_stage2_kd_local"))
        self._cmd([
            sys.executable,
            "scripts/train.py",
            "--config", self.config_path,
            "--train-cache", self.cache_root / "train",
            "--val-cache", self.cache_root / "val",
            "--teacher-cache", self.teacher_cache,
            "--init-checkpoint", self.state.stage1_checkpoint,
            "--runs-root", self.runs,
            "--run-name", "stage2_kd_local",
        ])
        run = self._latest_new_run("*_stage2_kd_local", before)
        checkpoint = run / "checkpoints" / "best.pt"
        self.state.stage2_run = str(run)
        self.state.stage2_checkpoint = str(checkpoint)
        print(f"[STAGE2] run={run}")
        print(f"[STAGE2] best={checkpoint}")
        self._print_tail_metrics(run)
        self._save_state()
        self._gpu("after-stage2")
        self._finish("train_stage2", started)

    def _composite(self, metrics: dict[str, Any]) -> float:
        """Return the validation score declared in the active YAML config."""
        target = float(metrics.get("target_top1_accuracy", 0.0))
        no_target = metrics.get("no_target_accuracy", 0.0)
        no_target = 0.0 if no_target is None or no_target != no_target else float(no_target)
        weights = self.config.validation.get("composite", {})
        target_weight = float(weights.get("target_top1_accuracy", 0.70))
        no_target_weight = float(weights.get("no_target_accuracy", 0.30))
        return target_weight * target + no_target_weight * no_target

    def select_final_on_validation(self) -> Path:
        started = self._banner("Session 8 — validation model selection")
        if not self.state.stage1_checkpoint:
            raise RuntimeError("Stage 1 checkpoint missing")
        stage1 = self.evaluate_checkpoint(Path(self.state.stage1_checkpoint), "val", "stage1_select")
        s1 = self._composite(stage1)
        stage2 = None
        s2 = float("-inf")
        if self.state.stage2_checkpoint:
            stage2 = self.evaluate_checkpoint(Path(self.state.stage2_checkpoint), "val", "stage2_select")
            s2 = self._composite(stage2)
        final = Path(self.state.stage2_checkpoint) if self.state.stage2_checkpoint and s2 > s1 else Path(self.state.stage1_checkpoint)
        self.state.final_checkpoint = str(final)
        report = {
            "selection_split": "val",
            "stage1_score": s1,
            "stage1_metrics": stage1,
            "stage2_score": s2 if stage2 is not None else None,
            "stage2_metrics": stage2,
            "selected_checkpoint": str(final),
        }
        (self.artifacts / "model_selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        self._save_state()
        self._finish("select_final_on_validation", started)
        return final

    def final_test(self) -> dict[str, Any]:
        started = self._banner("Session 9 — held-out final test")
        if not self.state.final_checkpoint:
            raise RuntimeError("select_final_on_validation() first")
        report = self.evaluate_checkpoint(Path(self.state.final_checkpoint), "test", "final")
        self.state.final_test_report = str(self.artifacts / "final_test.json")
        self._save_state()
        self._finish("final_test", started)
        return report

    def load_runtime(self):
        started = self._banner("Session 10 — load final deployment runtime")
        if not self.state.final_checkpoint:
            raise RuntimeError("final checkpoint is not selected")
        gc.collect()
        torch.cuda.empty_cache()
        from rt_lcod.inference.runtime import RTLCODRuntime

        self.runtime = RTLCODRuntime(self.config, checkpoint=self.state.final_checkpoint, device="cuda")
        print(f"[RUNTIME] checkpoint={self.state.final_checkpoint}")
        print(f"[RUNTIME] detector={self.config.model.detector_name}")
        print(f"[RUNTIME] input_size={self.config.runtime.get('input_size')} top_k={self.config.runtime.get('top_k')}")
        self._gpu("runtime-loaded")
        self._finish("load_runtime", started)
        return self.runtime

    def show_video_ui(self, video_path: str = "data/demo.mp4", prompt: str = "person"):
        started = self._banner("Session 11 — interactive video + prompt UI")
        if self.runtime is None:
            self.load_runtime()
        from rt_lcod.inference.video_ui import NotebookVideoPlayer

        raw = Path(video_path)
        path = raw if raw.is_absolute() else self.root / raw
        player = NotebookVideoPlayer(
            runtime=self.runtime,
            video_path=str(path.resolve()),
            prompt=prompt,
            log_every=30,
            display_width=960,
        )
        player.show()
        print("[VIDEO] UI ready. Type your video path and prompt, then click Play.")
        print("[VIDEO] infer every=1 is the raw per-frame model path; higher values are a practical speed mode.")
        self._gpu("video-ui-ready")
        self._finish("show_video_ui", started)
        return player
