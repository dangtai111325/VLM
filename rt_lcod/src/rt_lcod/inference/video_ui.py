from __future__ import annotations

from datetime import datetime
import threading
import time
from pathlib import Path

import cv2
import ipywidgets as widgets
import numpy as np
import torch
from IPython.display import display

from rt_lcod.inference.runtime import RTLCODRuntime, RuntimeResult


class NotebookVideoPlayer:
    """Interactive video player for a trained RT-LCOD checkpoint."""

    def __init__(
        self,
        runtime: RTLCODRuntime,
        video_path: str = "data/demo.mp4",
        prompt: str = "person",
        log_every: int = 30,
        display_width: int = 960,
        log_path: str | Path | None = None,
    ) -> None:
        self.runtime = runtime
        self.log_every = max(1, int(log_every))
        self.display_width = int(display_width)
        self.log_path = Path(log_path).resolve() if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_result: RuntimeResult | None = None

        self.video = widgets.Text(
            value=video_path,
            description="Video:",
            layout=widgets.Layout(width="78%"),
        )
        self.prompt = widgets.Text(
            value=prompt,
            description="Prompt:",
            layout=widgets.Layout(width="78%"),
        )
        self.conf = widgets.FloatSlider(
            value=float(runtime.config.runtime.get("confidence", 0.15)),
            min=0.02,
            max=0.60,
            step=0.01,
            description="YOLOE conf:",
            continuous_update=False,
            layout=widgets.Layout(width="62%"),
        )
        self.stride = widgets.IntSlider(
            value=1,
            min=1,
            max=5,
            step=1,
            description="Infer every:",
            continuous_update=False,
            layout=widgets.Layout(width="62%"),
        )
        self.realtime = widgets.Checkbox(value=True, description="play at source FPS")
        self.play_button = widgets.Button(description="▶ Play", button_style="success")
        self.pause_button = widgets.Button(description="⏸ Pause")
        self.stop_button = widgets.Button(description="■ Stop", button_style="danger")
        self.image = widgets.Image(format="jpeg")
        self.status = widgets.HTML(value="<b>Ready.</b>")
        self.log = widgets.Output(
            layout=widgets.Layout(border="1px solid #ddd", max_height="260px", overflow="auto")
        )

        self.play_button.on_click(self._on_play)
        self.pause_button.on_click(self._on_pause)
        self.stop_button.on_click(self._on_stop)

    def show(self) -> "NotebookVideoPlayer":
        log_note = f"<code>{self.log_path}</code>" if self.log_path else "not persisted"
        display(
            widgets.VBox(
                [
                    widgets.HTML("<h3>RT-LCOD Video Inference</h3>"),
                    self.video,
                    self.prompt,
                    widgets.HBox([self.conf, self.stride, self.realtime]),
                    widgets.HBox([self.play_button, self.pause_button, self.stop_button]),
                    self.status,
                    self.image,
                    widgets.HTML(f"<b>Runtime log</b> — file: {log_note}"),
                    self.log,
                ]
            )
        )
        return self

    def _print(self, text: str) -> None:
        timestamped = f"{datetime.now().isoformat(timespec='seconds')} {text}"
        with self.log:
            print(timestamped, flush=True)
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(timestamped + "\n")

    def _on_play(self, _button) -> None:
        if self._thread and self._thread.is_alive():
            self._pause.clear()
            self.status.value = "<b>Playing.</b>"
            return
        self._stop.clear()
        self._pause.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _on_pause(self, _button) -> None:
        self._pause.set()
        self.status.value = "<b>Paused.</b>"

    def _on_stop(self, _button) -> None:
        self._stop.set()
        self._pause.clear()
        self.status.value = "<b>Stopping…</b>"

    @staticmethod
    def _draw(frame: np.ndarray, result: RuntimeResult | None, prompt: str) -> np.ndarray:
        canvas = frame.copy()
        if result is not None and result.box is not None:
            x1, y1, x2, y2 = map(int, result.box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"TARGET {result.confidence:.2f}"
            cv2.putText(
                canvas,
                label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        elif result is not None:
            cv2.putText(
                canvas,
                "NO_TARGET",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
        cv2.putText(
            canvas,
            f"Prompt: {prompt}",
            (20, canvas.shape[0] - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        return canvas

    def _publish_frame(
        self,
        frame: np.ndarray,
        result: RuntimeResult | None,
        prompt: str,
    ) -> None:
        canvas = self._draw(frame, result, prompt)
        if canvas.shape[1] > self.display_width:
            scale = self.display_width / canvas.shape[1]
            canvas = cv2.resize(
                canvas,
                (self.display_width, int(canvas.shape[0] * scale)),
            )
        ok, encoded = cv2.imencode(
            ".jpg",
            canvas,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        if ok:
            self.image.value = encoded.tobytes()

    def _run(self) -> None:
        path = Path(self.video.value).expanduser().resolve()
        if not path.exists():
            self.status.value = f"<b style='color:red'>Video not found:</b> {path}"
            self._print(f"[VIDEO][ERROR] missing video: {path}")
            return
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            self.status.value = f"<b style='color:red'>Cannot open video:</b> {path}"
            self._print(f"[VIDEO][ERROR] cv2 cannot open: {path}")
            return

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._print(
            f"[VIDEO] start path={path} resolution={width}x{height} "
            f"source_fps={source_fps:.3f} frames={total_frames} stride={self.stride.value} "
            f"yolo_conf={self.conf.value:.2f}"
        )
        self.status.value = "<b>Playing.</b>"

        frame_index = 0
        inference_count = 0
        latencies: list[float] = []
        detector_latencies: list[float] = []
        region_latencies: list[float] = []
        head_latencies: list[float] = []
        prompt_change_latencies: list[float] = []
        selected = 0
        no_target = 0
        wall_started = time.perf_counter()

        try:
            while not self._stop.is_set():
                if self._pause.is_set():
                    time.sleep(0.05)
                    continue

                cycle_started = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                prompt = self.prompt.value.strip() or "person"
                stride = max(1, int(self.stride.value))

                if frame_index == 1 or frame_index % stride == 0:
                    self.runtime.config.runtime["confidence"] = float(self.conf.value)
                    infer_started = time.perf_counter()
                    try:
                        self._last_result = self.runtime.process_frame(frame, prompt=prompt)
                    except Exception as exc:
                        self._print(
                            f"[VIDEO][ERROR] frame={frame_index} {type(exc).__name__}: {exc}"
                        )
                        self.status.value = f"<b style='color:red'>Inference failed:</b> {exc}"
                        break

                    latency_ms = (time.perf_counter() - infer_started) * 1000.0
                    inference_count += 1
                    latencies.append(latency_ms)
                    detector_latencies.append(
                        self._last_result.timings_ms.get("detector", 0.0)
                    )
                    region_latencies.append(
                        self._last_result.timings_ms.get("region_encoder", 0.0)
                    )
                    head_latencies.append(
                        self._last_result.timings_ms.get("grounding_head", 0.0)
                    )
                    if "prompt_change" in self._last_result.timings_ms:
                        prompt_change_latencies.append(
                            self._last_result.timings_ms["prompt_change"]
                        )
                        self._print(
                            f"[PROMPT] frame={frame_index} prompt={prompt!r} "
                            f"change_ms={self._last_result.timings_ms['prompt_change']:.2f}"
                        )

                    if self._last_result.box is None:
                        no_target += 1
                    else:
                        selected += 1

                    if inference_count % self.log_every == 0:
                        gpu_allocated = (
                            torch.cuda.memory_allocated() / 2**20
                            if torch.cuda.is_available()
                            else 0.0
                        )
                        gpu_reserved = (
                            torch.cuda.memory_reserved() / 2**20
                            if torch.cuda.is_available()
                            else 0.0
                        )
                        self._print(
                            f"[VIDEO] frame={frame_index}/{total_frames or '?'} "
                            f"infer={inference_count} latency_ms={latency_ms:.2f} "
                            f"detector_ms={detector_latencies[-1]:.2f} "
                            f"region_ms={region_latencies[-1]:.2f} "
                            f"head_ms={head_latencies[-1]:.2f} "
                            f"candidates={self._last_result.candidate_count} "
                            f"result={'NO_TARGET' if self._last_result.box is None else 'BOX'} "
                            f"confidence={self._last_result.confidence:.3f} "
                            f"gpu_alloc_mib={gpu_allocated:.1f} gpu_reserved_mib={gpu_reserved:.1f}"
                        )

                self._publish_frame(frame, self._last_result, prompt)
                self.status.value = (
                    f"<b>Playing</b> — frame {frame_index}/{total_frames or '?'} — "
                    f"inferences {inference_count} — prompt: <code>{prompt}</code>"
                )
                if self.realtime.value:
                    remaining = (1.0 / source_fps) - (time.perf_counter() - cycle_started)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            capture.release()
            elapsed = time.perf_counter() - wall_started
            if latencies:
                latency_array = np.asarray(latencies, dtype=float)
                detector_array = np.asarray(detector_latencies, dtype=float)
                region_array = np.asarray(region_latencies, dtype=float)
                head_array = np.asarray(head_latencies, dtype=float)
                summary = (
                    f"[VIDEO] done frames={frame_index} inferences={inference_count} "
                    f"elapsed_s={elapsed:.2f} mean_ms={latency_array.mean():.2f} "
                    f"p50_ms={np.percentile(latency_array, 50):.2f} "
                    f"p95_ms={np.percentile(latency_array, 95):.2f} "
                    f"p99_ms={np.percentile(latency_array, 99):.2f} "
                    f"processing_fps={1000.0 / latency_array.mean():.2f} "
                    f"detector_mean_ms={detector_array.mean():.2f} "
                    f"region_mean_ms={region_array.mean():.2f} "
                    f"head_mean_ms={head_array.mean():.2f} "
                    f"selected={selected} no_target={no_target} "
                    f"prompt_changes={len(prompt_change_latencies)}"
                )
                self._print(summary)
                self.status.value = "<b>Finished.</b> " + summary.replace("[VIDEO] ", "")
            else:
                self.status.value = "<b>Stopped.</b>"
