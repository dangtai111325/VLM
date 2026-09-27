from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
from typing import Any

import torch


class RunLogger:
    def __init__(self, runs_root: str | Path, run_name: str):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name)
        self.run_dir = Path(runs_root) / f"{stamp}_{safe_name}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_jsonl = self.run_dir / "metrics.jsonl"
        self.metrics_csv = self.run_dir / "metrics.csv"
        self._csv_header_written = False

    def write_json(self, name: str, payload: Any) -> None:
        if is_dataclass(payload):
            payload = asdict(payload)
        (self.run_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    def log_metrics(self, payload: dict[str, Any]) -> None:
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **payload}
        with self.metrics_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        flat = {k: v for k, v in record.items() if isinstance(v, (str, int, float, bool)) or v is None}
        with self.metrics_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(flat)


def environment_snapshot() -> dict[str, Any]:
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        git_commit = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_commit,
        "pid": os.getpid(),
    }
