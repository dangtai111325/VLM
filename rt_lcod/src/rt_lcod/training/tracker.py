from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        if math.isnan(value):
            return
        self.total += float(value) * n
        self.count += int(n)

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)


@dataclass
class MetricTracker:
    meters: dict[str, AverageMeter] = field(default_factory=dict)

    def update(self, metrics: dict[str, float], n: int = 1) -> None:
        for key, value in metrics.items():
            self.meters.setdefault(key, AverageMeter()).update(float(value), n)

    def averages(self) -> dict[str, float]:
        return {key: meter.average for key, meter in self.meters.items()}
