from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time


@dataclass
class TimerBook:
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.totals[name] = self.totals.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            key: {"total_s": total, "count": self.counts[key], "mean_ms": 1000.0 * total / max(self.counts[key], 1)}
            for key, total in self.totals.items()
        }
