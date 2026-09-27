from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    image: str
    width: int
    height: int
    prompt: str
    target_class: str
    attribute: str | None
    relation: str | None
    reference_class: str | None
    target_boxes: list[list[float]]
    reference_boxes: list[list[float]]
    no_target: bool
    source: str = "unknown"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ManifestSample":
        slots = raw.get("slots", {})
        sample = cls(
            sample_id=str(raw["id"]), image=str(raw["image"]), width=int(raw["width"]), height=int(raw["height"]),
            prompt=str(raw["prompt"]), target_class=str(slots["target_class"]),
            attribute=slots.get("attribute") or None, relation=slots.get("relation") or None,
            reference_class=slots.get("reference_class") or None,
            target_boxes=[list(map(float, b)) for b in raw.get("target_boxes", [])],
            reference_boxes=[list(map(float, b)) for b in raw.get("reference_boxes", [])],
            no_target=bool(raw.get("no_target", False)), source=str(raw.get("source", "unknown")),
        )
        sample.validate()
        return sample

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{self.sample_id}: invalid image size")
        if self.relation and not self.reference_class:
            raise ValueError(f"{self.sample_id}: relation requires reference_class")
        if self.no_target and self.target_boxes:
            raise ValueError(f"{self.sample_id}: no_target sample cannot contain target_boxes")
        if (not self.no_target) and not self.target_boxes:
            raise ValueError(f"{self.sample_id}: positive sample requires target_boxes")
        for name, boxes in (("target", self.target_boxes), ("reference", self.reference_boxes)):
            for box in boxes:
                if len(box) != 4:
                    raise ValueError(f"{self.sample_id}: {name} box must contain 4 values")
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"{self.sample_id}: invalid {name} box {box}")


def load_manifest(path: str | Path) -> list[ManifestSample]:
    path = Path(path)
    samples: list[ManifestSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                samples.append(ManifestSample.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not samples:
        raise ValueError(f"manifest is empty: {path}")
    ids = [s.sample_id for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"manifest contains duplicate sample ids: {path}")
    return samples
