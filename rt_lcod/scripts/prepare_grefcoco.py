from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import re
import time
from typing import Any

from huggingface_hub import hf_hub_download
import requests


HF_REPO = "FudanCVL/gRefCOCO"
RELATIONS = {
    "to the left of": "left_of",
    "left of": "left_of",
    "to the right of": "right_of",
    "right of": "right_of",
    "next to": "next_to",
    "beside": "next_to",
    "near": "near",
    "in front of": "in_front_of",
    "behind": "behind",
    "above": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "inside": "inside",
    "overlapping": "overlapping",
}
ATTRIBUTES = [
    "black", "white", "red", "green", "blue", "yellow", "orange", "purple", "pink",
    "brown", "gray", "grey", "small", "large", "big", "little", "tall", "short",
]


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def extract_slots(sentence: str, target_class: str, category_names: list[str]) -> tuple[str | None, str | None, str | None]:
    """Extract at most one attribute and one one-hop spatial relation.

    The target class comes from gRefCOCO/COCO annotations, not from this heuristic.
    This keeps training labels stable while still respecting the V1 prompt contract.
    """
    text = sentence.lower().strip()
    attribute = next((a for a in ATTRIBUTES if _contains_phrase(text, a)), None)
    relation = None
    reference = None
    for surface, canonical in sorted(RELATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        match = re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", text)
        if not match:
            continue
        tail = text[match.end():]
        candidates = [c for c in category_names if c != target_class and _contains_phrase(tail, c.lower())]
        if candidates:
            reference = max(candidates, key=len)
            relation = canonical
            break
    return attribute, relation, reference


def _pick_sentence(ref: dict[str, Any], rng: random.Random) -> str | None:
    sentences = ref.get("sentences") or []
    if not sentences:
        return None
    values = [str(s.get("sent") or s.get("raw") or "").strip() for s in sentences]
    values = [v for v in values if v]
    return rng.choice(values) if values else None


def _download_metadata(raw_dir: Path) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DATA] metadata source: https://huggingface.co/datasets/{HF_REPO}")
    refs = Path(hf_hub_download(HF_REPO, "grefs(unc).json", repo_type="dataset", local_dir=raw_dir))
    instances = Path(hf_hub_download(HF_REPO, "instances.json", repo_type="dataset", local_dir=raw_dir))
    print(f"[DATA] refs={refs} ({refs.stat().st_size / 2**20:.1f} MiB)")
    print(f"[DATA] instances={instances} ({instances.stat().st_size / 2**20:.1f} MiB)")
    return refs, instances


def _build_pool(
    refs: list[dict[str, Any]],
    split_names: set[str],
    annotations: dict[int, dict[str, Any]],
    categories: dict[int, str],
    images: dict[int, dict[str, Any]],
    limit: int,
    seed: int,
    max_negative_fraction: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates = [r for r in refs if str(r.get("split")) in split_names]
    rng.shuffle(candidates)
    category_names = sorted(set(categories.values()), key=len, reverse=True)
    records: list[dict[str, Any]] = []
    negative_count = 0

    for ref in candidates:
        if len(records) >= limit:
            break
        ann_ids = [int(x) for x in _listify(ref.get("ann_id")) if x is not None]
        valid_ann_ids = [x for x in ann_ids if x >= 0 and x in annotations]
        no_target = len(valid_ann_ids) == 0
        if not no_target and len(valid_ann_ids) != 1:
            # V1 is single-target only.
            continue
        if no_target and negative_count >= max(1, int(limit * max_negative_fraction)):
            continue

        image_id = int(ref["image_id"])
        image = images.get(image_id)
        if not image:
            continue
        sentence = _pick_sentence(ref, rng)
        if not sentence:
            continue

        if no_target:
            cat_ids = [int(x) for x in _listify(ref.get("category_id")) if str(x).lstrip("-").isdigit()]
            valid_cats = [categories[x] for x in cat_ids if x in categories]
            if valid_cats:
                target_class = valid_cats[0]
            else:
                mentioned = [c for c in category_names if _contains_phrase(sentence.lower(), c.lower())]
                if not mentioned:
                    continue
                target_class = mentioned[0]
            target_boxes: list[list[float]] = []
            negative_count += 1
        else:
            ann = annotations[valid_ann_ids[0]]
            target_class = categories.get(int(ann["category_id"]))
            if not target_class:
                continue
            target_boxes = [_xywh_to_xyxy(ann["bbox"])]

        attribute, relation, reference_class = extract_slots(sentence, target_class, category_names)
        width, height = int(image["width"]), int(image["height"])
        file_name = str(image["file_name"])
        records.append({
            "id": f"gref_{ref['ref_id']}_{len(records)}",
            "image": f"images/train2014/{file_name}",
            "width": width,
            "height": height,
            "prompt": sentence,
            "slots": {
                "target_class": target_class,
                "attribute": attribute,
                "relation": relation,
                "reference_class": reference_class,
            },
            "target_boxes": target_boxes,
            "reference_boxes": [],
            "no_target": no_target,
            "source": "gRefCOCO",
        })
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _download_one(image_dir: Path, file_name: str, timeout: float = 30.0) -> tuple[str, str]:
    output = image_dir / file_name
    if output.exists() and output.stat().st_size > 1024:
        return file_name, "cached"
    output.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://images.cocodataset.org/train2014/{file_name}"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            output.write_bytes(response.content)
            if output.stat().st_size <= 1024:
                raise RuntimeError("downloaded file is unexpectedly small")
            return file_name, "downloaded"
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    return file_name, f"failed:{type(last_error).__name__}:{last_error}"


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(r["slots"]["target_class"] for r in rows)
    return {
        "samples": len(rows),
        "positive": sum(not r["no_target"] for r in rows),
        "no_target": sum(r["no_target"] for r in rows),
        "with_attribute": sum(bool(r["slots"]["attribute"]) for r in rows),
        "with_relation": sum(bool(r["slots"]["relation"]) for r in rows),
        "unique_images": len({r["image"] for r in rows}),
        "top_categories": categories.most_common(15),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a T4-sized gRefCOCO subset for RT-LCOD.")
    parser.add_argument("--root", default="data")
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--val", type=int, default=300)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--negative-fraction", type=float, default=0.30)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    started = time.perf_counter()
    root = Path(args.root).resolve()
    raw_dir = root / "raw" / "grefcoco"
    refs_path, instances_path = _download_metadata(raw_dir)
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    annotations = {int(a["id"]): a for a in instances["annotations"]}
    categories = {int(c["id"]): str(c["name"]) for c in instances["categories"]}
    images = {int(i["id"]): i for i in instances["images"]}
    print(f"[DATA] refs={len(refs):,} annotations={len(annotations):,} images={len(images):,} categories={len(categories)}")

    splits = {
        "train": _build_pool(refs, {"train"}, annotations, categories, images, args.train, args.seed, args.negative_fraction),
        "val": _build_pool(refs, {"val"}, annotations, categories, images, args.val, args.seed + 1, args.negative_fraction),
        "test": _build_pool(refs, {"testA", "testB"}, annotations, categories, images, args.test, args.seed + 2, args.negative_fraction),
    }
    for name, rows in splits.items():
        if not rows:
            raise RuntimeError(f"no usable rows produced for split={name}")
        manifest = root / "manifests" / f"{name}.jsonl"
        _write_jsonl(manifest, rows)
        print(f"[DATA] {name}: {json.dumps(_stats(rows), ensure_ascii=False)}")
        print(f"[DATA] manifest={manifest}")

    files = sorted({Path(r["image"]).name for rows in splits.values() for r in rows})
    image_dir = root / "images" / "train2014"
    downloaded = cached = failed = 0
    print(f"[DATA] downloading/checking {len(files):,} unique COCO images with {args.workers} workers")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_download_one, image_dir, name) for name in files]
        for index, future in enumerate(as_completed(futures), 1):
            name, state = future.result()
            if state == "downloaded":
                downloaded += 1
            elif state == "cached":
                cached += 1
            else:
                failed += 1
                print(f"[DATA][WARN] {name}: {state}")
            if index % 100 == 0 or index == len(futures):
                print(f"[DATA] images {index}/{len(futures)} downloaded={downloaded} cached={cached} failed={failed}")
    if failed:
        raise RuntimeError(f"{failed} image downloads failed; rerun the cell to resume")

    report = {
        "source": HF_REPO,
        "seed": args.seed,
        "requested": {"train": args.train, "val": args.val, "test": args.test},
        "splits": {k: _stats(v) for k, v in splits.items()},
        "images": {"unique": len(files), "downloaded": downloaded, "cached": cached, "failed": failed},
        "elapsed_s": time.perf_counter() - started,
    }
    report_path = root / "data_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[DATA] complete in {report['elapsed_s']:.1f}s; report={report_path}")


if __name__ == "__main__":
    main()
