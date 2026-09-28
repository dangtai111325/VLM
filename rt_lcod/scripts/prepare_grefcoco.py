from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
ATTRIBUTES = {
    "black", "white", "red", "green", "blue", "yellow", "orange", "purple", "pink",
    "brown", "gray", "grey", "small", "large", "big", "little", "tall", "short",
    "dark", "light",
}


def coco_image_url(image: dict[str, Any]) -> str:
    """Resolve the official COCO image URL for either the train or validation split."""
    file_name = str(image.get("file_name") or "")
    match = re.search(r"COCO_(train|val)2014_", file_name)
    if match:
        # gRefCOCO metadata often declares mscoco.org/images/<id>, which is a
        # redirect endpoint and can time out.  The filename deterministically
        # identifies the faster, direct COCO CDN object.
        return f"https://images.cocodataset.org/{match.group(1)}2014/{file_name}"
    declared = str(image.get("coco_url") or "").strip()
    if declared:
        return declared.replace("http://", "https://", 1)
    raise ValueError(f"cannot infer COCO split from file_name={file_name!r}")


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def extract_slots(
    sentence: str,
    target_class: str,
    category_names: list[str],
) -> tuple[str | None, str | None, str | None]:
    """Extract the locked V1 slots without pretending to parse unrestricted language.

    Target class is supplied by gRefCOCO/COCO annotation. Attribute is accepted only
    when a supported adjective immediately precedes the mentioned target class. A
    relation is accepted only if its tail explicitly mentions another COCO class.
    """
    text = sentence.lower().strip()
    attribute: str | None = None

    target_match = re.search(r"(?<!\w)" + re.escape(target_class.lower()) + r"(?!\w)", text)
    if target_match:
        prefix_tokens = re.findall(r"[\w'-]+", text[: target_match.start()])
        if prefix_tokens and prefix_tokens[-1] in ATTRIBUTES:
            attribute = prefix_tokens[-1]

    relation: str | None = None
    reference: str | None = None
    for surface, canonical in sorted(RELATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        match = re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", text)
        if not match:
            continue
        tail = text[match.end() :]
        candidates = [
            category
            for category in category_names
            if category != target_class and _contains_phrase(tail, category.lower())
        ]
        if candidates:
            reference = max(candidates, key=len)
            relation = canonical
            break
    return attribute, relation, reference


def _pick_sentence(ref: dict[str, Any], rng: random.Random) -> str | None:
    sentences = ref.get("sentences") or []
    if not sentences:
        return None
    values = [str(item.get("sent") or item.get("raw") or "").strip() for item in sentences]
    values = [value for value in values if value]
    return rng.choice(values) if values else None


def _download_metadata(raw_dir: Path) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DATA] metadata source: https://huggingface.co/datasets/{HF_REPO}")
    refs = Path(
        hf_hub_download(
            HF_REPO,
            "grefs(unc).json",
            repo_type="dataset",
            local_dir=raw_dir,
        )
    )
    instances = Path(
        hf_hub_download(
            HF_REPO,
            "instances.json",
            repo_type="dataset",
            local_dir=raw_dir,
        )
    )
    print(f"[DATA] refs={refs} ({refs.stat().st_size / 2**20:.1f} MiB)")
    print(f"[DATA] instances={instances} ({instances.stat().st_size / 2**20:.1f} MiB)")
    return refs, instances


def _build_pool(
    refs: list[dict[str, Any]],
    split_names: set[str],
    annotations: dict[int, dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    categories: dict[int, str],
    category_ids: dict[str, int],
    images: dict[int, dict[str, Any]],
    limit: int,
    seed: int,
    max_negative_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(seed)
    candidates = [ref for ref in refs if str(ref.get("split")) in split_names]
    rng.shuffle(candidates)
    category_names = sorted(set(categories.values()), key=len, reverse=True)
    records: list[dict[str, Any]] = []
    negative_count = 0
    build_stats = {
        "seen_refs": 0,
        "skipped_multi_target": 0,
        "skipped_negative_quota": 0,
        "skipped_missing_image": 0,
        "skipped_missing_sentence": 0,
        "skipped_missing_target_class": 0,
        "skipped_ambiguous_class_only": 0,
        "verified_reference_boxes": 0,
    }

    for ref in candidates:
        if len(records) >= limit:
            break
        build_stats["seen_refs"] += 1
        ann_ids = [int(value) for value in _listify(ref.get("ann_id")) if value is not None]
        valid_ann_ids = [ann_id for ann_id in ann_ids if ann_id >= 0 and ann_id in annotations]
        no_target = len(valid_ann_ids) == 0
        if not no_target and len(valid_ann_ids) != 1:
            build_stats["skipped_multi_target"] += 1
            continue
        if no_target and negative_count >= max(1, int(limit * max_negative_fraction)):
            build_stats["skipped_negative_quota"] += 1
            continue

        image_id = int(ref["image_id"])
        image = images.get(image_id)
        if not image:
            build_stats["skipped_missing_image"] += 1
            continue
        sentence = _pick_sentence(ref, rng)
        if not sentence:
            build_stats["skipped_missing_sentence"] += 1
            continue

        target_ann: dict[str, Any] | None = None
        target_category_id: int | None = None
        if no_target:
            cat_ids = [
                int(value)
                for value in _listify(ref.get("category_id"))
                if str(value).lstrip("-").isdigit()
            ]
            valid_classes = [categories[value] for value in cat_ids if value in categories]
            if valid_classes:
                target_class = valid_classes[0]
                target_category_id = category_ids.get(target_class)
            else:
                mentioned = [
                    category
                    for category in category_names
                    if _contains_phrase(sentence.lower(), category.lower())
                ]
                if not mentioned:
                    build_stats["skipped_missing_target_class"] += 1
                    continue
                target_class = mentioned[0]
                target_category_id = category_ids.get(target_class)
            target_boxes: list[list[float]] = []
        else:
            target_ann = annotations[valid_ann_ids[0]]
            target_category_id = int(target_ann["category_id"])
            target_class = categories.get(target_category_id)
            if not target_class:
                build_stats["skipped_missing_target_class"] += 1
                continue
            target_boxes = [_xywh_to_xyxy(target_ann["bbox"])]

        attribute, relation, reference_class = extract_slots(
            sentence,
            target_class,
            category_names,
        )
        image_annotations = annotations_by_image.get(image_id, [])

        # If several objects of the target class are present but none of the V1 language
        # slots can disambiguate them, the student has no learnable input for that choice.
        # Skip such impossible supervision instead of injecting label noise.
        if not no_target and target_category_id is not None:
            same_class = [
                ann
                for ann in image_annotations
                if int(ann.get("category_id", -1)) == target_category_id
            ]
            if len(same_class) > 1 and attribute is None and relation is None:
                build_stats["skipped_ambiguous_class_only"] += 1
                continue

        reference_boxes: list[list[float]] = []
        if relation and reference_class:
            reference_category_id = category_ids.get(reference_class)
            if reference_category_id is not None:
                reference_annotations = [
                    ann
                    for ann in image_annotations
                    if int(ann.get("category_id", -1)) == reference_category_id
                    and (target_ann is None or int(ann["id"]) != int(target_ann["id"]))
                ]
                # Direct pairwise relation BCE is enabled only for an unambiguous single
                # reference instance. Multiple references remain usable through L_target.
                if len(reference_annotations) == 1:
                    reference_boxes = [_xywh_to_xyxy(reference_annotations[0]["bbox"])]
                    build_stats["verified_reference_boxes"] += 1

        if no_target:
            negative_count += 1

        records.append(
            {
                "id": f"gref_{ref['ref_id']}_{len(records)}",
                "image": f"images/train2014/{image['file_name']}",
                "coco_url": coco_image_url(image),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "prompt": sentence,
                "slots": {
                    "target_class": target_class,
                    "attribute": attribute,
                    "relation": relation,
                    "reference_class": reference_class,
                },
                "target_boxes": target_boxes,
                "reference_boxes": reference_boxes,
                "no_target": no_target,
                "source": "gRefCOCO",
            }
        )
    return records, build_stats


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _download_one(
    image_dir: Path,
    file_name: str,
    source_url: str,
    allow_http_fallback: bool = False,
    timeout: float = 30.0,
) -> tuple[str, str]:
    output = image_dir / file_name
    if output.exists() and output.stat().st_size > 1024:
        return file_name, "cached"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                source_url,
                timeout=timeout,
                headers={"User-Agent": "rt-lcod-local-data-prep/0.2"},
            )
            response.raise_for_status()
            temporary.write_bytes(response.content)
            if temporary.stat().st_size <= 1024:
                raise RuntimeError("downloaded file is unexpectedly small")
            temporary.replace(output)
            return file_name, "downloaded"
        except requests.exceptions.SSLError as exc:
            last_error = exc
            fallback_url = source_url.replace("https://", "http://", 1)
            if not allow_http_fallback or fallback_url == source_url:
                temporary.unlink(missing_ok=True)
                time.sleep(1.5 * (attempt + 1))
                continue
            try:
                response = requests.get(
                    fallback_url,
                    timeout=timeout,
                    headers={"User-Agent": "rt-lcod-local-data-prep/0.2"},
                )
                response.raise_for_status()
                temporary.write_bytes(response.content)
                if temporary.stat().st_size <= 1024:
                    raise RuntimeError("fallback download is unexpectedly small")
                temporary.replace(output)
                return file_name, "downloaded_http_fallback"
            except Exception as fallback_exc:
                last_error = fallback_exc
                temporary.unlink(missing_ok=True)
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            time.sleep(1.5 * (attempt + 1))
    return file_name, f"failed:url={source_url}: {type(last_error).__name__}:{last_error}"


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = Counter(row["slots"]["target_class"] for row in rows)
    return {
        "samples": len(rows),
        "positive": sum(not row["no_target"] for row in rows),
        "no_target": sum(row["no_target"] for row in rows),
        "with_attribute": sum(bool(row["slots"]["attribute"]) for row in rows),
        "with_relation": sum(bool(row["slots"]["relation"]) for row in rows),
        "with_verified_reference_box": sum(bool(row["reference_boxes"]) for row in rows),
        "unique_images": len({row["image"] for row in rows}),
        "top_categories": category_counts.most_common(15),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a local RTX-sized, V1-compatible gRefCOCO subset for RT-LCOD."
    )
    parser.add_argument("--root", default="data")
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--val", type=int, default=300)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--negative-fraction", type=float, default=0.30)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--allow-http-fallback",
        action="store_true",
        help="use public HTTP COCO URLs only when HTTPS fails certificate validation",
    )
    parser.add_argument(
        "--prefer-http-coco",
        action="store_true",
        help="use public HTTP COCO URLs directly (only for networks with a known HTTPS certificate mismatch)",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    root = Path(args.root).resolve()
    raw_dir = root / "raw" / "grefcoco"
    refs_path, instances_path = _download_metadata(raw_dir)
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    instances = json.loads(instances_path.read_text(encoding="utf-8"))

    annotations = {int(item["id"]): item for item in instances["annotations"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in instances["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    categories = {int(item["id"]): str(item["name"]) for item in instances["categories"]}
    category_ids = {name: category_id for category_id, name in categories.items()}
    images = {int(item["id"]): item for item in instances["images"]}
    print(
        f"[DATA] refs={len(refs):,} annotations={len(annotations):,} "
        f"images={len(images):,} categories={len(categories)}"
    )

    # gRefCOCO's official ``val`` split is almost entirely no-target prompts.  It is
    # valuable for rejection evaluation, but cannot serve as the only validation
    # signal for target grounding.  Build one shuffled, V1-compatible pool from the
    # official train refs, then split it once into disjoint train/validation records.
    # The held-out test remains strictly from official testA/testB refs.
    train_val_total = args.train + args.val
    train_val_rows, train_val_build = _build_pool(
        refs,
        {"train"},
        annotations,
        annotations_by_image,
        categories,
        category_ids,
        images,
        train_val_total,
        args.seed,
        args.negative_fraction,
    )
    if len(train_val_rows) < train_val_total:
        raise RuntimeError(
            f"train/validation pool requested={train_val_total} produced={len(train_val_rows)}"
        )

    test_rows, test_build = _build_pool(
        refs,
        {"testA", "testB"},
        annotations,
        annotations_by_image,
        categories,
        category_ids,
        images,
        args.test,
        args.seed + 2,
        args.negative_fraction,
    )
    if len(test_rows) < args.test:
        raise RuntimeError(f"test pool requested={args.test} produced={len(test_rows)}")

    # Stratify the local validation split explicitly.  Taking a contiguous slice
    # after pool construction can concentrate all no-target rows in train because
    # the pool's negative quota is reached early.
    split_rng = random.Random(args.seed + 1)
    positives = [row for row in train_val_rows if not row["no_target"]]
    negatives = [row for row in train_val_rows if row["no_target"]]
    split_rng.shuffle(positives)
    split_rng.shuffle(negatives)
    train_negative_count = int(args.train * args.negative_fraction)
    val_negative_count = int(args.val * args.negative_fraction)
    train_positive_count = args.train - train_negative_count
    val_positive_count = args.val - val_negative_count
    if len(positives) < train_positive_count + val_positive_count:
        raise RuntimeError("insufficient positive rows for the requested train/validation split")
    if len(negatives) < train_negative_count + val_negative_count:
        raise RuntimeError("insufficient no-target rows for the requested train/validation split")
    train_rows = (
        positives[:train_positive_count]
        + negatives[:train_negative_count]
    )
    val_rows = (
        positives[train_positive_count : train_positive_count + val_positive_count]
        + negatives[train_negative_count : train_negative_count + val_negative_count]
    )
    split_rng.shuffle(train_rows)
    split_rng.shuffle(val_rows)
    splits: dict[str, list[dict[str, Any]]] = {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
    }
    build_reports: dict[str, dict[str, Any]] = {
        "train_val_source": train_val_build,
        "test_source": test_build,
    }
    for split, rows in splits.items():
        manifest = root / "manifests" / f"{split}.jsonl"
        _write_jsonl(manifest, rows)
        print(f"[DATA] {split}: {json.dumps(_stats(rows), ensure_ascii=False)}")
        print(f"[DATA] manifest={manifest}")
    print(f"[DATA] train/val source build: {json.dumps(train_val_build, ensure_ascii=False)}")
    print(f"[DATA] test source build: {json.dumps(test_build, ensure_ascii=False)}")

    image_urls = {
        Path(row["image"]).name: (
            str(row["coco_url"]).replace("https://", "http://", 1)
            if args.prefer_http_coco
            else str(row["coco_url"])
        )
        for rows in splits.values()
        for row in rows
    }
    files = sorted(image_urls.items())
    image_dir = root / "images" / "train2014"
    downloaded = cached = http_fallback = failed = 0
    print(
        f"[DATA] downloading/checking {len(files):,} unique COCO images "
        f"with {args.workers} workers"
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                _download_one,
                image_dir,
                name,
                source_url,
                args.allow_http_fallback,
            )
            for name, source_url in files
        ]
        for index, future in enumerate(as_completed(futures), 1):
            name, state = future.result()
            if state == "downloaded":
                downloaded += 1
            elif state == "downloaded_http_fallback":
                downloaded += 1
                http_fallback += 1
            elif state == "cached":
                cached += 1
            else:
                failed += 1
                print(f"[DATA][WARN] {name}: {state}")
            if index % 100 == 0 or index == len(futures):
                print(
                    f"[DATA] images {index}/{len(futures)} downloaded={downloaded} "
                    f"cached={cached} http_fallback={http_fallback} failed={failed}"
                )
    if failed:
        raise RuntimeError(f"{failed} image downloads failed; rerun the cell to resume")

    report = {
        "source": HF_REPO,
        "seed": args.seed,
        "requested": {"train": args.train, "val": args.val, "test": args.test},
        "splits": {name: _stats(rows) for name, rows in splits.items()},
        "build_reports": build_reports,
        "images": {
            "unique": len(files),
            "downloaded": downloaded,
            "cached": cached,
            "http_fallback": http_fallback,
            "failed": failed,
        },
        "elapsed_s": time.perf_counter() - started,
    }
    report_path = root / "data_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[DATA] complete in {report['elapsed_s']:.1f}s; report={report_path}")


if __name__ == "__main__":
    main()
