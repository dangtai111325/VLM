from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Iterable

from huggingface_hub import snapshot_download
import pandas as pd
from PIL import Image
import requests
from tqdm.auto import tqdm

from .config import Config, Paths
from .utils import RunState, atomic_json_dump, stage_timer


def xywh_to_xyxy(box: Iterable[float]) -> list[float]:
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def download_annotations(cfg: Config, paths: Paths, state: RunState) -> tuple[Path, Path]:
    with stage_timer("Session 2A — Tải annotation gRefCOCO"):
        local = Path(
            snapshot_download(
                repo_id=cfg.hf_dataset_repo,
                repo_type="dataset",
                allow_patterns=[cfg.split_file, cfg.instances_file],
                local_dir=paths.annotation,
            )
        )
        refs_path = local / cfg.split_file
        instances_path = local / cfg.instances_file
        if not refs_path.exists() or not instances_path.exists():
            raise FileNotFoundError(
                "Không tìm thấy gRefCOCO annotation sau khi tải. "
                f"Expected: {refs_path} và {instances_path}"
            )
        state.mark("annotations_downloaded", refs=str(refs_path), instances=str(instances_path))
        return refs_path, instances_path


def build_manifest(
    cfg: Config,
    paths: Paths,
    state: RunState,
    refs_path: Path,
    instances_path: Path,
) -> tuple[pd.DataFrame, Path, Path]:
    manifest_path = paths.data / f"manifest_{cfg.signature()}.jsonl"
    image_meta_path = paths.data / f"image_meta_{cfg.signature()}.json"

    if manifest_path.exists() and image_meta_path.exists():
        print(f"✓ Dùng lại manifest: {manifest_path}")
        return pd.read_json(manifest_path, lines=True), manifest_path, image_meta_path

    with stage_timer("Session 2B — Chuẩn hóa annotation thành DOD manifest"):
        refs = json.loads(refs_path.read_text(encoding="utf-8"))
        instances = json.loads(instances_path.read_text(encoding="utf-8"))

        anns = {int(a["id"]): a for a in instances["annotations"]}
        images = {int(x["id"]): x for x in instances["images"]}
        cats = {int(c["id"]): c["name"] for c in instances["categories"]}

        rows: list[dict[str, Any]] = []
        for ref in tqdm(refs, desc="Chuẩn hóa refs", unit="ref"):
            image_id = int(ref["image_id"])
            image_meta = images[image_id]

            ann_ids = ref.get("ann_id", [])
            if not isinstance(ann_ids, list):
                ann_ids = [ann_ids]
            valid_anns = [
                anns[int(ann_id)]
                for ann_id in ann_ids
                if int(ann_id) != -1 and int(ann_id) in anns
            ]
            gt_boxes = [xywh_to_xyxy(a["bbox"]) for a in valid_anns]

            category_ids = ref.get("category_id", [])
            if not isinstance(category_ids, list):
                category_ids = [category_ids]
            target_categories = [
                cats[int(category_id)]
                for category_id in category_ids
                if int(category_id) in cats
            ]

            for sentence in ref.get("sentences", []):
                query = str(sentence.get("sent", "")).strip()
                if not query:
                    continue
                rows.append(
                    {
                        "sample_id": f'{ref["ref_id"]}_{sentence["sent_id"]}',
                        "ref_id": int(ref["ref_id"]),
                        "sent_id": int(sentence["sent_id"]),
                        "split": str(ref["split"]),
                        "image_id": image_id,
                        "file_name": image_meta["file_name"],
                        "coco_url": image_meta.get(
                            "coco_url",
                            f'https://images.cocodataset.org/train2014/{image_meta["file_name"]}',
                        ),
                        "width": int(image_meta["width"]),
                        "height": int(image_meta["height"]),
                        "query": query,
                        "gt_boxes": gt_boxes,
                        "target_categories": target_categories,
                        "no_target": len(gt_boxes) == 0,
                    }
                )

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError("Manifest rỗng; schema gRefCOCO có thể đã thay đổi.")

        limits = {
            "train": cfg.max_train_samples,
            "val": cfg.max_val_samples,
            "test": cfg.max_test_samples,
            "testA": cfg.max_test_samples,
            "testB": cfg.max_test_samples,
        }
        parts: list[pd.DataFrame] = []
        for split_name, part in frame.groupby("split", sort=False):
            limit = limits.get(split_name)
            parts.append(part.iloc[:limit] if limit else part)
        frame = pd.concat(parts, ignore_index=True)

        frame.to_json(manifest_path, orient="records", lines=True, force_ascii=False)
        image_meta = (
            frame[["image_id", "file_name", "coco_url", "width", "height"]]
            .drop_duplicates("image_id")
            .to_dict("records")
        )
        atomic_json_dump(image_meta, image_meta_path)

        summary = frame.groupby("split").agg(
            samples=("sample_id", "size"),
            no_target=("no_target", "sum"),
            images=("image_id", "nunique"),
        )
        print(summary)
        state.mark(
            "manifest_built",
            samples=len(frame),
            images=int(frame["image_id"].nunique()),
            splits=sorted(frame["split"].unique().tolist()),
        )
        return frame, manifest_path, image_meta_path


def _download_one_image(cfg: Config, paths: Paths, row: dict[str, Any], retries: int = 4) -> None:
    output = paths.images / row["file_name"]
    if output.exists():
        try:
            with Image.open(output) as image:
                image.verify()
            return
        except Exception:
            output.unlink(missing_ok=True)

    urls = [
        row.get("coco_url"),
        f'https://images.cocodataset.org/train2014/{row["file_name"]}',
        f'http://images.cocodataset.org/train2014/{row["file_name"]}',
    ]
    urls = [url for url in dict.fromkeys(urls) if url]
    partial = output.with_suffix(output.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(retries):
        for url in urls:
            try:
                with requests.get(url, stream=True, timeout=cfg.request_timeout) as response:
                    response.raise_for_status()
                    with open(partial, "wb") as handle:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            if chunk:
                                handle.write(chunk)
                with Image.open(partial) as image:
                    image.verify()
                partial.replace(output)
                return
            except Exception as exc:
                last_error = exc
                partial.unlink(missing_ok=True)
        time.sleep(min(2**attempt, 8))

    raise RuntimeError(f"Không tải được {row['file_name']}: {last_error}")


def download_images(cfg: Config, paths: Paths, state: RunState, image_meta_path: Path) -> None:
    image_rows = json.loads(image_meta_path.read_text(encoding="utf-8"))
    with stage_timer("Session 3 — Tải các ảnh COCO cần dùng"):
        for row in tqdm(image_rows, desc="COCO images", unit="img"):
            _download_one_image(cfg, paths, row)
        state.mark("images_downloaded", count=len(image_rows))


def available_test_splits(manifest: pd.DataFrame) -> list[str]:
    available = set(manifest["split"].unique().tolist())
    preferred = [name for name in ["test", "testA", "testB"] if name in available]
    return preferred or (["val"] if "val" in available else [])
