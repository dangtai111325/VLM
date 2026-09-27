"""Persistent LocateAnything-3B JSONL worker.

This worker is intentionally separate from the notebook kernel because the
LocateAnything model card pins transformers==4.57.1, while the main benchmark
environment uses newer Transformers for detector integrations.

Protocol:
- prints READY after loading
- reads one JSON object per line from stdin
- writes one JSON object per line to stdout
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys

import torch
from PIL import Image


def parse_boxes(answer: str, width: int, height: int) -> list[dict]:
    boxes = []
    for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = [int(group) for group in match.groups()]
        boxes.append(
            {
                "x1": x1 / 1000 * width,
                "y1": y1 / 1000 * height,
                "x2": x2 / 1000 * width,
                "y2": y2 / 1000 * height,
            }
        )
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    with contextlib.redirect_stdout(sys.stderr):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

    print("READY", flush=True)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
            if request.get("command") == "shutdown":
                break

            image = Image.open(request["image"]).convert("RGB")
            phrase = request["prompt"].strip()
            question = (
                "Locate all the instances that match the following description: "
                f"{phrase}."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            text = processor.py_apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            images, videos = processor.process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=images,
                videos=videos,
                return_tensors="pt",
            ).to(device)

            pixel_values = inputs["pixel_values"].to(dtype)
            with torch.inference_mode():
                response = model.generate(
                    pixel_values=pixel_values,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_grid_hws=inputs.get("image_grid_hws", None),
                    tokenizer=tokenizer,
                    max_new_tokens=int(request.get("max_new_tokens", 2048)),
                    use_cache=True,
                    generation_mode=request.get("mode", "hybrid"),
                    temperature=0.7,
                    do_sample=False,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    verbose=False,
                )

            answer = response[0] if isinstance(response, tuple) else response
            answer = str(answer)
            boxes = parse_boxes(answer, *image.size)
            print(
                json.dumps(
                    {"answer": answer, "boxes": boxes},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
