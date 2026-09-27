from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from rt_lcod.data.schema import load_manifest


def replace_slot(sample, slot: str, new_value: str):
    slots = {
        "target_class": sample.target_class,
        "attribute": sample.attribute,
        "relation": sample.relation,
        "reference_class": sample.reference_class,
    }
    slots[slot] = new_value
    pieces = []
    if slots["attribute"]:
        pieces.append(slots["attribute"])
    pieces.append(slots["target_class"])
    if slots["relation"] and slots["reference_class"]:
        pieces.extend([slots["relation"].replace("_", " "), slots["reference_class"]])
    return " ".join(pieces), slots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate hard-negative prompts. Verification is required before training.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attributes", nargs="*", default=["red", "blue", "green", "black", "white"])
    parser.add_argument("--relations", nargs="*", default=["next_to", "left_of", "right_of", "above", "below"])
    parser.add_argument("--classes", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    samples = load_manifest(args.manifest)
    rng = random.Random(args.seed)
    classes = args.classes or sorted({s.target_class for s in samples} | {s.reference_class for s in samples if s.reference_class})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            mutations = []
            choices = [x for x in classes if x != sample.target_class]
            if choices:
                mutations.append(("target_class", rng.choice(choices)))
            if sample.attribute:
                choices = [x for x in args.attributes if x != sample.attribute]
                if choices:
                    mutations.append(("attribute", rng.choice(choices)))
            if sample.relation:
                choices = [x for x in args.relations if x != sample.relation]
                if choices:
                    mutations.append(("relation", rng.choice(choices)))
            if sample.reference_class:
                choices = [x for x in classes if x != sample.reference_class]
                if choices:
                    mutations.append(("reference_class", rng.choice(choices)))
            for slot, value in mutations:
                prompt, slots = replace_slot(sample, slot, value)
                record = {
                    "source_sample_id": sample.sample_id,
                    "image": sample.image,
                    "prompt": prompt,
                    "slots": slots,
                    "mutation": {"slot": slot, "value": value},
                    "verified_negative": False,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    print(f"wrote {written} candidate negatives to {output}")
    print("IMPORTANT: verify each candidate against annotations before using it as no-target supervision.")


if __name__ == "__main__":
    main()
