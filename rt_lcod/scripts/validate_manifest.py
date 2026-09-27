from __future__ import annotations

import argparse
from collections import Counter

from rt_lcod.data.schema import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    samples = load_manifest(args.manifest)
    relations = Counter(s.relation or "<none>" for s in samples)
    attributes = Counter("present" if s.attribute else "absent" for s in samples)
    negatives = sum(s.no_target for s in samples)
    print(f"validated: {len(samples)} samples")
    print(f"no-target: {negatives} ({negatives / len(samples):.1%})")
    print(f"attributes: {dict(attributes)}")
    print(f"relations: {dict(relations)}")


if __name__ == "__main__":
    main()
