# Data directory

Generated data is intentionally not committed.

Recommended local layout:

```text
data/
  manifests/
    train.jsonl
    val.jsonl
    test.jsonl
  images/
  candidate_cache/
    train/
    val/
  teacher_cache/
    train/
  robot/
```

The canonical manifest schema is described in `../../plan.md` and validated by `scripts/validate_manifest.py`.
