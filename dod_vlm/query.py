from __future__ import annotations

import re


RELATION_PATTERNS: list[tuple[str, str]] = [
    (r"\bto the left of\b|\bleft of\b", "left_of"),
    (r"\bto the right of\b|\bright of\b", "right_of"),
    (r"\bon top of\b", "on_top_of"),
    (r"\babove\b|\bover\b", "above"),
    (r"\bbelow\b|\bunder\b", "below"),
    (r"\bnext to\b|\bbeside\b|\bnear\b", "near"),
    (r"\bin front of\b", "in_front_of"),
    (r"\bbehind\b", "behind"),
    (r"\binside\b", "inside"),
    (r"\boverlapping\b|\boverlaps?\b", "overlap"),
]
RELATION_NAMES = [
    "none",
    "left_of",
    "right_of",
    "above",
    "below",
    "near",
    "in_front_of",
    "behind",
    "inside",
    "on_top_of",
    "overlap",
]
REL2ID = {name: index for index, name in enumerate(RELATION_NAMES)}

_COMMAND_PREFIX = re.compile(
    r"^(?:please\s+)?(?:find|locate|detect|show|select|identify)\s+(?:me\s+)?(?:all\s+|the\s+|a\s+|an\s+)?",
    re.IGNORECASE,
)
_ARTICLES = re.compile(r"\b(?:the|a|an)\b", re.IGNORECASE)


def clean_phrase(text: str) -> str:
    text = _COMMAND_PREFIX.sub("", text.strip())
    text = _ARTICLES.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.;:")


def parse_query(query: str) -> dict[str, str | int | bool]:
    """Deterministic parser used identically in training and inference.

    It is deliberately lightweight: A0 still receives the complete sentence, while
    A1 receives a best-effort target phrase, optional anchor phrase, and relation.
    No ground-truth category is injected into the model input.
    """

    raw = query.strip()
    if not raw:
        raise ValueError("query không được rỗng")

    lowered = raw.lower()
    relation_name = "none"
    target_surface = raw
    anchor_phrase = ""

    for pattern, name in RELATION_PATTERNS:
        match = re.search(pattern, lowered)
        if match is not None:
            relation_name = name
            target_surface = raw[: match.start()]
            anchor_phrase = clean_phrase(raw[match.end() :])
            break

    target_phrase = clean_phrase(target_surface) or clean_phrase(raw)
    return {
        "target_phrase": target_phrase,
        "anchor_phrase": anchor_phrase,
        "relation_name": relation_name,
        "relation_id": REL2ID[relation_name],
        "has_relation": relation_name != "none" and bool(anchor_phrase),
    }
