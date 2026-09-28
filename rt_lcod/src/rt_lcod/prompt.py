from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class ParsedPrompt:
    raw: str
    target_class: str
    attribute: str | None = None
    relation: str | None = None
    reference_class: str | None = None

    def validate(self) -> None:
        if not self.target_class.strip():
            raise ValueError("target_class must be non-empty")
        if self.relation and not self.reference_class:
            raise ValueError("reference_class is required when relation is present")


DEFAULT_RELATIONS = {
    "next to": "next_to",
    "beside": "next_to",
    "near": "near",
    "left of": "left_of",
    "to the left of": "left_of",
    "right of": "right_of",
    "to the right of": "right_of",
    "above": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "inside": "inside",
    "in": "inside",
    "on": "on",
    "overlapping": "overlapping",
}

_STOPWORDS = {"the", "a", "an", "object", "thing"}


class RuleSlotParser:
    """Deterministic V1 parser for one target, <=1 attribute and <=1 relation.

    This parser intentionally favors reproducibility over unrestricted NLP. It accepts
    common English relation phrases and assumes the noun immediately before/after the
    relation is the target/reference class. Tokens before the target noun are treated as
    one attribute phrase. A trainable token tagger can replace this class later without
    changing the grounding model API.
    """

    def __init__(self, relations: dict[str, str] | None = None):
        self.relations = relations or DEFAULT_RELATIONS
        escaped = sorted((re.escape(k) for k in self.relations), key=len, reverse=True)
        self.pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

    @staticmethod
    def _clean_tokens(text: str) -> list[str]:
        tokens = re.findall(r"[\w'-]+", text.lower())
        return [t for t in tokens if t not in _STOPWORDS]

    @staticmethod
    def _noun_phrase(tokens: Iterable[str]) -> tuple[str | None, str | None]:
        values = list(tokens)
        if not values:
            return None, None
        target = values[-1]
        attribute = " ".join(values[:-1]).strip() or None
        return target, attribute

    def parse(self, text: str) -> ParsedPrompt:
        raw = text.strip()
        if not raw:
            raise ValueError("prompt cannot be empty")

        match = self.pattern.search(raw.lower())
        if match is None:
            target_tokens = self._clean_tokens(raw)
            target, attribute = self._noun_phrase(target_tokens)
            if target is None:
                raise ValueError(f"cannot parse target class from: {text!r}")
            result = ParsedPrompt(raw=raw, target_class=target, attribute=attribute)
            result.validate()
            return result

        relation_surface = match.group(1).lower()
        relation = self.relations[relation_surface]
        left = raw[: match.start()]
        right = raw[match.end():]
        target, attribute = self._noun_phrase(self._clean_tokens(left))
        reference, _ = self._noun_phrase(self._clean_tokens(right))
        if target is None or reference is None:
            raise ValueError(f"cannot parse one-hop relation from: {text!r}")
        result = ParsedPrompt(
            raw=raw,
            target_class=target,
            attribute=attribute,
            relation=relation,
            reference_class=reference,
        )
        result.validate()
        return result
