from __future__ import annotations

from dataclasses import dataclass
import re


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
    "on top of": "on_top_of",
    "overlapping": "overlapping",
}

ATTRIBUTE_WORDS = {
    "black", "white", "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown",
    "gray", "grey", "small", "large", "big", "little", "tall", "short", "dark", "light",
}
_STOPWORDS = {"the", "a", "an", "object", "thing", "please", "find", "locate", "detect"}


class RuleSlotParser:
    """Deterministic parser for the locked V1 contract.

    Contract: target class + at most one simple attribute + at most one first-order
    relation + one reference class. Multiword classes such as ``fire extinguisher`` or
    ``traffic light`` are preserved instead of being reduced to their final token.
    """

    def __init__(self, relations: dict[str, str] | None = None):
        self.relations = relations or DEFAULT_RELATIONS
        escaped = sorted((re.escape(k) for k in self.relations), key=len, reverse=True)
        self.pattern = re.compile(r"(?<!\w)(" + "|".join(escaped) + r")(?!\w)", re.IGNORECASE)

    @staticmethod
    def _clean_tokens(text: str) -> list[str]:
        tokens = re.findall(r"[\w'-]+", text.lower())
        return [t for t in tokens if t not in _STOPWORDS]

    @staticmethod
    def _target_phrase(text: str) -> tuple[str | None, str | None]:
        tokens = RuleSlotParser._clean_tokens(text)
        if not tokens:
            return None, None
        attribute = None
        if tokens and tokens[0] in ATTRIBUTE_WORDS:
            attribute = tokens.pop(0)
        target = " ".join(tokens).strip() or None
        return target, attribute

    @staticmethod
    def _reference_phrase(text: str) -> str | None:
        tokens = RuleSlotParser._clean_tokens(text)
        # A reference attribute is outside the V1 contract. Preserve the noun phrase but
        # drop one leading simple adjective so YOLOE receives the object class itself.
        if tokens and tokens[0] in ATTRIBUTE_WORDS:
            tokens = tokens[1:]
        return " ".join(tokens).strip() or None

    def parse(self, text: str) -> ParsedPrompt:
        raw = text.strip()
        if not raw:
            raise ValueError("prompt cannot be empty")

        match = self.pattern.search(raw.lower())
        if match is None:
            target, attribute = self._target_phrase(raw)
            if target is None:
                raise ValueError(f"cannot parse target class from: {text!r}")
            result = ParsedPrompt(raw=raw, target_class=target, attribute=attribute)
            result.validate()
            return result

        relation_surface = match.group(1).lower()
        relation = self.relations[relation_surface]
        target, attribute = self._target_phrase(raw[: match.start()])
        reference = self._reference_phrase(raw[match.end():])
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
