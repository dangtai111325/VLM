import pytest

from rt_lcod.prompt import RuleSlotParser


def test_parse_class_attribute_relation():
    parsed = RuleSlotParser().parse("the red cup next to the pillow")
    assert parsed.target_class == "cup"
    assert parsed.attribute == "red"
    assert parsed.relation == "next_to"
    assert parsed.reference_class == "pillow"


def test_parse_class_only():
    parsed = RuleSlotParser().parse("cup")
    assert parsed.target_class == "cup"
    assert parsed.attribute is None
    assert parsed.relation is None


def test_empty_prompt_rejected():
    with pytest.raises(ValueError):
        RuleSlotParser().parse("   ")
