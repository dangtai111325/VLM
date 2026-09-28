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


def test_parse_multiword_class():
    parsed = RuleSlotParser().parse("red fire extinguisher")
    assert parsed.target_class == "fire extinguisher"
    assert parsed.attribute == "red"
    assert parsed.relation is None


def test_parse_multiword_relation_reference():
    parsed = RuleSlotParser().parse("white fire extinguisher to the left of traffic light")
    assert parsed.target_class == "fire extinguisher"
    assert parsed.attribute == "white"
    assert parsed.relation == "left_of"
    assert parsed.reference_class == "traffic light"


def test_empty_prompt_rejected():
    with pytest.raises(ValueError):
        RuleSlotParser().parse("   ")
