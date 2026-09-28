from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_grefcoco.py"
spec = spec_from_file_location("prepare_grefcoco", SCRIPT)
assert spec is not None and spec.loader is not None
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_target_attribute_and_relation_slots():
    attribute, relation, reference = module.extract_slots(
        "the red cup next to the pillow",
        "cup",
        ["cup", "pillow"],
    )
    assert attribute == "red"
    assert relation == "next_to"
    assert reference == "pillow"


def test_reference_attribute_not_mislabeled_as_target_attribute():
    attribute, relation, reference = module.extract_slots(
        "the cup next to the red pillow",
        "cup",
        ["cup", "pillow"],
    )
    assert attribute is None
    assert relation == "next_to"
    assert reference == "pillow"


def test_multiword_reference_class():
    attribute, relation, reference = module.extract_slots(
        "the white bottle left of the traffic light",
        "bottle",
        ["bottle", "traffic light", "light"],
    )
    assert attribute == "white"
    assert relation == "left_of"
    assert reference == "traffic light"


def test_coco_image_url_uses_declared_or_inferred_split():
    assert module.coco_image_url(
        {"file_name": "COCO_val2014_000000123456.jpg"}
    ) == "https://images.cocodataset.org/val2014/COCO_val2014_000000123456.jpg"
    assert module.coco_image_url(
        {
            "file_name": "COCO_train2014_000000123456.jpg",
            "coco_url": "http://images.cocodataset.org/train2014/COCO_train2014_000000123456.jpg",
        }
    ) == "https://images.cocodataset.org/train2014/COCO_train2014_000000123456.jpg"
