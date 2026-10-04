from collections import Counter

import pytest

from clinical_deid.annotation_policy import POLICY_VERSION, annotation_errors
from clinical_deid.annotation_policy_places import PLACE_FIXTURES, VALUE_POOLS, place_value_pool


FIXTURES = {row["id"]: row for row in PLACE_FIXTURES}


def gold(row):
    return [(row["text"][s["start"]:s["end"]], s["label"]) for s in row["spans"]]


def test_fixtures_are_structurally_valid_v2_rows():
    assert len(FIXTURES) == len(PLACE_FIXTURES)
    for row in PLACE_FIXTURES:
        assert row["annotation_policy"] == POLICY_VERSION
        assert annotation_errors(row["text"], row["spans"]) == []


@pytest.mark.parametrize("key", [k for k in FIXTURES if k.startswith("n") and k[1].isdigit()])
def test_public_reference_negatives_have_no_place_spans(key):
    assert not any(label in {"LOCATION", "FACILITY", "ADDRESS"} for _, label in gold(FIXTURES[key]))


def test_leading_abbreviation_stays_inside_full_care_site():
    assert gold(FIXTURES["p4_full_care_site"]) == [("St. Francis Hospital", "FACILITY")]


def test_care_mention_is_annotated_but_citation_is_not():
    row = FIXTURES["mixed_care_beats_citation"]
    assert gold(row) == [("Mayo Clinic", "FACILITY")]
    assert row["spans"][0]["start"] == row["text"].index("Mayo Clinic")


def test_state_inside_locality_is_kept_and_state_alone_is_not():
    assert gold(FIXTURES["state_inside_locality"]) == [("Tacoma, WA", "LOCATION")]


def test_value_pool_is_deterministic_normalized_and_shared_by_abbreviations():
    assert place_value_pool("Chicago") == place_value_pool("  CHICAGO ")
    assert place_value_pool("Café City") == place_value_pool("Café City")
    expansion = "University of California, San Francisco"
    assert place_value_pool("UCSF", expansion) == place_value_pool(expansion)
    with pytest.raises(ValueError):
        place_value_pool(" ")


def test_value_pools_are_roughly_eighty_twenty():
    counts = Counter(place_value_pool(f"Testford {index}") for index in range(5000))
    assert set(counts) == set(VALUE_POOLS)
    assert 0.77 < counts["train"] / 5000 < 0.83
