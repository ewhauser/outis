import unicodedata

import pytest

from clinical_deid.annotation_policy import (
    ANNOTATION_POLICY, POLICY_FIXTURES, POLICY_VERSION, annotation_errors,
    canonical_identity_group, identity_split,
)


def test_executable_policy_fixtures_have_valid_unicode_character_offsets():
    assert len({row["id"] for row in POLICY_FIXTURES}) == len(POLICY_FIXTURES)
    for row in POLICY_FIXTURES:
        assert row["annotation_policy"] == POLICY_VERSION
        assert not annotation_errors(row["text"], row["spans"])
    row = next(row for row in POLICY_FIXTURES if row["id"] == "titles_and_unicode")
    assert [row["text"][span["start"]:span["end"]] for span in row["spans"]] == ["Zoë O’Rían-Soto", "Ana María de la Cruz"]
    assert row["text"][row["spans"][0]["end"]] == ","
    assert ANNOTATION_POLICY["legacy_policy_compatibility"] == "not_certified"


def test_context_not_surface_spelling_decides_person_annotation():
    fixtures = {row["id"]: row for row in POLICY_FIXTURES}
    assert fixtures["ordinary_pronouns"]["spans"] == []
    explicit = fixtures["explicit_unusual_name"]
    assert explicit["cohort"] == "unusual_name"
    assert len(explicit["spans"]) == 1
    span = explicit["spans"][0]
    assert explicit["text"][span["start"]:span["end"]] == "He"
    assert explicit["text"].count("He") == 2
    eponym = fixtures["eponym_and_name"]
    assert [eponym["text"][item["start"]:item["end"]] for item in eponym["spans"]] == ["Mira Wilson"]


def test_age_fixtures_exclude_suffix_units_sex_and_below_threshold_values():
    fixtures = {row["id"]: row for row in POLICY_FIXTURES}
    for key, expected in {
        "age_numeric_boundaries": ["92", "91"],
        "age_word_boundaries": ["ninety-two"],
        "age_compact_boundaries": ["92", "93", "94"],
    }.items():
        row = fixtures[key]
        assert [row["text"][span["start"]:span["end"]] for span in row["spans"]] == expected
        assert all(span["label"] == "AGE" for span in row["spans"])


def test_date_and_location_fixtures_retain_complete_authored_expression():
    fixtures = {row["id"]: row for row in POLICY_FIXTURES}
    row = fixtures["patient_partial_dates"]
    assert [row["text"][span["start"]:span["end"]] for span in row["spans"]] == ["March 2021", "1930"]
    row = fixtures["maximal_location"]
    assert [(span["label"], row["text"][span["start"]:span["end"]]) for span in row["spans"]] == [
        ("LOCATION", "Cedar Haven, UT"), ("FACILITY", "Silver Finch Medical Center"), ("LOCATION", "North Cedar Haven")]


def test_identity_group_normalizes_equivalent_unicode_and_case():
    name = "Zoë Núñez"
    assert canonical_identity_group(name) == canonical_identity_group(unicodedata.normalize("NFD", name).upper())
    assert identity_split(canonical_identity_group(name)) in {"train", "validation"}
    with pytest.raises(ValueError):
        identity_split("name")
    with pytest.raises(ValueError):
        canonical_identity_group("")


@pytest.mark.parametrize("spans", [
    [{"start": False, "end": 3, "label": "NAME"}],
    [{"start": 0, "end": 99, "label": "NAME"}],
    [{"start": 0, "end": 3, "label": "UNKNOWN"}],
    [{"start": 0, "end": 3, "label": "NAME"}, {"start": 2, "end": 5, "label": "DATE"}],
    [{"start": 0, "end": 4, "label": "NAME"}],
    [{"start": 0, "end": 3, "label": "NAME", "text": "Sam"}],
])
def test_structural_annotation_errors_are_rejected(spans):
    assert annotation_errors("Ana visits", spans)
