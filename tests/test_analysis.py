import json

import pytest

from clinical_deid.analysis import analyze_entities
from clinical_deid.entities import Entity


def analyzed(value, label, **options):
    return analyze_entities(value, [Entity(0, len(value), label)], **options)["entities"][0]["analysis"]


@pytest.mark.parametrize("value,expected,precision", [
    ("2024-02-29", "2024-02-29", "day"),
    ("2024/2/9", "2024-02-09", "day"),
    ("02/29/2024", "2024-02-29", "day"),
    ("February 29, 2024", "2024-02-29", "day"),
    ("29 Feb. 2024", "2024-02-29", "day"),
    ("January 1st, 2024", "2024-01-01", "day"),
    ("Sept 2024", "2024-09", "month"),
    ("2024", "2024", "year"),
])
def test_dates_parse_complete_value_with_explicit_precision(value, expected, precision):
    result = analyzed(value, "DATE", include_sensitive=True)
    assert result["status"] == "parsed"
    assert result["normalized"] == expected
    assert result["precision"] == precision
    assert "normalized" not in analyzed(value, "DATE")


@pytest.mark.parametrize("value", ["2023-02-29", "2024-13-01", "2024-01-00", "2024-00-01", "31 February 2024", "0000", "January 1th, 2024", "February 11st, 2024", "21th January 2024"])
def test_impossible_calendar_values_are_invalid(value):
    assert analyzed(value, "DOB", include_sensitive=True)["status"] == "invalid"


@pytest.mark.parametrize("value,status", [
    ("1/2/24", "ambiguous"), ("1/2", "ambiguous"),
    ("tomorrow", "unsupported"), ("admitted 2024-02-29", "unsupported"),
    ("Feb 1-3, 2024", "unsupported"), ("2024-02-29 12:30", "unsupported"),
    ("2024/02-29", "unsupported"),
])
def test_uncertain_or_partial_dates_do_not_invent_components(value, status):
    result = analyzed(value, "DATE", include_sensitive=True)
    assert result["status"] == status
    assert "normalized" not in result


def test_numeric_dates_use_explicit_english_locale_and_disclose_ambiguity():
    us = analyzed("03/04/2024", "DATE", include_sensitive=True)
    gb = analyzed("03/04/2024", "DATE", locale="en-GB", include_sensitive=True)
    assert us["normalized"] == "2024-03-04"
    assert gb["normalized"] == "2024-04-03"
    assert us["locale_dependent"] is True
    assert analyzed("31/01/2024", "DATE")["status"] == "invalid"


@pytest.mark.parametrize("value,normalized", [
    ("45", {"value": 45, "unit": "years"}),
    ("45-year-old", {"value": 45, "unit": "years"}),
    ("45 y/o", {"value": 45, "unit": "years"}),
    ("age 45 years", {"value": 45, "unit": "years"}),
    ("18 months old", {"value": 18, "unit": "months"}),
    ("0 days", {"value": 0, "unit": "days"}),
    ("90+", {"minimum": 90, "unit": "years", "inclusive": True}),
])
def test_ages_preserve_units_and_ranges_without_inventing_birthdates(value, normalized):
    result = analyzed(value, "AGE", include_sensitive=True)
    assert result["status"] == "parsed"
    assert result["normalized"] == normalized
    assert "normalized" not in analyzed(value, "AGE")


@pytest.mark.parametrize("value,status", [("131 years", "invalid"), ("in his forties", "unsupported"), ("40-45", "unsupported"), ("-1", "unsupported")])
def test_invalid_or_approximate_ages_do_not_become_exact(value, status):
    assert analyzed(value, "AGE")["status"] == status


def test_address_components_are_conservative_and_sensitive():
    value = "123 North Main St Apt 4B, Salt Lake City, UT 84101-1234, USA"
    public = analyzed(value, "ADDRESS")
    private = analyzed(value, "ADDRESS", include_sensitive=True)
    assert public == {"kind": "address", "status": "parsed", "format": "us_postal"}
    assert private["components"] == {
        "street_number": "123", "street": "North Main St", "unit": "Apt 4B",
        "city": "Salt Lake City", "state": "UT", "postal_code": "84101-1234", "country": "US",
    }
    assert analyzed("123 Main St", "ADDRESS")["status"] == "unsupported"
    assert analyzed("123 Main St, Nowhere, XX 12345", "ADDRESS")["status"] == "unsupported"


@pytest.mark.parametrize("value,valid", [
    ("4111 1111 1111 1111", True), ("5555-5555-5555-4444", True),
    ("4111111111111112", False), ("0000000000000000", False),
    ("4111111111111111x", False), ("12345", False),
])
def test_card_checksum_does_not_expose_card_number(value, valid):
    result = analyzed(value, "CREDIT_CARD")
    assert result["checksum_valid"] is valid
    assert value not in json.dumps(result)


@pytest.mark.parametrize("value,valid", [("021000021", True), ("021000022", False), ("000000000", False)])
def test_aba_routing_checksum(value, valid):
    assert analyzed(value, "ROUTING_NUMBER")["checksum_valid"] is valid


def span(text, value, label):
    start = text.index(value)
    return Entity(start, start + len(value), label)


@pytest.mark.parametrize("text,relationship", [
    ("Mara was born on 2024-01-01", "born_on"),
    ("Mara discussed 2024-01-01", None),
    ("Mara was not born on 2024-01-01", None),
    ("If Mara was born on 2024-01-01", None),
    ("Mara was born on\n2024-01-01", None),
])
def test_relationships_require_explicit_supported_nonnegated_adjacent_predicate(text, relationship):
    result = analyze_entities(text, [span(text, "Mara", "NAME"), span(text, "2024-01-01", "DATE")])
    expected = [] if relationship is None else [{"type": relationship, "source_entity": 0, "target_entity": 1, "source": "rule:explicit_relation"}]
    assert result["relationships"] == expected
    assert result["metadata"]["relationship_status"] == "experimental"
    assert "Mara" not in json.dumps(result)
    assert "2024-01-01" not in json.dumps(result)


def test_unknown_labels_are_preserved_without_fabricated_analysis():
    result = analyze_entities("STUDY-42", [Entity(0, 8, "STUDY_ID")])
    assert "analysis" not in result["entities"][0]


@pytest.mark.parametrize("text", [
    "It is not true that John has diabetes.",
    "If patient John has diabetes, order a test.",
    "John has diabetes?",
    "John has diabetes ruled out.",
    "John has diabetes in the family.",
    "John has diabetes if the pending result is positive.",
    "John has diabetes in his mother.",
])
def test_relationships_do_not_turn_uncertain_or_family_mentions_into_assertions(text):
    result = analyze_entities(text, [span(text, "John", "NAME"), span(text, "diabetes", "CONDITION")])
    assert result["relationships"] == []


@pytest.mark.parametrize("options", [{"locale": "fr-FR"}, {"include_sensitive": "yes"}])
def test_invalid_analysis_options_fail_before_extracting_values(options):
    with pytest.raises(ValueError, match="Invalid analysis options"):
        analyze_entities("secret", [Entity(0, 6, "NAME")], **options)


def test_malformed_or_overlapping_entities_fail_closed():
    with pytest.raises(ValueError, match="Invalid entity output"):
        analyze_entities("secret", [Entity(0, 7, "NAME")])
    with pytest.raises(ValueError, match="Overlapping detector output"):
        analyze_entities("secret", [Entity(0, 3, "NAME"), Entity(2, 6, "ID")])
