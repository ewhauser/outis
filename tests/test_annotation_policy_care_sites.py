from clinical_deid.data.annotation_policy import (
    POLICY_FIXTURES,
    POLICY_VERSION,
    annotation_errors,
)
from clinical_deid.data.annotation_policy_care_sites import (
    CARE_SITE_FIXTURES,
    CARE_SITE_FOLLOWUP_FIXTURES,
)
from clinical_deid.data.annotation_policy_places import PLACE_FIXTURES


def gold(key):
    row = {r["id"]: r for r in (*CARE_SITE_FIXTURES, *CARE_SITE_FOLLOWUP_FIXTURES)}[key]
    return [(s["label"], row["text"][s["start"] : s["end"]]) for s in row["spans"]]


def test_fixtures_are_valid_and_ids_unique_across_modules():
    ids = [
        r["id"]
        for r in (
            *POLICY_FIXTURES,
            *PLACE_FIXTURES,
            *CARE_SITE_FIXTURES,
            *CARE_SITE_FOLLOWUP_FIXTURES,
        )
    ]
    assert len(ids) == len(set(ids))
    for row in (*CARE_SITE_FIXTURES, *CARE_SITE_FOLLOWUP_FIXTURES):
        assert row["annotation_policy"] == POLICY_VERSION
        assert not annotation_errors(row["text"], row["spans"])


def test_city_modifying_unnamed_site_is_location_without_generic_word():
    assert gold("p7_city_modifies_unnamed_site") == [
        ("LOCATION", "Vellmont"),
        ("LOCATION", "Corriston"),
    ]
    assert gold("p7_named_site_is_still_facility") == [
        ("FACILITY", "Vellmont Clinic"),
        ("LOCATION", "Corriston"),
    ]


def test_service_organisations_are_facilities_with_their_type_words():
    assert gold("service_organisations_are_facilities") == [
        ("FACILITY", "Vossmere VNA"),
        ("FACILITY", "Tindle Brook DME"),
        ("FACILITY", "Saint Odile Adult Day Program"),
        ("FACILITY", "Yarrowick Laboratory Services"),
    ]
    assert gold("unnamed_service_is_not_phi") == []
    assert gold("employer_without_care_role") == [("FACILITY", "Hospice of Marrowdale")]


def test_the_is_inside_only_when_part_of_the_name():
    assert gold("the_part_of_name") == [
        ("FACILITY", "The Marlowe at Fennick Ridge"),
        ("FACILITY", "The Quillon at Ashby Mill"),
    ]
    assert gold("the_as_article") == [
        ("FACILITY", "Pellham Drug & Gift"),
        ("FACILITY", "Ashgrove Family Health"),
    ]


def test_followup_decisions():
    assert gold("county_name_is_location") == [
        ("LOCATION", "Kestrine County"),
        ("LOCATION", "Tollund County, OR"),
    ]
    assert gold("capitalised_site_word_names_site") == [
        ("FACILITY", "Ombrey Clinic"),
        ("LOCATION", "Gorsefield"),
    ]
    assert gold("case_hidden_role_decides") == [
        ("FACILITY", "OMBREY CLINIC"),
        ("FACILITY", "THE HEARTH AT OMBREY"),
        ("LOCATION", "gorsefield"),
    ]
    assert gold("the_service_organisation") == [
        ("FACILITY", "The Hearth at Ombrey"),
        ("FACILITY", "Saint Brevin VNA"),
    ]
    assert gold("p7_care_team_and_lowercase_pharmacy") == [
        ("LOCATION", "Gorsefield"),
        ("FACILITY", "harlow vane drug"),
    ]
    assert gold("ordinary_place_word_any_case") == []
    assert gold("employer_and_care_roles") == [("FACILITY", "Ombrey General Hospital")]
    row = {r["id"]: r for r in CARE_SITE_FOLLOWUP_FIXTURES}["employer_and_care_roles"]
    assert row["text"].count("Ombrey General Hospital") == 2
