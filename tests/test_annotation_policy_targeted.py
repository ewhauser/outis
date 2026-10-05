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
from clinical_deid.data.annotation_policy_targeted import TARGETED_FIXTURES


def gold(key):
    row = {r["id"]: r for r in TARGETED_FIXTURES}[key]
    return [(s["label"], row["text"][s["start"] : s["end"]]) for s in row["spans"]]


def test_targeted_fixtures_are_valid_and_ids_unique():
    every = (
        *POLICY_FIXTURES,
        *PLACE_FIXTURES,
        *CARE_SITE_FIXTURES,
        *CARE_SITE_FOLLOWUP_FIXTURES,
        *TARGETED_FIXTURES,
    )
    ids = [r["id"] for r in every]
    assert len(ids) == len(set(ids))
    for row in TARGETED_FIXTURES:
        assert row["annotation_policy"] == POLICY_VERSION
        assert not annotation_errors(row["text"], row["spans"])


def test_targeted_decisions():
    assert [v for _, v in gold("history_list_years")] == [
        "2011",
        "2019",
        "1988",
        "Mar 2015",
    ]
    assert gold("public_years_and_durations_are_not_dates") == []
    assert gold("pharmacy_then_zip") == [
        ("FACILITY", "Quarrington Drug"),
        ("LOCATION", "53728"),
        ("FACILITY", "Felsham Apothecary"),
        ("LOCATION", "36184"),
    ]
    assert gold("phone_written_extension") == [
        ("PHONE", "(612) 555-0193, extension 2276"),
        ("PHONE", "719.555.0104 EXT. 318"),
    ]
    assert gold("funeral_home_is_facility") == [
        ("FACILITY", "Felsham Funeral Chapel"),
        ("FACILITY", "Orlinde & Sons Funeral Home"),
    ]
