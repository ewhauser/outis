from outis.data.annotation_policy import POLICY_VERSION, annotation_errors
from outis.data.annotation_policy_facility_names import FACILITY_NAME_FIXTURES


def gold(key):
    row = {r["id"]: r for r in FACILITY_NAME_FIXTURES}[key]
    return [(s["label"], row["text"][s["start"] : s["end"]]) for s in row["spans"]]


def test_fixtures_are_valid():
    for row in FACILITY_NAME_FIXTURES:
        assert row["annotation_policy"] == POLICY_VERSION
        assert not annotation_errors(row["text"], row["spans"])


def test_official_name_with_town_is_one_facility():
    assert gold("official_name_ending_in_town")[0] == (
        "FACILITY",
        "Northfield Health-Larkin Bay",
    )


def test_short_name_label_follows_role():
    assert gold("short_name_equal_to_town") == [
        ("FACILITY", "Tessford"),
        ("LOCATION", "Tessford"),
    ]
