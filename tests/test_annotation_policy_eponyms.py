from clinical_deid.data.annotation_policy import POLICY_VERSION, annotation_errors
from clinical_deid.data.annotation_policy_eponyms import EPONYM_FIXTURES


def gold(key):
    row = {r["id"]: r for r in EPONYM_FIXTURES}[key]
    return [(s["label"], row["text"][s["start"] : s["end"]]) for s in row["spans"]]


def test_eponym_fixtures_are_valid():
    for row in EPONYM_FIXTURES:
        assert row["annotation_policy"] == POLICY_VERSION
        assert not annotation_errors(row["text"], row["spans"])


def test_instruments_are_not_annotated_but_people_are():
    assert gold("eponymous_instruments_shorthand") == []
    assert gold("eponymous_instruments_fields") == []
    assert gold("same_surname_two_roles") == [
        ("NAME", "Corin Wells"),
        ("NAME", "Ana Murphy"),
    ]
