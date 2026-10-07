from clinical_deid.data.annotation_policy import POLICY_VERSION, annotation_errors
from clinical_deid.data.annotation_policy_ages import AGE_FIXTURES


def gold(key):
    row = {r["id"]: r for r in AGE_FIXTURES}[key]
    return [(s["label"], row["text"][s["start"] : s["end"]]) for s in row["spans"]]


def test_age_fixtures_are_valid():
    for row in AGE_FIXTURES:
        assert row["annotation_policy"] == POLICY_VERSION
        assert not annotation_errors(row["text"], row["spans"])


def test_all_ages_decisions():
    assert [v for _, v in gold("ages_any_value")] == ["72", "89", "93", "six"]
    assert gold("ages_of_relatives") == [("AGE", "seventy-two"), ("AGE", "58")]
    assert gold("public_ages_and_durations") == []


def test_hypothetical_ages_and_decades_2026_10_07():
    assert gold("hypothetical_ages") == []
    assert [v for _, v in gold("decades")] == ["seventies", "40s", "sixties"]
