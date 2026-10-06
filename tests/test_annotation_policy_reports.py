from clinical_deid.data.annotation_policy import POLICY_VERSION, annotation_errors
from clinical_deid.data.annotation_policy_reports import REPORT_FIXTURES


def gold(key):
    row = {r["id"]: r for r in REPORT_FIXTURES}[key]
    return [(s["label"], row["text"][s["start"] : s["end"]]) for s in row["spans"]]


def test_report_fixtures_are_valid():
    for row in REPORT_FIXTURES:
        assert row["annotation_policy"] == POLICY_VERSION
        assert not annotation_errors(row["text"], row["spans"])


def test_report_decisions():
    assert gold("lab_identifiers") == [
        ("FACILITY", "Ostravel Reference Laboratory"),
        ("ID", "32D9917046"),
        ("ID", "SP-26-118734"),
    ]
    assert gold("initials_only_person") == [
        ("NAME", "JKL"),
        ("NAME", "MKT"),
        ("NAME", "Quennell"),
    ]
    assert gold("report_codes_not_phi") == []
    assert gold("report_datetimes") == [
        ("ID", "RA-7720931"),
        ("DATE", "03/14/2026"),
        ("DATE", "Mar 15, 2026"),
    ]
