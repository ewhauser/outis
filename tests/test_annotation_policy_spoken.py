from outis.data.annotation_policy import POLICY_VERSION, annotation_errors
from outis.data.annotation_policy_spoken import SPOKEN_FIXTURES


def gold(key):
    row = {r["id"]: r for r in SPOKEN_FIXTURES}[key]
    return [(s["label"], row["text"][s["start"] : s["end"]]) for s in row["spans"]]


def test_spoken_fixtures_are_valid():
    for row in SPOKEN_FIXTURES:
        assert row["annotation_policy"] == POLICY_VERSION
        assert not annotation_errors(row["text"], row["spans"])


def test_spoken_decisions():
    assert gold("spoken_phone_and_area_code") == [
        ("PHONE", "five five five, uh, oh one four two"),
        ("PHONE", "six one two"),
    ]
    assert [label for label, _ in gold("spoken_ssn_mrn_zip")] == [
        "SSN",
        "MRN",
        "LOCATION",
    ]
    assert gold("spelled_name") == [("NAME", "Brannock"), ("NAME", "b r a n n o c k")]
    assert gold("spoken_email") == [("EMAIL", "dally dot brannock at example dot com")]
    assert [v for _, v in gold("spoken_dates")] == [
        "oh three, oh three, forty-one",
        "march third",
        "oh nine",
        "forty-one",
        "the twelfth",
    ]
    assert gold("spoken_correction") == [
        ("PHONE", "five five six"),
        ("PHONE", "five five seven, oh one four two"),
    ]


def test_verification_decisions():
    assert gold("partial_address_verification") == [
        ("ADDRESS", "wenlow lane"),
        ("ADDRESS", "four twelve"),
        ("ADDRESS", "apartment two b"),
        ("LOCATION", "five three seven two eight"),
    ]
    assert gold("split_across_turns") == [
        ("DATE", "march"),
        ("DATE", "third, forty-one"),
        ("PHONE", "five five five, oh one four two"),
    ]
