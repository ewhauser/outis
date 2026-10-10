from outis.policy import (
    DetectionPolicy,
    age_value,
    apply_policy,
    policy_metadata,
)
from outis.types import Span

TEXT = "A 72-year-old, her mother died at seventy-two, and her aunt is 93; age unknown-ish."


def spans():
    return [
        Span(2, 4, "AGE", 1.0, "model"),
        Span(34, 45, "AGE", 1.0, "model"),
        Span(63, 65, "AGE", 1.0, "model"),
        Span(67, 70, "AGE", 1.0, "model"),
    ]


def values(entities):
    return [TEXT[e.start : e.end] for e in entities]


def test_default_keeps_ages_89_and_under_in_place():
    kept = apply_policy(TEXT, spans(), None)
    assert values(kept) == ["93", "age"]  # unparsed ages stay removed


def test_all_ages_policy_removes_every_age():
    kept = apply_policy(TEXT, spans(), DetectionPolicy(ages="all"))
    assert values(kept) == ["72", "seventy-two", "93", "age"]


def test_age_value_parses_numbers_and_words():
    assert age_value("72") == 72
    assert age_value("72-year-old") == 72
    assert age_value("seventy-two") == 72
    assert age_value("ninety-three") == 93
    assert age_value("unknown") is None


def test_policy_metadata_reports_ages_setting():
    assert policy_metadata(None)["ages"] == "over_89"
    assert policy_metadata({"ages": "all"})["ages"] == "all"
