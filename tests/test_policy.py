import json
import time

import pytest

from outis.entities import Entity
from outis.policy import (
    DetectionPolicy,
    apply_policy,
    policy_metadata,
    validate_policy,
)
from outis.types import Span


def values(text, spans):
    return [(text[item.start : item.end], item.label) for item in spans]


def test_default_policy_preserves_existing_detector_decisions_and_order():
    text = "Mara on 2026-01-01"
    spans = [Span(8, 18, "DATE", 0.8, "test"), Span(0, 4, "NAME", 0.9, "test")]
    result = apply_policy(text, spans)
    assert [item.as_dict() for item in result] == [item.as_dict() for item in spans]


def test_custom_labels_do_not_modify_model_taxonomy_and_use_unicode_offsets():
    text = "🩺 Subject STUDY-482901"
    result = apply_policy(
        text,
        [],
        {"custom_entities": [{"label": "STUDY_ID", "pattern": r"\bSTUDY-\d{6}\b"}]},
    )
    assert values(text, result) == [("STUDY-482901", "STUDY_ID")]
    assert result[0].start == 10
    assert result[0].source == "policy:custom"
    with pytest.raises(ValueError):
        Span(0, 1, "STUDY_ID")


def test_include_then_exclude_with_profile_and_forced_blocklist_precedence():
    text = "Mara 2026-01-01 secret"
    original = [Span(0, 4, "NAME"), Span(5, 15, "DATE")]
    assert values(
        text, apply_policy(text, original, {"profile": "identifiers_only"})
    ) == [("Mara", "NAME")]
    result = apply_policy(
        text,
        original,
        {
            "include": [],
            "exclude": ["ID"],
            "allowlist": ["secret"],
            "blocklist": ["secret"],
        },
    )
    assert values(text, result) == [("secret", "ID")]
    result = apply_policy(
        text, original, {"include": ["NAME", "DATE"], "exclude": ["DATE"]}
    )
    assert values(text, result) == [("Mara", "NAME")]


@pytest.mark.parametrize(
    "policy,text",
    [
        ({"allowlist": ["Mara"]}, "Mara"),
        (
            {"profile": "extended", "allowlist": ["4111111111111111"]},
            "4111111111111111",
        ),
        (
            {
                "custom_entities": [{"label": "STUDY_ID", "pattern": r"STUDY-\d+"}],
                "allowlist": ["STUDY-42"],
            },
            "STUDY-42",
        ),
        (
            {"medical_terms": {"CONDITION": ["asthma"]}, "allowlist": ["asthma"]},
            "asthma",
        ),
    ],
)
def test_allowlist_applies_to_all_detector_sources(policy, text):
    original = [Span(0, len(text), "NAME")] if text == "Mara" else []
    assert apply_policy(text, original, policy) == []


def test_allowlist_exact_case_sensitive_and_never_crops_partial_span():
    text = "Mara Calder"
    original = [Span(0, len(text), "NAME")]
    assert apply_policy(text, original, {"allowlist": ["Mara"]})[0].end == len(text)
    assert apply_policy(text, original, {"allowlist": ["mara calder"]})[0].end == len(
        text
    )


def test_medical_dictionaries_are_opt_in_literal_bounded_case_insensitive():
    text = "Asthma and metformin, not asthmaish."
    assert apply_policy(text, []) == []
    policy = {"medical_terms": {"CONDITION": ["asthma"], "MEDICATION": ["metformin"]}}
    assert values(text, apply_policy(text, [], policy)) == [
        ("Asthma", "CONDITION"),
        ("metformin", "MEDICATION"),
    ]


def test_overlapping_rules_union_full_coverage_with_deterministic_label():
    text = "abcdefghijk"
    original = [Span(0, 5, "NAME")]
    policy = {
        "custom_entities": [{"label": "CUSTOM", "pattern": "defghi"}],
        "blocklist": ["hijk"],
    }
    result = apply_policy(text, original, policy)
    assert values(text, result) == [(text, "ID")]
    assert result[0].source == "policy:blocklist"


@pytest.mark.parametrize(
    "text,value,label",
    [
        ("CVV: 123", "123", "CVV"),
        ("routing number: 021000021", "021000021", "ROUTING_NUMBER"),
        ("passport: AB1234567", "AB1234567", "PASSPORT"),
        ("driver's license: D-718291", "D-718291", "DRIVER_LICENSE"),
        ("dob: 1980-01-02", "1980-01-02", "DOB"),
        ("username: mara@example.test", "mara@example.test", "USERNAME"),
        ('password: "secret words"', '"secret words"', "PASSWORD"),
        (
            "passphrase: correct horse battery staple",
            "correct horse battery staple",
            "PASSWORD",
        ),
        (
            "password: first;second other-field: third",
            "first;second other-field: third",
            "PASSWORD",
        ),
        (
            "iban: GB82 WEST 1234 5698 7654 32",
            "GB82 WEST 1234 5698 7654 32",
            "BANK_ACCOUNT",
        ),
    ],
)
def test_extended_identifiers_preserve_complete_value(text, value, label):
    assert values(text, apply_policy(text, [], {"profile": "extended"})) == [
        (value, label)
    ]
    assert apply_policy(text, []) == []


@pytest.mark.parametrize(
    "policy",
    [
        {"include": ["UNDEFINED"]},
        {"profile": "unknown"},
        {"custom_entities": [{"label": "SECRET", "pattern": "[SECRET"}]},
        {"custom_entities": [{"label": "SECRET", "pattern": "a*"}]},
        {"custom_entities": [{"label": "SECRET", "pattern": "x{1000000000}"}]},
        {"custom_entities": [{"label": "SECRET", "pattern": r"\\{1000000000}"}]},
        {"custom_entities": [{"label": "SECRET", "pattern": "(x{1000}){1000}"}]},
        {"custom_entities": [{"label": "SECRET", "pattern": "(?x)x{ 10001 }"}]},
        {"custom_entities": [{"label": "SECRET", "pattern": "(?ix)x{ 10001 }"}]},
        {"custom_entities": [{"label": "SECRET", "pattern": "(?x:x{ 10001 })"}]},
        {"custom_entities": [{"label": "SECRET", "pattern": "(?V1x)x{ 10001 }"}]},
        {"custom_entities": [{"label": "SECRET", "pattern": "a{,10001}b"}]},
        {"custom_entities": [{"label": "SECRET", "pattern": "(a{,1000}){,1000}b"}]},
        {"custom_entities": [{"label": "bad label", "pattern": "SECRET"}]},
        {"allowlist": ["SECRET" * 100]},
        {"medical_terms": {"OTHER": ["SECRET"]}},
        {"medical_terms": {"CONDITION": [""]}},
    ],
)
def test_policy_errors_are_generic_and_exclude_submitted_data(policy):
    with pytest.raises(ValueError, match="^Invalid detection policy$") as error:
        validate_policy(policy)
    assert "SECRET" not in str(error.value)


def test_validation_rechecks_constructed_models():
    invalid = DetectionPolicy.model_construct(include=["UNDEFINED"])
    with pytest.raises(ValueError, match="Invalid detection policy"):
        validate_policy(invalid)


@pytest.mark.parametrize("pattern", ["(?x)a{ 10001 }", "a{,10001}b"])
def test_unsafe_repetition_is_rejected_before_pattern_compilation(monkeypatch, pattern):
    def unexpected_compile(*args, **kwargs):
        raise AssertionError("Untrusted expression reached compilation")

    monkeypatch.setattr("outis.policy.regex.compile", unexpected_compile)
    with pytest.raises(ValueError, match="Invalid detection policy"):
        validate_policy({"custom_entities": [{"label": "CUSTOM", "pattern": pattern}]})


def test_policy_metadata_never_exposes_custom_patterns_or_literal_values():
    policy = {
        "custom_entities": [{"label": "CUSTOM", "pattern": "private-secret"}],
        "blocklist": ["private-value"],
    }
    metadata = policy_metadata(policy)
    assert metadata["custom_entity_count"] == 1
    assert metadata["blocklist_count"] == 1
    assert "private" not in json.dumps(metadata)


def test_regex_timeout_fails_closed_without_partial_results():
    text = "a" * 50_000 + "!"
    started = time.monotonic()
    with pytest.raises(ValueError, match="Policy processing budget exceeded"):
        apply_policy(
            text,
            [Span(0, 1, "NAME")],
            {"custom_entities": [{"label": "CUSTOM", "pattern": "(a+)+$"}]},
        )
    assert time.monotonic() - started < 2


def test_zero_width_on_nonempty_text_fails_closed():
    with pytest.raises(ValueError, match="Empty custom matches"):
        apply_policy(
            "private",
            [],
            {"custom_entities": [{"label": "CUSTOM", "pattern": "(?=private)"}]},
        )


def test_count_limit_and_invalid_detector_offsets_fail_closed(monkeypatch):
    monkeypatch.setattr("outis.policy.MAX_POLICY_MATCHES", 2)
    with pytest.raises(ValueError, match="Detection count"):
        apply_policy(
            "xxx", [], {"custom_entities": [{"label": "CUSTOM", "pattern": "x"}]}
        )
    with pytest.raises(ValueError, match="Invalid detection offsets"):
        apply_policy("x", [Entity(0, 2, "NAME")])
    with pytest.raises(ValueError, match="Invalid detection output"):
        apply_policy("x", [object()])
