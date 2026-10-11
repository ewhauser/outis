"""The HIPAA age boundary: 89 stays in place, 90 is removed.

Each case sits on one side of the boundary so that moving the comparison by a
single year (``> 88`` or ``> 90``) fails here, for both numeric and written
ages, in the policy, the replacement, and the HTTP response.
"""

import re

import pytest
from fastapi.testclient import TestClient

from outis.api import create_app
from outis.policy import apply_policy
from outis.replacement.surrogates import _age, _age_label
from outis.service import Deidentifier
from outis.types import Span

SECRET = b"test-only-secret-never-use-in-production-12345"
KEPT = ["89", "89-year-old", "89 y/o", "eighty-nine", "Eighty nine"]
REMOVED = ["90", "90-year-old", "90 y/o", "90+", "ninety", "Ninety"]


class AgeDetector:
    """Marks the value after ``Age:`` as an AGE span."""

    def detect_batch(self, texts):
        return [
            [
                Span(match.start(1), match.end(1), "AGE")
                for match in re.finditer(r"Age: ([^.]+)\.", text)
            ]
            for text in texts
        ]


def kept_values(text, ages=None):
    span = Span(0, len(text), "AGE", 1.0, "model")
    policy = None if ages is None else {"ages": ages}
    return [text[e.start : e.end] for e in apply_policy(text, [span], policy)]


@pytest.mark.parametrize("age", KEPT)
def test_policy_keeps_age_89(age):
    assert kept_values(age) == []


@pytest.mark.parametrize("age", REMOVED)
def test_policy_removes_age_90(age):
    assert kept_values(age) == [age]


@pytest.mark.parametrize("age", KEPT + REMOVED)
def test_all_ages_policy_removes_both_sides(age):
    assert kept_values(age, "all") == [age]


@pytest.mark.parametrize("age", KEPT)
def test_replacement_of_age_89_is_not_aggregated(age):
    # An 89 only reaches replacement under ages="all"; it is not "90 or older".
    assert _age_label(age) == "AGE"
    assert _age(age) == "[AGE]"


@pytest.mark.parametrize("age", REMOVED)
def test_replacement_of_age_90_is_aggregated(age):
    assert _age_label(age) == "AGE_90_PLUS"
    assert _age(age) == "90+"


@pytest.fixture
def client():
    service = Deidentifier(detector=AgeDetector(), secret=SECRET)
    with TestClient(create_app(service)) as client:
        yield client


def post(client, text, **request):
    response = client.post("/v1/deidentify", json={"text": text, **request})
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize("age", KEPT)
def test_http_releases_age_89(client, age):
    text = f"Age: {age}. Seen today."
    result = post(client, text)
    assert result["text"] == text
    assert result["entity_count"] == 0


@pytest.mark.parametrize("age", REMOVED)
@pytest.mark.parametrize(
    "mode,replacement", [("redact", "[AGE_90_PLUS]"), ("synthetic", "90+")]
)
def test_http_aggregates_age_90(client, age, mode, replacement):
    request = {"mode": mode}
    if mode == "synthetic":
        request["patient_id"] = "age-boundary-patient"
    result = post(client, f"Age: {age}. Seen today.", **request)
    assert result["text"] == f"Age: {replacement}. Seen today."
    assert [entity["label"] for entity in result["entities"]] == ["AGE"]


@pytest.mark.parametrize("age", KEPT)
def test_http_all_ages_policy_removes_age_89(client, age):
    result = post(client, f"Age: {age}. Seen today.", policy={"ages": "all"})
    assert result["text"] == "Age: [AGE]. Seen today."
