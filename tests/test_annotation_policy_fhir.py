import json

from clinical_deid.data.annotation_policy import POLICY_VERSION, annotation_errors
from clinical_deid.data.annotation_policy_fhir import FHIR_FIXTURES


def row(key):
    return {r["id"]: r for r in FHIR_FIXTURES}[key]


def gold(key):
    r = row(key)
    return [(s["label"], r["text"][s["start"] : s["end"]]) for s in r["spans"]]


def test_fhir_fixtures_are_valid_json_with_valid_offsets():
    for r in FHIR_FIXTURES:
        assert r["annotation_policy"] == POLICY_VERSION
        assert not annotation_errors(r["text"], r["spans"])
        json.loads(r["text"])


def test_fhir_decisions():
    assert gold("fhir_ids_and_systems") == [
        ("URL", "https://fhir.example.test/Patient/pt-77"),
        ("ID", "pt-77"),
        ("MRN", "MRN-7720931"),
    ]
    assert gold("fhir_reference") == [("ID", "pt-77")]
    assert gold("fhir_datetime") == [("DATE", "2026-03-14"), ("DATE", "1941-03-03")]
    assert gold("fhir_name_address_parts") == [
        ("NAME", "Brannock"),
        ("NAME", "Odalys"),
        ("NAME", "Q"),
        ("ADDRESS", "412 Wenlow Lane"),
        ("ADDRESS", "Apt 2B"),
        ("LOCATION", "Corriston"),
        ("LOCATION", "53728"),
    ]
    assert gold("fhir_narrative") == [
        ("NAME", "Odalys Brannock"),
        ("DATE", "3 March 1941"),
    ]
