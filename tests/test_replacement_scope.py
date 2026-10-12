"""Patient and project replacement scopes, on invented values only."""

import hashlib
import importlib.util
import json
import logging
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from outis.api import create_app
from outis.entities import Entity
from outis.replacement.primitives import project_key
from outis.replacement.synthetic import synthetic_value
from outis.replacement.transform import ReplacementOptions, render
from outis.service import Deidentifier
from outis.types import Span

SECRET = b"outis-scope-regression-secret-0123456789"
TEXT = (
    "Dr. Mara Calder saw the patient at Exampleford Rehabilitation Hospital in "
    "Brindlemoor on 03/02/2024. Address 12 Quarry Lane. Call 555-867-5309 or "
    "mara.calder@example.org; portal https://portal.example.org/chart; host "
    "10.20.30.40. MRN A12345, SSN 123-45-6789, account 998877, ID ZX-41. Age 93. "
    "Calder followed up on 2024-03-09."
)
VALUES = [
    ("Mara Calder", "NAME"),
    ("Exampleford Rehabilitation Hospital", "FACILITY"),
    ("Brindlemoor", "LOCATION"),
    ("03/02/2024", "DATE"),
    ("12 Quarry Lane", "ADDRESS"),
    ("555-867-5309", "PHONE"),
    ("mara.calder@example.org", "EMAIL"),
    ("https://portal.example.org/chart", "URL"),
    ("10.20.30.40", "IP"),
    ("A12345", "MRN"),
    ("123-45-6789", "SSN"),
    ("998877", "ACCOUNT"),
    ("ZX-41", "ID"),
    ("93", "AGE"),
    ("Calder", "NAME"),
    ("2024-03-09", "DATE"),
]
NON_DATE = {
    "NAME",
    "FACILITY",
    "LOCATION",
    "ADDRESS",
    "PHONE",
    "EMAIL",
    "URL",
    "IP",
    "MRN",
    "SSN",
    "ACCOUNT",
    "ID",
}


def spans_for(text, values):
    spans, cursor = [], 0
    for value, label in values:
        start = text.index(value, cursor)
        spans.append(Entity(start, start + len(value), label))
        cursor = start + len(value)
    return spans


SPANS = spans_for(TEXT, VALUES)


def run(patient, *, secret=SECRET, project_id=None, **options):
    return render(
        TEXT,
        SPANS,
        options={"type": "synthetic", "entity_ids": "keyed", **options},
        mode="synthetic",
        patient_id=patient,
        secret=secret,
        project_id=project_id,
    )


def project(patient, project_id="trial-a", **kwargs):
    return run(patient, scope="project", project_id=project_id, **kwargs)


def ids(result, labels):
    return [
        entity["entity_id"]
        for entity in result["entities"]
        if entity["label"] in labels
    ]


def replacements(result, labels=NON_DATE):
    return [
        entity["replacement"]
        for entity in result["entities"]
        if entity["label"] in labels
    ]


def test_default_patient_scope_outputs_are_pinned():
    # Generated from origin/main before project scope existed; must never change.
    expected = {
        "patient-a": (
            "Dr. Casey Brook saw the patient at Synthetic Rehabilitation Hospital "
            "3E0AAF in Example City AE5A4A on 09/13/2023. Address 788 Synthetic Way, "
            "Example City, ZZ 00000. Call (202) 555-0100 or "
            "person.69619bce7148@example.invalid; portal "
            "https://synthetic-15f514e93aef.invalid; host 192.0.2.131. MRN "
            "SYN-MRN-85FFB8DB5E1E, SSN 000-34-7759, account "
            "SYN-ACCOUNT-82308E467990, ID SYN-ID-480BD8DD462C. Age 90+. Brook "
            "followed up on 2023-09-20.",
            "k908539699c3a20d8cfc4cd5557e69a28",
            "d01986b2161bdc09feb908b53e56553773838f2ce6601caa525b784701bedce9",
        ),
        "patient-b": (
            "Dr. Avery Cobalt saw the patient at Synthetic Rehabilitation Hospital "
            "BFE238 in Example City D16A61 on 04/22/2024. Address 615 Synthetic Way, "
            "Example City, ZZ 00000. Call (202) 555-0113 or "
            "person.c1fc57493751@example.invalid; portal "
            "https://synthetic-3b54a1a5162f.invalid; host 192.0.2.170. MRN "
            "SYN-MRN-11FE0B0A47A1, SSN 000-77-4659, account "
            "SYN-ACCOUNT-D81E6E8AA1D2, ID SYN-ID-1B9BBCF409CF. Age 90+. Cobalt "
            "followed up on 2024-04-29.",
            "kcfdbdf16b810a1a516a3920a405dbd75",
            "484dece76f431cc2a5ee93803ead6cdded057531bbce7add09abe919d0152cf9",
        ),
    }
    for patient, (text, first_id, ids_digest) in expected.items():
        for result in (run(patient), run(patient, scope="patient")):
            assert result["text"] == text
            ids = [entity["entity_id"] for entity in result["entities"]]
            assert ids[0] == ids[14] == first_id
            joined = ",".join(ids).encode()
            assert hashlib.sha256(joined).hexdigest() == ids_digest


def test_project_scope_shares_non_date_surrogates_and_ids_across_patients():
    first, second = project("patient-a"), project("patient-b")
    assert replacements(first) == replacements(second)
    assert ids(first, NON_DATE) == ids(second, NON_DATE)
    # One identity keeps one surrogate within the note as well.
    assert first["entities"][0]["replacement"].endswith(
        first["entities"][14]["replacement"]
    )
    assert first["preservation"]["patient_consistent_replacements"]
    for value, _ in VALUES:
        if value not in {"93"}:
            assert value not in first["text"]


def test_project_scope_shifts_dates_per_patient():
    first, second = project("patient-a"), project("patient-b")
    dates = [replacements(r, {"DATE"}) for r in (first, second)]
    assert dates[0] != dates[1]
    # A shared DATE ID would group one original date's differently shifted
    # copies across patients, and the bounded shift range would then pin the
    # original date. DATE IDs therefore stay per patient.
    assert not set(ids(first, {"DATE"})) & set(ids(second, {"DATE"}))
    assert ids(first, {"DATE"}) == ids(project("patient-a"), {"DATE"})
    assert all(value not in {"03/02/2024", "2024-03-09"} for value in dates[0])
    for result in (first, second):
        assert result["preservation"]["date_intervals_preserved"]
    # The per-patient offset is also project-specific, unlike the patient scope.
    assert dates[0] != replacements(run("patient-a"), {"DATE"})
    assert dates[0] != replacements(project("patient-a", "trial-b"), {"DATE"})
    # A caller-supplied offset still overrides the derived one.
    fixed = render(
        TEXT,
        SPANS,
        options={"type": "synthetic", "scope": "project"},
        mode="synthetic",
        patient_id="patient-a",
        secret=SECRET,
        date_shift_days=7,
        project_id="trial-a",
    )
    assert replacements(fixed, {"DATE"}) == ["03/09/2024", "2024-03-16"]
    with pytest.raises(ValueError, match="patient_id"):
        render(
            TEXT,
            SPANS,
            options={"type": "synthetic", "scope": "project"},
            mode="synthetic",
            secret=SECRET,
            project_id="trial-a",
        )


def test_different_project_ids_and_scopes_are_unlinkable():
    a, b, patient = (
        project("patient-a"),
        project("patient-a", "trial-b"),
        run("patient-a"),
    )
    for left, right in ((a, b), (a, patient)):
        for one, other in zip(replacements(left), replacements(right)):
            assert one != other
        assert not {e["entity_id"] for e in left["entities"]} & {
            e["entity_id"] for e in right["entities"]
        }
    assert project_key(SECRET, "trial-a") != project_key(SECRET, "trial-b")
    assert project_key(SECRET, "trial-a") != project_key(SECRET[::-1], "trial-a")


def test_project_scope_uses_wider_reserved_fictional_ranges():
    phones = replacements(project("patient-a"), {"PHONE"})
    ips = replacements(project("patient-a"), {"IP"})
    assert re.fullmatch(r"\([2-9][0-8]\d\) 555-01\d\d", phones[0])
    key = project_key(SECRET, "trial-a")
    many = {
        synthetic_value(f"555-867-{n:04d}", "PHONE", key, None, {}) for n in range(500)
    }
    assert all(re.fullmatch(r"\([2-9][0-8]\d\) 555-01\d\d", p) for p in many)
    assert not any(p[2:4] == "11" for p in many)
    assert len({p[:5] for p in many}) > 100
    assert re.fullmatch(r"(192\.0\.2|198\.51\.100|203\.0\.113)\.\d+", ips[0])
    address = replacements(project("patient-a"), {"ADDRESS"})[0]
    assert re.fullmatch(
        r"\d{3} Synthetic Way, Example City [0-9A-F]{6}, ZZ 00000", address
    )


@pytest.mark.parametrize(
    "options",
    [
        {"scope": "tenant"},
        {"scope": "study"},
        # The key comes from the request's project_id, not a replacement field.
        {"scope": "project", "project_id": "trial-a"},
        {"scope": "project", "study_id": "trial-a"},
    ],
)
def test_project_scope_options_are_validated(options):
    with pytest.raises(ValueError):
        ReplacementOptions.model_validate(options)


@pytest.mark.parametrize("project_id", [None, "", "trial a", "trial.a", "x" * 61, 7])
def test_project_scope_requires_a_valid_project_id(project_id):
    with pytest.raises(ValueError, match="project_id"):
        project("patient-a", project_id)


def test_patient_scope_ignores_project_id():
    assert run("patient-a", project_id="trial-a") == run("patient-a")


def test_alias_and_nickname_linking_work_in_project_scope():
    text = (
        "Mara Calder arrived. Mara was calm; Calder signed. "
        "Pip is the preferred nickname of Mira Voss."
    )
    values = [
        ("Mara Calder", "NAME"),
        ("Mara", "NAME"),
        ("Calder", "NAME"),
        ("Pip", "NAME"),
        ("Mira Voss", "NAME"),
    ]
    options = {"type": "synthetic", "scope": "project"}
    results = [
        render(
            text,
            spans_for(text, values),
            options=options,
            mode="synthetic",
            patient_id=patient,
            secret=SECRET,
            project_id="trial-a",
        )
        for patient in ("patient-a", "patient-b")
    ]
    assert replacements(results[0]) == replacements(results[1])
    full, first, last, pip, owner = replacements(results[0])
    assert full == f"{first} {last}"
    assert pip == owner.split()[0]
    assert (
        results[0]["entities"][0]["entity_id"] == results[0]["entities"][1]["entity_id"]
    )
    assert (
        results[0]["entities"][3]["entity_id"] == results[0]["entities"][4]["entity_id"]
    )


class Detector:
    def detect_batch(self, texts):
        return [
            [
                Span(m.start(), m.end(), "NAME" if m[0][0].isalpha() else "PHONE")
                for m in re.finditer(r"Mara Calder|Calder|\d{3}-\d{3}-\d{4}", text)
            ]
            for text in texts
        ]


@pytest.fixture
def service():
    return Deidentifier(detector=Detector(), secret=SECRET)


PROJECT = {"type": "synthetic", "entity_ids": "keyed", "scope": "project"}


def test_service_and_api_never_return_or_log_key_material(service, caplog):
    derived = project_key(SECRET, "trial-a")
    secrets = [SECRET, derived]
    text = "Mara Calder 555-867-5309. Calder agreed."
    caplog.set_level(logging.DEBUG)
    with TestClient(create_app(service)) as client:
        payloads = [
            client.post(
                "/v1/deidentify",
                json={
                    "text": text,
                    "mode": "synthetic",
                    "patient_id": patient,
                    "replacement": PROJECT,
                    "project_id": "trial-a",
                },
            )
            for patient in ("patient-a", "patient-b")
        ]
        # Without an explicit project_id, every caller would share "main".
        rejected = client.post(
            "/v1/deidentify",
            json={
                "text": text,
                "mode": "synthetic",
                "patient_id": "patient-a",
                "replacement": PROJECT,
            },
        )
        ready = client.get("/ready")
    assert all(response.status_code == 200 for response in payloads)
    assert rejected.status_code == 422
    bodies = [response.json() for response in payloads]
    assert [e["replacement"] for e in bodies[0]["entities"]] == [
        e["replacement"] for e in bodies[1]["entities"]
    ]
    exposed = (
        json.dumps(bodies)
        + rejected.text
        + ready.text
        + "".join(
            str(r) + "".join(map(str, r.__dict__.values())) for r in caplog.records
        )
    )
    for key in secrets:
        for encoding in (key.hex(), key.decode("latin-1"), repr(key)):
            assert encoding not in exposed
    assert "trial-a" not in exposed


def test_shared_batches_stay_within_one_patient_in_project_scope(service):
    items = [
        {
            "text": "Mara Calder arrived.",
            "mode": "synthetic",
            "patient_id": "patient-a",
            "replacement": PROJECT,
            "project_id": "trial-a",
        },
        {
            "text": "Calder returned.",
            "mode": "synthetic",
            "patient_id": "patient-b",
            "replacement": PROJECT,
            "project_id": "trial-a",
        },
    ]
    for option in ("shared_entities", "link_batch"):
        with pytest.raises(ValueError, match="patient scope"):
            service.deidentify_batch(items, **{option: True})
    same = [dict(item, patient_id="patient-a") for item in items]
    linked = service.deidentify_batch(same, shared_entities=True)
    full, alias = (result["entities"][0] for result in linked)
    assert full["entity_id"] == alias["entity_id"]
    assert full["replacement"].endswith(alias["replacement"])
    # Project-scoped keyed IDs agree with an unshared request for another patient.
    other = service.deidentify(
        "Mara Calder arrived.",
        mode="synthetic",
        patient_id="patient-b",
        replacement=PROJECT,
        project_id="trial-a",
    )
    assert other["entities"][0]["entity_id"] == full["entity_id"]
    assert other["entities"][0]["replacement"] == full["replacement"]


def test_service_project_scope_requires_an_explicit_project_id(service):
    request = {
        "text": "Mara Calder arrived.",
        "mode": "synthetic",
        "patient_id": "patient-a",
        "replacement": PROJECT,
    }
    with pytest.raises(ValueError, match="explicit project_id"):
        service.deidentify_batch([request])
    with pytest.raises(ValueError, match="explicit project_id"):
        service.deidentify(
            "Mara Calder arrived.",
            mode="synthetic",
            patient_id="patient-a",
            replacement=PROJECT,
        )
    # An explicit "main" is a deliberate choice and is keyed like any other.
    main = service.deidentify_batch([{**request, "project_id": "main"}])[0]
    other = service.deidentify_batch([{**request, "project_id": "trial-a"}])[0]
    assert main["entities"][0]["replacement"] != other["entities"][0]["replacement"]


def _collisions():
    spec = importlib.util.spec_from_file_location(
        "surrogate_collisions",
        Path(__file__).parents[1] / "scripts" / "surrogate_collisions.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_project_scope_collision_rates_match_documented_bounds():
    collisions = _collisions()
    rate = {
        label: collisions.collision_share(
            label, collisions.distinct_values(label, 2_000)
        )
        for label in ("NAME", "PHONE", "SSN", "MRN (8 digits)", "FACILITY", "IP")
    }
    assert rate["NAME"] < 0.03
    assert rate["PHONE"] < 0.05
    assert rate["SSN"] < 0.01
    assert rate["MRN (8 digits)"] == rate["FACILITY"] == 0
    # Only 762 documentation addresses exist; collisions are expected.
    assert rate["IP"] > 0.5
    patient_phone = collisions.collision_share(
        "PHONE", collisions.distinct_values("PHONE", 2_000), scope="patient"
    )
    assert patient_phone == 1.0
