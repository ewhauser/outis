"""Site dictionaries: loading, matching, policy merge, service and API contracts.

Every facility, department and identifier below is fictional.
"""

import json
import random
import re
import time

import pytest
from fastapi.testclient import TestClient

from outis import api as api_module
from outis.api import create_app
from outis.operations import diagnostics
from outis.policy import apply_policy, policy_metadata, validate_policy
from outis.service import Deidentifier
from outis.site_dictionaries import (
    SiteDictionaries,
    SiteDictionaryError,
    load_site_dictionaries,
)
from outis.types import Span

SECRET = b"test-only-site-dictionary-secret-0123456789"
FACILITY = "Quillfeather Memorial Infirmary"
DEPARTMENT = "Moonwell Cardiology Annex"
DICTIONARIES = [
    {
        "name": "facilities",
        "label": "FACILITY",
        "terms": [FACILITY, "Brambleton Vale Clinic", "Zorvane Heights"],
    },
    {"name": "departments", "label": "DEPARTMENT", "terms": [DEPARTMENT]},
    {
        "name": "local-ids",
        "label": "ID",
        "terms": ["QX-"],
        "word_boundaries": False,
        "case_sensitive": True,
        "patterns": [r"\bQFM-\d{6}\b"],
    },
]
TERMS = [term for spec in DICTIONARIES for term in spec["terms"]]


def values(text, spans):
    return [(text[item.start : item.end], item.label) for item in spans]


def site(*specs):
    return load_site_dictionaries(list(specs) or DICTIONARIES)


class NameDetector:
    """Labels 'Mara Calder' as NAME, and optionally fixed extra spans."""

    def __init__(self, extra=()):
        self.extra = extra
        self.calls = []

    def detect_batch(self, texts):
        self.calls.append(list(texts))
        return [
            [
                Span(m.start(), m.end(), "NAME", 0.9, "test-detector")
                for m in re.finditer(r"Mara Calder", text)
            ]
            + [span for span in self.extra if span.end <= len(text)]
            for text in texts
        ]


@pytest.fixture
def service():
    return Deidentifier(
        detector=NameDetector(), secret=SECRET, site_dictionaries=DICTIONARIES
    )


# Loading and validation


def test_manifest_file_with_relative_terms_file(tmp_path):
    (tmp_path / "lists").mkdir()
    (tmp_path / "lists" / "facilities.txt").write_text(
        "# fictional sites\n\nBrambleton Vale Clinic\n  Zorvane   Heights  \n",
        encoding="utf-8",
    )
    manifest = tmp_path / "dictionaries.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "dictionaries": [
                    {
                        "name": "facilities",
                        "label": "FACILITY",
                        "terms": [FACILITY],
                        "terms_file": "lists/facilities.txt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_site_dictionaries(manifest)
    assert loaded.dictionaries[0].terms == (
        FACILITY,
        "Brambleton Vale Clinic",
        "Zorvane Heights",
    )
    summary = loaded.summary()
    assert summary["count"] == 1 and summary["term_count"] == 3
    assert summary["dictionaries"][0]["name"] == "facilities"
    assert re.fullmatch(r"[0-9a-f]{16}", summary["sha256"])
    # The same content loaded another way has the same hash.
    same = load_site_dictionaries(
        [
            {
                "name": "facilities",
                "label": "FACILITY",
                "terms": ["Zorvane Heights", FACILITY, "Brambleton Vale Clinic"],
            }
        ]
    )
    assert same.summary()["sha256"] == summary["sha256"]


def test_directory_of_dictionaries_named_after_files(tmp_path):
    (tmp_path / "b-facilities.json").write_text(
        json.dumps({"label": "FACILITY", "terms": [FACILITY]})
    )
    (tmp_path / "a-departments.json").write_text(
        json.dumps(
            {"name": "departments", "label": "DEPARTMENT", "terms": [DEPARTMENT]}
        )
    )
    (tmp_path / "notes.txt").write_text("ignored")
    loaded = load_site_dictionaries(tmp_path)
    assert loaded.names == ("departments", "b-facilities")


def test_none_and_loaded_instances_pass_through():
    assert load_site_dictionaries(None) is None
    loaded = site()
    assert load_site_dictionaries(loaded) is loaded
    assert isinstance(loaded, SiteDictionaries)


SECRET_TERM = "Xyloquent Private Sanatorium"


@pytest.mark.parametrize(
    "specs",
    [
        [],
        [{"name": "Bad Name", "label": "FACILITY", "terms": [SECRET_TERM]}],
        [{"name": "x", "label": "facility", "terms": [SECRET_TERM]}],
        [{"name": "x", "label": "CONDITION", "terms": [SECRET_TERM]}],
        [{"name": "x", "label": "FACILITY", "terms": [SECRET_TERM], "extra": 1}],
        [{"name": "x", "label": "FACILITY", "terms": []}],
        [{"name": "x", "label": "FACILITY", "terms": "Xyloquent"}],
        [{"name": "x", "label": "FACILITY", "terms": [SECRET_TERM, "   "]}],
        [{"name": "x", "label": "FACILITY", "terms": [SECRET_TERM, "--"]}],
        [{"name": "x", "label": "FACILITY", "terms": [SECRET_TERM, "Y" * 257]}],
        [{"name": "x", "label": "FACILITY", "terms": [SECRET_TERM, 7]}],
        [
            {
                "name": "x",
                "label": "FACILITY",
                "terms": [SECRET_TERM],
                "case_sensitive": "no",
            }
        ],
        [{"name": "x", "label": "ID", "patterns": ["(" + SECRET_TERM]}],
        [{"name": "x", "label": "ID", "patterns": [r"\d*"]}],
        [{"name": "x", "label": "ID", "patterns": [r"(?x)\d"]}],
        [{"name": "x", "label": "ID", "patterns": [r"\d{100000}"]}],
        [{"name": "x", "label": "ID", "patterns": [r"\d"] * 33}],
        [{"name": "x", "label": "ID", "terms_file": "/nonexistent/terms.txt"}],
        [
            {"name": "x", "label": "FACILITY", "terms": [SECRET_TERM]},
            {"name": "x", "label": "NAME", "terms": [SECRET_TERM]},
        ],
        [{"name": f"d{n}", "label": "ID", "terms": [SECRET_TERM]} for n in range(65)],
        {"version": 2, "dictionaries": []},
        {"dictionaries": "facilities"},
        42,
    ],
)
def test_invalid_configuration_fails_without_echoing_terms(specs):
    with pytest.raises(SiteDictionaryError) as error:
        load_site_dictionaries(specs)
    assert "Xyloquent" not in str(error.value)
    assert "Sanatorium" not in str(error.value)


def test_invalid_files_fail_without_echoing_content(tmp_path):
    with pytest.raises(SiteDictionaryError):
        load_site_dictionaries(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text('{"dictionaries": [' + SECRET_TERM)
    with pytest.raises(SiteDictionaryError) as error:
        load_site_dictionaries(broken)
    assert "Xyloquent" not in str(error.value)
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe" + SECRET_TERM.encode("utf-16-le"))
    with pytest.raises(SiteDictionaryError):
        load_site_dictionaries(
            [{"name": "x", "label": "ID", "terms_file": str(tmp_path / "bad.txt")}]
        )
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SiteDictionaryError):
        load_site_dictionaries(empty)


def test_invalid_configuration_fails_at_construction_before_model_load():
    with pytest.raises(SiteDictionaryError):
        Deidentifier(site_dictionaries=[{"name": "x", "label": "ID"}])


# Matching


def test_word_boundaries_case_and_whitespace():
    loaded = site()
    text = (
        "Seen at quillfeather memorial\ninfirmary; not Brambleton Vale Clinics "
        "or xZorvane Heights. Refs QX-77 and abQX-12 but not qx-5."
    )
    hits = [
        (text[start:end], loaded.names[index])
        for start, end, index in loaded.find(text, loaded.select())
    ]
    assert hits == [
        ("quillfeather memorial\ninfirmary", "facilities"),
        ("QX-", "local-ids"),
        ("QX-", "local-ids"),
    ]


def test_case_sensitive_dictionary_requires_exact_case():
    loaded = site(
        {
            "name": "exact",
            "label": "FACILITY",
            "terms": [FACILITY],
            "case_sensitive": True,
        }
    )
    text = f"{FACILITY.upper()} then {FACILITY}"
    assert [
        text[start:end] for start, end, _ in loaded.find(text, loaded.select())
    ] == [FACILITY]


def test_longest_match_wins_and_contained_terms_are_dropped():
    loaded = site(
        {
            "name": "facilities",
            "label": "FACILITY",
            "terms": [
                "Zorvane",
                "Zorvane Heights",
                "Heights Annex",
                "Zorvane Heights Annex",
            ],
        }
    )
    text = "Moved from Zorvane Heights Annex to Zorvane."
    assert [
        text[start:end] for start, end, _ in loaded.find(text, loaded.select())
    ] == ["Zorvane Heights Annex", "Zorvane"]


def test_offsets_survive_case_folding_that_changes_length():
    loaded = site(
        {"name": "f", "label": "FACILITY", "terms": ["Brambleton Vale Clinic"]}
    )
    # "İ".lower() has two characters; offsets must still index the original.
    text = "İİ Brambleton Vale Clinic ΟΔΟΣ"
    hits = loaded.find(text, loaded.select())
    assert [text[start:end] for start, end, _ in hits] == ["Brambleton Vale Clinic"]
    sigma = site({"name": "f", "label": "LOCATION", "terms": ["οδος"]})
    assert [text[s:e] for s, e, _ in sigma.find(text, sigma.select())] == ["ΟΔΟΣ"]


def test_identical_terms_keep_the_first_configured_dictionary():
    loaded = site(
        {"name": "first", "label": "FACILITY", "terms": ["Zorvane Heights"]},
        {"name": "second", "label": "LOCATION", "terms": ["zorvane heights"]},
    )
    text = "At Zorvane Heights."
    assert values(text, apply_policy(text, [], None, loaded)) == [
        ("Zorvane Heights", "FACILITY")
    ]


# Policy merge and priority


def test_dictionary_spans_use_site_source_and_flow_through_default_policy():
    text = f"Mara Calder transferred to {FACILITY}."
    spans = [Span(0, 11, "NAME", 0.9, "model")]
    result = apply_policy(text, spans, None, site())
    assert values(text, result) == [("Mara Calder", "NAME"), (FACILITY, "FACILITY")]
    assert [item.source for item in result] == [
        "model",
        "site_dictionary:facilities",
    ]
    assert result[1].score == 1.0


def test_dictionary_label_overrides_model_and_merges_overlaps():
    text = f"Arrived from {FACILITY} East Wing."
    start = text.index(FACILITY)
    # The model saw a partial NAME inside the facility and a longer suffix.
    model = [
        Span(start, start + 12, "NAME", 0.99, "model"),
        Span(start + 13, len(text) - 1, "LOCATION", 0.8, "model"),
    ]
    result = apply_policy(text, model, None, site())
    assert values(text, result) == [(f"{FACILITY} East Wing", "FACILITY")]
    assert result[0].source == "site_dictionary:facilities"


def test_priority_is_below_blocklist_custom_and_extended():
    text = f"Visit {FACILITY} today"
    for policy, label in (
        ({"blocklist": ["Memorial"]}, "ID"),
        ({"custom_entities": [{"label": "SITE", "pattern": "Quillfeather"}]}, "SITE"),
        ({"medical_terms": {"CONDITION": ["Memorial Infirmary"]}}, "FACILITY"),
    ):
        result = apply_policy(text, [], policy, site())
        assert values(text, result) == [(FACILITY, label)]


def test_allowlist_profile_include_exclude_and_selection():
    text = f"{FACILITY} {DEPARTMENT} QFM-123456"
    loaded = site()
    assert values(text, apply_policy(text, [], None, loaded)) == [
        (FACILITY, "FACILITY"),
        (DEPARTMENT, "DEPARTMENT"),
        ("QFM-123456", "ID"),
    ]
    assert values(text, apply_policy(text, [], {"allowlist": [FACILITY]}, loaded)) == [
        (DEPARTMENT, "DEPARTMENT"),
        ("QFM-123456", "ID"),
    ]
    # identifiers_only omits FACILITY; a custom-labelled dictionary stays on.
    assert values(
        text, apply_policy(text, [], {"profile": "identifiers_only"}, loaded)
    ) == [(DEPARTMENT, "DEPARTMENT"), ("QFM-123456", "ID")]
    assert values(text, apply_policy(text, [], {"exclude": ["FACILITY"]}, loaded)) == [
        (DEPARTMENT, "DEPARTMENT"),
        ("QFM-123456", "ID"),
    ]
    # include replaces the enabled set; custom site labels cannot be named.
    assert values(text, apply_policy(text, [], {"include": ["ID"]}, loaded)) == [
        ("QFM-123456", "ID")
    ]
    assert apply_policy(text, [], {"site_dictionaries": []}, loaded) == []
    assert values(
        text, apply_policy(text, [], {"site_dictionaries": ["departments"]}, loaded)
    ) == [(DEPARTMENT, "DEPARTMENT")]
    with pytest.raises(ValueError, match="Unknown site dictionary"):
        apply_policy(text, [], {"site_dictionaries": ["nope"]}, loaded)
    with pytest.raises(ValueError, match="Unknown site dictionary"):
        apply_policy(text, [], {"site_dictionaries": ["facilities"]})
    with pytest.raises(ValueError):
        validate_policy({"site_dictionaries": ["Not Valid"]})


def test_policy_metadata_reports_names_never_terms():
    loaded = site()
    assert policy_metadata(None, loaded)["site_dictionaries"] == [
        "facilities",
        "departments",
        "local-ids",
    ]
    assert policy_metadata({"site_dictionaries": ["local-ids"]}, loaded)[
        "site_dictionaries"
    ] == ["local-ids"]
    assert policy_metadata({"site_dictionaries": []})["site_dictionaries"] == []
    assert "site_dictionaries" not in policy_metadata(None)


def test_dictionary_matches_count_toward_match_limit():
    loaded = site({"name": "f", "label": "FACILITY", "terms": ["Zorvane"]})
    with pytest.raises(ValueError, match="Detection count"):
        apply_policy("Zorvane " * 10_001, [], None, loaded)


# Service


def test_detect_and_redact_through_the_service(service):
    text = f"Mara Calder seen in {DEPARTMENT} at {FACILITY}; ref QFM-123456."
    detected = service.detect(text)
    assert [
        (text[e["start"] : e["end"]], e["label"], e["source"])
        for e in detected["entities"]
    ] == [
        ("Mara Calder", "NAME", "test-detector"),
        (DEPARTMENT, "DEPARTMENT", "site_dictionary:departments"),
        (FACILITY, "FACILITY", "site_dictionary:facilities"),
        ("QFM-123456", "ID", "site_dictionary:local-ids"),
    ]
    assert detected["policy"]["site_dictionaries"] == [
        "facilities",
        "departments",
        "local-ids",
    ]
    redacted = service.deidentify(text)
    assert redacted["text"] == "[NAME] seen in [DEPARTMENT] at [FACILITY]; ref [ID]."
    assert [e["replacement"] for e in redacted["entities"]][1:] == [
        "[DEPARTMENT]",
        "[FACILITY]",
        "[ID]",
    ]


def test_synthetic_replacement_of_dictionary_spans(service):
    text = f"Transferred from {FACILITY}. {FACILITY} sent records."
    first = service.deidentify(text, mode="synthetic", patient_id="patient-1")
    again = service.deidentify(text, mode="synthetic", patient_id="patient-1")
    assert first == again
    facilities = [e for e in first["entities"] if e["label"] == "FACILITY"]
    assert len(facilities) == 2
    assert facilities[0]["replacement"] == facilities[1]["replacement"]
    assert facilities[0]["replacement"] != FACILITY
    assert FACILITY not in first["text"]
    custom = service.deidentify(
        f"In {DEPARTMENT}.",
        replacement={"type": "synthetic", "per_type": {"DEPARTMENT": "mask"}},
        patient_id="patient-1",
    )
    assert custom["text"] == "In " + "*" * len(DEPARTMENT) + "."


def test_service_selection_errors_before_inference(service):
    with pytest.raises(ValueError, match="Unknown site dictionary"):
        service.detect("Mara Calder", policy={"site_dictionaries": ["nope"]})
    assert service.detector.calls == []
    plain = Deidentifier(detector=NameDetector())
    with pytest.raises(ValueError, match="Unknown site dictionary"):
        plain.detect("Mara Calder", policy={"site_dictionaries": ["facilities"]})


def test_linked_batches_and_structured_fields_use_dictionaries(service):
    results = service.deidentify_batch(
        [{"text": f"At {FACILITY}"}, {"text": "Then Zorvane Heights"}],
        link_batch=True,
    )
    assert [r["text"] for r in results] == ["At [FACILITY]", "Then [FACILITY]"]
    from outis.formats.structured import process_structured

    output = process_structured(service, {"site": f"{FACILITY}", "note": "none"})
    assert output["data"] == {"site": "[FACILITY]", "note": "none"}


def test_responses_never_contain_dictionary_terms(service):
    text = f"Mara Calder at {FACILITY} in {DEPARTMENT}, ref QFM-123456 QX-9."
    payloads = [
        service.detect(text),
        service.deidentify(text, policy={"profile": "clinical_default"}),
        service.deidentify(text, mode="synthetic", patient_id="p-1"),
        service.analyze(text),
        service.site_dictionaries.summary(),
        diagnostics(service),
    ]
    serialized = json.dumps(payloads, ensure_ascii=False)
    for term in [*TERMS, "QFM-123456", r"\bQFM"]:
        assert term not in serialized
    with TestClient(create_app(service, api_key="k")) as client:
        for path in ("/ready", "/diagnostics"):
            body = client.get(path, headers={"X-API-Key": "k"}).text
            assert all(term not in body for term in TERMS)
            assert "QFM" not in body


# Off by default


def test_without_dictionaries_outputs_are_unchanged():
    text = f"Mara Calder transferred to {FACILITY}."
    configured_off = Deidentifier(detector=NameDetector(), secret=SECRET)
    explicit_none = Deidentifier(
        detector=NameDetector(), secret=SECRET, site_dictionaries=None
    )
    assert configured_off.site_dictionaries is None
    for instance in (configured_off, explicit_none):
        assert instance.detect(text) == {
            "entities": [
                {
                    "start": 0,
                    "end": 11,
                    "label": "NAME",
                    "score": 0.9,
                    "source": "test-detector",
                    "entity_id": instance.detect(text)["entities"][0]["entity_id"],
                }
            ],
            "entity_count": 1,
            "model_version": "injected-detector",
            "policy": {
                "profile": "clinical_default",
                "version": "1",
                "custom_entity_count": 0,
                "allowlist_count": 0,
                "blocklist_count": 0,
                "medical_dictionary_labels": [],
                "ages": "over_89",
            },
        }
        legacy = instance.deidentify(text)
        assert legacy["text"] == f"[NAME] transferred to {FACILITY}."
        assert "policy" not in legacy
        assert "site_dictionaries" not in diagnostics(instance)
    assert json.dumps(configured_off.deidentify(text), sort_keys=True) == json.dumps(
        explicit_none.deidentify(text), sort_keys=True
    )
    with TestClient(create_app(configured_off)) as client:
        assert client.get("/ready").json() == {
            "status": "ready",
            "model_version": "injected-detector",
            "synthetic_enabled": True,
        }
        assert "site_dictionaries" not in client.get("/diagnostics").json()


def test_disabled_selection_matches_unconfigured_entities(service):
    text = f"Mara Calder transferred to {FACILITY}."
    plain = Deidentifier(detector=NameDetector(), secret=SECRET)
    off = service.deidentify(text, policy={"site_dictionaries": []})
    reference = plain.deidentify(text, policy={})
    assert off["text"] == reference["text"]
    assert off["entities"] == reference["entities"]
    assert off["policy"].pop("site_dictionaries") == []
    assert off["policy"] == reference["policy"]


# API wiring


def test_ready_and_diagnostics_report_summary(service):
    with TestClient(create_app(service)) as client:
        ready = client.get("/ready").json()
        assert ready["site_dictionaries"] == service.site_dictionaries.summary()
        assert ready["site_dictionaries"]["dictionaries"][0] == {
            "name": "facilities",
            "label": "FACILITY",
            "term_count": 3,
            "pattern_count": 0,
            "case_sensitive": False,
            "word_boundaries": True,
            "sha256": service.site_dictionaries.dictionaries[0].digest,
        }
        assert (
            client.get("/diagnostics").json()["site_dictionaries"]
            == ready["site_dictionaries"]
        )
        response = client.post(
            "/v1/detect",
            json={"text": "x", "policy": {"site_dictionaries": ["nope"]}},
        )
        assert response.status_code == 422
        invalid = client.post(
            "/v1/detect", json={"text": "x", "policy": {"site_dictionaries": [1]}}
        )
        assert invalid.status_code == 422
        assert invalid.json()["errors"][0]["loc"][:3] == [
            "body",
            "policy",
            "site_dictionaries",
        ]


def test_server_reads_site_dictionaries_from_environment(tmp_path, monkeypatch):
    manifest = tmp_path / "site.json"
    manifest.write_text(json.dumps({"dictionaries": DICTIONARIES}))
    captured = {}

    def factory(model_dir, **options):
        captured.update(options)
        return Deidentifier(
            detector=NameDetector(),
            secret=options["secret"],
            site_dictionaries=options.get("site_dictionaries"),
        )

    monkeypatch.setattr(api_module, "Deidentifier", factory)
    monkeypatch.setenv("OUTIS_SITE_DICTIONARIES", str(manifest))
    monkeypatch.setenv("OUTIS_SECRET", SECRET.decode())
    with TestClient(create_app()) as client:
        assert client.get("/ready").json()["site_dictionaries"]["count"] == 3
    assert captured["site_dictionaries"] == str(manifest)
    captured.clear()
    monkeypatch.setenv("OUTIS_SITE_DICTIONARIES", "")
    with TestClient(create_app()) as client:
        assert "site_dictionaries" not in client.get("/ready").json()
    assert "site_dictionaries" not in captured


# Cost


def _fictional_terms(count, seed=0):
    rng = random.Random(seed)
    syllables = ["zor", "vel", "quin", "bram", "tor", "lux", "nev", "ost", "pell"]
    kinds = ["Clinic", "Infirmary", "Medical Center", "Care Annex", "Pavilion"]

    def word():
        return "".join(rng.choice(syllables) for _ in range(3)).title()

    terms = set()
    while len(terms) < count:
        terms.add(f"{word()} {word()} {rng.choice(kinds)}")
    return sorted(terms)


def test_request_cost_does_not_grow_with_term_count():
    note = (
        "Pt seen in follow-up. BP 132/84, HR 72. Continue lisinopril 10 mg daily; "
        "recheck BMP in 2 weeks. Discussed diet and exercise.\n"
    ) * 800
    note = note[:100_000]
    timings = {}
    for count in (10, 100_000):
        terms = _fictional_terms(count)
        loaded = load_site_dictionaries(
            [{"name": "facilities", "label": "FACILITY", "terms": terms}]
        )
        text = note[:50_000] + " " + terms[0] + " " + note[50_000:]
        selected = loaded.select()
        started = time.perf_counter()
        for _ in range(3):
            hits = loaded.find(text, selected)
        timings[count] = (time.perf_counter() - started) / 3
        assert [text[start:end] for start, end, _ in hits] == [terms[0]]
    # Typically ~10 ms either way; the bounds only catch term-proportional work.
    assert timings[100_000] < 1.0
    assert timings[100_000] < 5 * timings[10] + 0.05
