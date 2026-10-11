"""Exercise extension boundaries through the real service without a model load."""

import hashlib
import json
import re

import pytest

from outis.entities import Entity
from outis.formats.structured import process_structured
from outis.policy import DetectionPolicy
from outis.replacement.transform import ReplacementOptions, reidentify
from outis.service import Deidentifier
from outis.types import ENTITY_TYPES, Span

SECRET = b"test-only-service-extension-secret-0123456789"
MODEL_LABELS = (
    "NAME",
    "DATE",
    "ADDRESS",
    "PHONE",
    "EMAIL",
    "MRN",
    "SSN",
    "ACCOUNT",
    "ID",
    "URL",
    "IP",
    "AGE",
    "LOCATION",
    "FACILITY",
)


class RecordingDetector:
    def __init__(self, find=None):
        self.calls = []
        self.find = find or (
            lambda text: [
                Span(match.start(), match.end(), "NAME", 0.91, "test-detector")
                for match in re.finditer(
                    r"Mara Calder|Mara|Calder|Zoë Núñez|Robert Jones", text
                )
            ]
        )

    def detect_batch(self, texts):
        self.calls.append(list(texts))
        return [self.find(text) for text in texts]


@pytest.fixture
def service():
    return Deidentifier(detector=RecordingDetector(), secret=SECRET)


def positions(result):
    return [
        (entity["start"], entity["end"], entity["label"])
        for entity in result["entities"]
    ]


def test_detection_and_deidentification_share_policy_and_unicode_offsets(service):
    text = "🩺 Zoë Núñez: STUDY-482901, Mara Calder; glucose 109 mg/dL."
    policy = {
        "allowlist": ["Mara Calder"],
        "custom_entities": [{"label": "STUDY_ID", "pattern": r"STUDY-\d{6}"}],
    }
    detected = service.detect(text, policy=policy)
    rendered = service.deidentify(text, policy=policy)
    assert (
        positions(detected)
        == positions(rendered)
        == [
            (2, 11, "NAME"),
            (13, 25, "STUDY_ID"),
        ]
    )
    assert rendered["text"] == "🩺 [NAME]: [STUDY_ID], Mara Calder; glucose 109 mg/dL."
    assert [entity["source"] for entity in detected["entities"]] == [
        "test-detector",
        "policy:custom",
    ]
    for entity in rendered["entities"]:
        assert (
            rendered["text"][entity["processed_start"] : entity["processed_end"]]
            == entity["replacement"]
        )
    assert detected["policy"] == rendered["policy"]
    assert "STUDY-482901" not in json.dumps(detected)
    assert "Zoë Núñez" not in json.dumps(detected, ensure_ascii=False)


def test_policy_overlap_unions_complete_coverage_in_both_paths():
    text = "AA-123456-tail"
    detector = RecordingDetector(lambda value: [Span(0, 8, "ID")])
    service = Deidentifier(detector=detector)
    policy = {"custom_entities": [{"label": "STUDY_ID", "pattern": "123456-tail"}]}
    detected = service.detect(text, policy=policy)
    rendered = service.deidentify(text, policy=policy)
    assert positions(detected) == positions(rendered) == [(0, len(text), "STUDY_ID")]
    assert rendered["text"] == "[STUDY_ID]"


def test_linked_detection_uses_context_and_independent_detection_does_not():
    def contextual_names(text):
        return (
            [
                Span(match.start(), match.end(), "NAME")
                for match in re.finditer("Calder", text)
            ]
            if "Patient family:" in text
            else []
        )

    detector = RecordingDetector(contextual_names)
    service = Deidentifier(detector=detector)
    items = [{"text": "Patient family:"}, {"text": "🩺 Calder arrived."}]
    independent = service.detect_batch(items)
    linked = service.detect_batch(items, link_batch=True)
    assert [result["entity_count"] for result in independent] == [0, 0]
    assert positions(linked[1]) == [(2, 8, "NAME")]
    assert detector.calls == [
        [item["text"] for item in items],
        ["Patient family:\n🩺 Calder arrived."],
    ]


@pytest.mark.parametrize("mode", ["redact", "synthetic"])
def test_linked_cross_boundary_entity_is_clipped_and_redacted_with_local_offsets(mode):
    def across_separator(text):
        if "Mara\nCalder" in text:
            start = text.index("Mara")
            return [Span(start, start + len("Mara\nCalder"), "NAME")]
        return []

    service = Deidentifier(detector=RecordingDetector(across_separator), secret=SECRET)
    items = [
        {"text": text, "mode": mode, "patient_id": "patient"}
        for text in ["🩺 Mara", "Calder is stable."]
    ]
    results = service.deidentify_batch(items, link_batch=True)
    assert [positions(result) for result in results] == [
        [(2, 6, "NAME")],
        [(0, 6, "NAME")],
    ]
    assert [result["text"] for result in results] == ["🩺 [NAME]", "[NAME] is stable."]
    assert {
        entity["source"] for result in results for entity in result["entities"]
    } == {"context_fragment"}
    assert len({result["entities"][0]["entity_id"] for result in results}) == 1
    for result in results:
        entity = result["entities"][0]
        assert (
            result["text"][entity["processed_start"] : entity["processed_end"]]
            == "[NAME]"
        )


@pytest.mark.parametrize("link_batch", [False, True])
def test_full_name_and_reduced_alias_share_synthetic_identity_across_segments(
    service, link_batch
):
    texts = ["Mara Calder", "Mara", "Calder"]
    requests = [
        {"text": text, "mode": "synthetic", "patient_id": "same-patient"}
        for text in texts
    ]
    results = service.process_segments(requests, link_batch=link_batch)
    full, first, last = [result["text"] for result in results]
    assert full == first + " " + last
    assert [full, first, last] != texts
    assert len({result["entities"][0]["entity_id"] for result in results}) == 1
    assert results == service.process_segments(requests, link_batch=link_batch)


def test_shared_identity_namespace_survives_more_than_one_inference_batch(service):
    texts = ["Mara Calder"] * 17 + ["Robert Jones", "Mara", "Calder"]
    result = process_structured(
        service, texts, options={"replacement": {"pattern": "[{label}_{id}]"}}
    )
    assert result["data"] == ["[NAME_e1]"] * 17 + [
        "[NAME_e2]",
        "[NAME_e1]",
        "[NAME_e1]",
    ]
    assert [len(batch) for batch in service.detector.calls] == [16, 4]
    assert len(result["entities"]) == 20
    assert result["entities"][17]["entity_id"] == "e2"
    assert result["entities"][18]["location"] == {
        "format": "json",
        "path": [18],
        "part": "value",
    }


@pytest.mark.parametrize("context", [{"link_batch": True}, {"shared_entities": True}])
@pytest.mark.parametrize(
    "different",
    [
        {"patient_id": "another-patient"},
        {"project_id": "another-project"},
        {"policy": {"allowlist": ["Mara Calder"]}},
        {"replacement": {"type": "mask"}},
        {"mode": "synthetic"},
    ],
)
def test_context_scope_mismatch_fails_before_inference(service, context, different):
    first = {"text": "Mara Calder", "patient_id": "patient", "project_id": "study"}
    second = {**first, **different}
    with pytest.raises(ValueError, match="Shared context"):
        service.deidentify_batch([first, second], **context)
    assert service.detector.calls == []


@pytest.mark.parametrize(
    "different",
    [
        {"project_id": "other"},
        {"policy": {"exclude": ["NAME"]}},
    ],
)
def test_detect_scope_mismatch_fails_before_inference(service, different):
    with pytest.raises(ValueError, match="Shared context"):
        service.detect_batch(
            [{"text": "Mara Calder"}, {"text": "Mara", **different}], link_batch=True
        )
    assert service.detector.calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"policy": {"custom_entities": [{"label": "STUDY_ID", "pattern": "("}]}},
        {"policy": {"custom_entities": [{"label": "STUDY_ID", "pattern": "a*"}]}},
        {
            "policy": {
                "custom_entities": [{"label": "STUDY_ID", "pattern": "a{999999999}"}]
            }
        },
        {"policy": {"include": ["UNREGISTERED_LABEL"]}},
        {"replacement": {"pattern": "{label.__class__}"}},
        {"replacement": {"mask_character": "\n"}},
        {"replacement": {"include_mapping": "yes"}},
        {"replacement": {"entity_ids": "keyed"}},
        {"replacement": {"type": "synthetic"}},
    ],
)
def test_invalid_policy_or_replacement_fails_before_inference(service, changes):
    with pytest.raises(ValueError):
        service.deidentify("SENSITIVE-INPUT-CANARY", **changes)
    assert service.detector.calls == []


@pytest.mark.parametrize(
    "replacement", [{"type": "synthetic"}, {"entity_ids": "keyed"}]
)
@pytest.mark.parametrize("secret", [None, b"short"])
def test_invalid_secret_fails_before_inference(replacement, secret):
    detector = RecordingDetector()
    service = Deidentifier(detector=detector, secret=secret)
    with pytest.raises(ValueError, match="secret"):
        service.deidentify("Mara Calder", replacement=replacement, patient_id="patient")
    assert detector.calls == []


def test_constructed_invalid_policy_is_revalidated_before_inference(service):
    policy = DetectionPolicy.model_construct(profile="unsafe-profile")
    with pytest.raises(ValueError, match="Invalid detection policy"):
        service.detect("Mara Calder", policy=policy)
    assert service.detector.calls == []


def test_constructed_invalid_replacement_is_revalidated_before_inference(service):
    replacement = ReplacementOptions.model_construct(pattern="{private}")
    with pytest.raises(ValueError, match="Invalid replacement options"):
        service.deidentify("Mara Calder", replacement=replacement)
    assert service.detector.calls == []


def test_extended_and_custom_labels_do_not_expand_trained_vocabulary():
    detector = RecordingDetector(lambda text: [])
    service = Deidentifier(detector=detector)
    policy = {
        "profile": "extended",
        "custom_entities": [{"label": "STUDY_ID", "pattern": "STUDY-123"}],
    }
    result = service.detect("passport: AB234567; STUDY-123", policy=policy)
    assert [entity["label"] for entity in result["entities"]] == [
        "PASSPORT",
        "STUDY_ID",
    ]
    assert ENTITY_TYPES == MODEL_LABELS
    with pytest.raises(ValueError, match="Invalid PHI span"):
        Span(0, 2, "STUDY_ID")
    assert Entity(0, 2, "STUDY_ID").label == "STUDY_ID"


def test_shared_context_marker_collision_with_another_segment_is_rejected(service):
    requests = [
        {"text": text, "replacement": {"include_mapping": True}}
        for text in ["Mara Calder", "Literal [NAME_e1] already exists."]
    ]
    with pytest.raises(ValueError, match="collides"):
        service.process_segments(requests)


def test_shared_mapping_cannot_assign_same_marker_to_different_originals(service):
    requests = [
        {"text": text, "replacement": {"include_mapping": True}}
        for text in ["Mara Calder", "Mara"]
    ]
    with pytest.raises(ValueError, match="[Aa]mbiguous|[Dd]uplicate"):
        service.process_segments(requests)


def test_distinct_markers_across_segments_produce_combinable_restoration_maps(service):
    requests = [
        {"text": text, "replacement": {"include_mapping": True}}
        for text in ["Mara Calder", "Robert Jones"]
    ]
    results = service.process_segments(requests)
    restored = reidentify(
        "\n".join(result["text"] for result in results),
        [mapping for result in results for mapping in result["mappings"]],
        restore_sensitive=True,
    )
    assert restored == {"text": "Mara Calder\nRobert Jones", "restored_count": 2}


def test_global_mention_indices_make_reversible_aliases_distinct_across_fields(service):
    original = ["Mara Calder", "Mara", "Calder"]
    result = process_structured(
        service,
        original,
        options={
            "replacement": {
                "include_mapping": True,
                "pattern": "[{label}_{id}_{index}]",
            },
        },
    )
    assert result["data"] == ["[NAME_e1_1]", "[NAME_e1_2]", "[NAME_e1_3]"]
    assert len(result["mappings"]) == 3
    restored = reidentify(
        "\n".join(result["data"]), result["mappings"], restore_sensitive=True
    )
    assert restored == {"text": "\n".join(original), "restored_count": 3}


def test_structured_repeated_identity_has_one_unambiguous_restoration_mapping(service):
    result = process_structured(
        service,
        ["Mara Calder"] * 20,
        options={
            "replacement": {"include_mapping": True},
        },
    )
    assert result["mappings"] == [
        {"marker": "[NAME_e1]", "original": "Mara Calder", "label": "NAME"}
    ]
    assert (
        reidentify(
            " ".join(result["data"]), result["mappings"], restore_sensitive=True
        )["restored_count"]
        == 20
    )


def test_policy_allowlist_does_not_survive_into_subsequent_requests(service):
    first = service.deidentify(
        "Mara Calder", policy={"allowlist": ["Mara Calder"]}, project_id="first"
    )
    second = service.deidentify("Mara Calder", project_id="second")
    assert first["text"] == "Mara Calder"
    assert second["text"] == "[NAME]"
    assert len(service.detector.calls) == 2


def test_invalid_later_batch_item_prevents_any_inference(service):
    with pytest.raises(ValueError, match="Invalid detection policy"):
        service.deidentify_batch(
            [
                {"text": "Mara Calder"},
                {"text": "Robert Jones", "policy": {"include": ["UNREGISTERED"]}},
            ]
        )
    assert service.detector.calls == []


def test_output_expansion_is_bounded_across_shared_segments():
    detector = RecordingDetector(
        lambda text: [
            Span(match.start(), match.end(), "ID") for match in re.finditer("a", text)
        ]
    )
    service = Deidentifier(detector=detector)
    requests = [{"text": "a " * 3000, "replacement": {"pattern": "X" * 256}}] * 2
    with pytest.raises(ValueError, match="[Oo]utput.*limit|[Oo]utput.*bound"):
        service.process_segments(requests)


def test_usage_tracks_success_and_failure_without_sensitive_identifiers(service):
    project, patient, canary = (
        "private-project-canary",
        "private-patient-canary",
        "SENSITIVE-NOTE-CANARY",
    )
    service.deidentify(f"Mara Calder {canary}", project_id=project, patient_id=patient)

    def fail(text):
        assert service.usage.snapshot()["active_requests"] == 1
        raise RuntimeError("detector unavailable")

    service.detector.find = fail
    with pytest.raises(RuntimeError, match="detector unavailable"):
        service.detect(canary, project_id=project)
    snapshot = service.usage.snapshot()
    project_key = hashlib.sha256(project.encode()).hexdigest()[:16]
    row = snapshot["projects"][project_key]
    assert snapshot["active_requests"] == 0
    assert row["requests"] == row["documents"] == 2
    assert row["successes"] == row["failures"] == 1
    assert row["characters"] == len("Mara Calder ") + 2 * len(canary)
    serialized = json.dumps(snapshot) + service.usage.prometheus()
    assert all(
        value not in serialized
        for value in (project, patient, canary, "Mara Calder", SECRET.decode())
    )
    assert (
        f'deid_failures_total{{project="{project_key}"}} 1'
        in service.usage.prometheus()
    )


def test_explicit_empty_replacement_uses_configured_defaults(service):
    legacy = service.deidentify("Mara Calder")
    configured = service.deidentify("Mara Calder", replacement={})
    explicit = service.deidentify("Mara Calder", replacement={"type": "marker"})
    assert legacy["text"] == "[NAME]"
    assert configured == explicit
    assert configured["text"] == "[NAME_e1]"


def test_shared_segments_prepare_full_context_only_once(service, monkeypatch):
    import outis.replacement.transform as transform

    prepare = transform.prepare_context
    calls = []

    def counted(text, spans, **kwargs):
        calls.append(len(text))
        return prepare(text, spans, **kwargs)

    monkeypatch.setattr(transform, "prepare_context", counted)
    results = service.process_segments(
        [{"text": "Mara Calder", "replacement": {}}] * 64
    )
    assert len(calls) == 1
    assert {result["text"] for result in results} == {"[NAME_e1]"}


def test_detector_lock_is_released_between_model_batches(monkeypatch):
    import outis.service as service_module

    service = Deidentifier(detector=RecordingDetector(), secret=SECRET)
    held = []
    checked = service_module.checked_entities

    def recording(text, spans):
        held.append(service.lock.locked())
        return checked(text, spans)

    monkeypatch.setattr(service_module, "checked_entities", recording)
    service.process_segments(
        [{"text": f"note {index}"} for index in range(40)], shared_entities=False
    )
    assert len(service.detector.calls) == 3
    assert held and not any(held)
