"""Service and HTTP privacy checks with injected detectors, no model load."""
import re
import importlib.util
import io
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

from clinical_deid.api import create_app
from clinical_deid.service import Deidentifier
from clinical_deid.types import Span

SECRET = b"test-only-secret-never-use-in-production-12345"
TEXT = "Zoë Núñez was seen on 2024-10-03. Glucose 109 mg/dL. Review 2024-10-17."
VALID_PRESERVATION = {
    "clinical_text_outside_spans_preserved": True,
    "date_intervals_preserved": False,
    "exact_dates_shifted": 0,
    "unshiftable_dates_redacted": 0,
    "patient_consistent_replacements": False,
}


class StubDetector:
    def __init__(self):
        self.calls = []

    def detect_batch(self, texts):
        self.calls.append(list(texts))
        return [[Span(match.start(), match.end(), "NAME" if match[0] == "Zoë Núñez" else "DATE")
                 for match in re.finditer(r"Zoë Núñez|\d{4}-\d{2}-\d{2}", text)] for text in texts]


@pytest.fixture
def service():
    return Deidentifier(detector=StubDetector(), secret=SECRET)


def test_python_redaction_and_http_response_schema(service):
    result = service.deidentify(TEXT)
    assert result["mode"] == "redact"
    assert "Zoë Núñez" not in result["text"]
    assert "Glucose 109 mg/dL" in result["text"]
    assert result["model_version"] == "injected-detector"
    assert result["entity_count"] == 3
    assert all("original" not in entity for entity in result["entities"])
    with TestClient(create_app(service, api_key="test-key")) as client:
        response = client.post("/v1/deidentify", json={"text": TEXT}, headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        assert response.json() == result
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("phone", ["206-555-0184 extension 197", "+44 20 7946 0184 ext 197"])
@pytest.mark.parametrize("email", ["quinn.river@example.org", "ali\u0301ce@dept.example.xn--p1ai"])
@pytest.mark.parametrize("mode", ["redact", "synthetic"])
@pytest.mark.parametrize("separator", ["\t", "\u00a0\t\u2009", "\u2009\t\u00a0"])
@pytest.mark.parametrize("endpoint", ["whole_email", "first_char", "after_at"])
def test_http_contact_primitives_survive_ner_extension_overlap(phone, email, mode, separator, endpoint):
    from clinical_deid.rules import augment

    text = f"Telecare details\n{phone}{separator}{email}\nGlucose 109 mg/dL; metformin 500 mg."
    phone_start = text.index(phone)
    email_start = text.index(email)
    email_stop = email_start + len(email)
    if endpoint == "first_char":
        email_stop = email_start + 1
    elif endpoint == "after_at":
        email_stop = text.index("@", email_start) + 1

    class ContactDetector:
        def detect_batch(self, texts):
            assert texts == [text]
            # Reproduce an actual failure: NER places the phone extension
            # inside EMAIL and detects only the beginning of the telephone.
            native = [
                Span(phone_start, phone_start + 12, "PHONE", .9, "encoder"),
                Span(text.index("197"), email_stop, "EMAIL", .9, "encoder"),
            ]
            return [augment(text, native)]

    service = Deidentifier(detector=ContactDetector(), secret=SECRET)
    request = {"text": text, "mode": mode}
    if mode == "synthetic":
        request["patient_id"] = "contact-boundary-test-patient"
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/deidentify", json=request)
    assert response.status_code == 200
    result = response.json()
    assert result["entity_count"] == 2
    assert [(entity["start"], entity["end"], entity["label"]) for entity in result["entities"]] == [
        (phone_start, phone_start + len(phone), "PHONE"),
        (email_start, email_start + len(email), "EMAIL"),
    ]
    assert phone not in result["text"] and email not in result["text"]
    assert "Glucose 109 mg/dL; metformin 500 mg." in result["text"]
    assert "contact-boundary-test-patient" not in response.text
    if mode == "redact":
        assert result["text"] == f"Telecare details\n[PHONE]{separator}[EMAIL]\nGlucose 109 mg/dL; metformin 500 mg."


def test_synthetic_mode_preserves_date_intervals_and_same_anchor(service):
    request = {"text": TEXT, "mode": "synthetic", "patient_id": "private-patient-anchor", "date_shift_days": 14}
    first = service.deidentify(**request)
    second = service.deidentify(**request)
    assert first == second
    assert first["mode"] == "synthetic"
    assert "2024-10-17" in first["text"]
    assert "2024-10-31" in first["text"]
    assert first["preservation"]["exact_dates_shifted"] == 2
    assert first["preservation"]["date_intervals_preserved"]
    assert "private-patient-anchor" not in str(first)
    assert SECRET.decode() not in str(first)
    assert "date_shift_days" not in first


def test_http_batch_uses_detector_batch_once_and_preserves_item_order(service):
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/batch", json={"items": [{"text": TEXT}, {"text": "No PHI, 5 mg daily."}]})
        assert response.status_code == 200
        assert service.detector.calls == [[TEXT, "No PHI, 5 mg daily."]]
        assert response.json()[0]["entity_count"] == 3
        assert response.json()[1]["text"] == "No PHI, 5 mg daily."


def test_authentication_and_readiness(service):
    with TestClient(create_app(service, api_key="test-key")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 401
        assert client.post("/v1/deidentify", json={"text": TEXT}).status_code == 401
        assert client.get("/ready", headers={"X-API-Key": "wrong-key"}).status_code == 401
        assert client.get("/ready", headers={"X-API-Key": b"\xff"}).status_code == 401
        ready = client.get("/ready", headers={"X-API-Key": "test-key"})
        assert ready.status_code == 200
        assert ready.json()["synthetic_enabled"]
        assert not service.detector.calls


@pytest.mark.parametrize("changes", [
    {"mode": "bad"}, {"mode": "synthetic"}, {"mode": "synthetic", "patient_id": " "},
    {"date_shift_days": 14}, {"mode": "synthetic", "patient_id": "p", "date_shift_days": 0},
    {"mode": "synthetic", "patient_id": "p", "date_shift_days": True},
    {"mode": "synthetic", "patient_id": "p", "date_shift_days": 3651},
    {"patient_id": "x" * 257},
])
def test_invalid_policy_rejected_without_phi_echo_or_inference(service, changes):
    marker = "SENSITIVE-REQUEST-MARKER"
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/deidentify", json={"text": marker, **changes})
        assert response.status_code == 422
        assert marker not in response.text
        assert not service.detector.calls


def test_unknown_json_keys_cannot_echo_phi_in_validation_location(service):
    marker = "PRIVATE-PERSON-KEY"
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/deidentify", json={"text": TEXT, marker: "private value"})
        assert response.status_code == 422
        assert marker not in response.text
        assert "private value" not in response.text


@pytest.mark.parametrize("items", [[], [{"text": "x"}] * 17, [{"text": "x" * 50001}] * 2])
def test_batch_bounds_align_python_and_http_and_do_not_run_detector(service, items):
    with pytest.raises(ValueError):
        service.deidentify_batch(items)
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/batch", json={"items": items})
        assert response.status_code == 422
        assert len(response.text) < 1000
    assert not service.detector.calls


def test_text_length_bound_has_small_safe_response(service):
    marker = "PRIVATE-TEXT-" * 10000
    with pytest.raises(ValueError):
        service.deidentify(marker)
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/deidentify", json={"text": marker})
        assert response.status_code == 422
        assert "PRIVATE-TEXT-" not in response.text
        assert len(response.text) < 1000
    assert not service.detector.calls


def test_missing_secret_fails_closed_before_detector():
    service = Deidentifier(detector=StubDetector())
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/deidentify", json={"text": TEXT, "mode": "synthetic", "patient_id": "p"})
        assert response.status_code == 422
        assert TEXT not in response.text
        assert not service.detector.calls


def test_detector_failure_is_safe_and_handled_without_reraising(caplog):
    marker = "SENSITIVE-INTERNAL-PATIENT-NAME"
    class BrokenDetector:
        def detect_batch(self, texts):
            raise RuntimeError(marker)
    service = Deidentifier(detector=BrokenDetector())
    # Default raise_server_exceptions=True catches accidental re-raising to the
    # ASGI server, whose error logger could otherwise print the exception PHI.
    with TestClient(create_app(service)) as client:
        for endpoint, body in (("/v1/deidentify", {"text": TEXT}), ("/v1/batch", {"items": [{"text": TEXT}]})):
            response = client.post(endpoint, json=body)
            assert response.status_code == 500
            assert response.json() == {"detail": "Deidentification failed; no output released"}
            assert marker not in response.text
            assert response.headers["cache-control"] == "no-store"
    assert marker not in caplog.text


def test_invalid_detector_offsets_fail_closed(service):
    service.detector.detect_batch = lambda texts: [[Span(0, 999999, "NAME")] for text in texts]
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/deidentify", json={"text": TEXT})
        assert response.status_code == 422
        assert TEXT not in response.text


def test_malformed_detector_batch_output_fails_closed(service):
    service.detector.detect_batch = lambda texts: []
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/batch", json={"items": [{"text": TEXT}]})
        assert response.status_code == 500
        assert TEXT not in response.text


@pytest.mark.parametrize("score", ["PRIVATE-OUTPUT-MARKER", float("nan"), float("inf"), -.01, 1.01])
def test_invalid_service_response_score_is_handled_without_phi_rethrow(score, caplog):
    marker = "PRIVATE-OUTPUT-MARKER"
    malformed = {"text": marker, "mode": "redact", "warnings": [], "preservation": VALID_PRESERVATION.copy(),
                 "model_version": "malformed-service", "entity_count": 1,
                 "entities": [{"start": 0, "end": 3, "label": "NAME", "replacement": "[NAME]",
                               "score": score, "source": "injected"}]}
    class MalformedService:
        def deidentify(self, **payload):
            return malformed

        def deidentify_batch(self, payloads):
            return [malformed for payload in payloads]
    with TestClient(create_app(MalformedService())) as client:
        for endpoint, payload in (("/v1/deidentify", {"text": TEXT}), ("/v1/batch", {"items": [{"text": TEXT}]})):
            response = client.post(endpoint, json=payload)
            assert response.status_code == 500
            assert response.json() == {"detail": "Deidentification failed; no output released"}
            assert marker not in response.text
            assert response.headers["cache-control"] == "no-store"
    assert marker not in caplog.text


@pytest.mark.parametrize("metadata", [
    None,
    {},
    {**VALID_PRESERVATION, "PRIVATE-PRESERVATION-MARKER": object()},
    {**VALID_PRESERVATION, "clinical_text_outside_spans_preserved": object()},
    {**VALID_PRESERVATION, "date_intervals_preserved": "PRIVATE-PRESERVATION-MARKER"},
    {**VALID_PRESERVATION, "patient_consistent_replacements": 1},
    {**VALID_PRESERVATION, "exact_dates_shifted": -1},
    {**VALID_PRESERVATION, "exact_dates_shifted": True},
    {**VALID_PRESERVATION, "unshiftable_dates_redacted": 1.5},
])
def test_invalid_preservation_metadata_is_handled_before_serialization(metadata, caplog):
    marker = "PRIVATE-PRESERVATION-MARKER"
    malformed = {"text": marker, "mode": "redact", "warnings": [], "preservation": metadata,
                 "model_version": "malformed-service", "entity_count": 0, "entities": []}
    class MalformedService:
        def deidentify(self, **payload):
            return malformed

        def deidentify_batch(self, payloads):
            return [malformed for payload in payloads]
    # TestClient's default rethrow behavior reproduces an uncaught serialization
    # exception if arbitrary service metadata escapes endpoint validation.
    with TestClient(create_app(MalformedService())) as client:
        for endpoint, payload in (("/v1/deidentify", {"text": TEXT}), ("/v1/batch", {"items": [{"text": TEXT}]})):
            response = client.post(endpoint, json=payload)
            assert response.status_code == 500
            assert response.json() == {"detail": "Deidentification failed; no output released"}
            assert marker not in response.text
            assert response.headers["cache-control"] == "no-store"
    assert marker not in caplog.text


def test_openapi_preservation_metadata_has_exact_typed_fields(service):
    with TestClient(create_app(service)) as client:
        schema = client.get("/openapi.json").json()["components"]["schemas"]
    assert schema["DeidentifyResponse"]["properties"]["preservation"]["$ref"].endswith("/PreservationResponse")
    preservation = schema["PreservationResponse"]
    assert preservation["additionalProperties"] is False
    assert set(preservation["required"]) == set(VALID_PRESERVATION)
    assert set(preservation["properties"]) == set(VALID_PRESERVATION)
    for key, field in preservation["properties"].items():
        assert field["description"]
        if key in {"exact_dates_shifted", "unshiftable_dates_redacted"}:
            assert field["type"] == "integer"
            assert field["minimum"] == 0
        else:
            assert field["type"] == "boolean"


def test_missing_response_fields_are_caught_before_fastapi_response_validator(caplog):
    marker = "PRIVATE-MALFORMED-RESPONSE"
    class MalformedService:
        def deidentify(self, **payload):
            return {"text": marker}

        def deidentify_batch(self, payloads):
            return [{"text": marker} for payload in payloads]
    with TestClient(create_app(MalformedService())) as client:
        assert client.post("/v1/deidentify", json={"text": TEXT}).status_code == 500
        assert client.post("/v1/batch", json={"items": [{"text": TEXT}]}).status_code == 500
    assert marker not in caplog.text


def smoke_module():
    specification = importlib.util.spec_from_file_location("smoke_api", Path(__file__).parents[1] / "scripts" / "smoke_api.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def fake_smoke_http(monkeypatch, module, *, mutate_redaction=False):
    class SmokeDetector:
        def detect_batch(self, texts):
            results = []
            for text in texts:
                spans = []
                pattern = r"Mara Calder|\d{4}-\d{2}-\d{2}|MR-\d{6}|\(202\) 555-0143|mara\.calder@example\.org"
                for match in re.finditer(pattern, text):
                    value = match[0]
                    label = ("NAME" if value == "Mara Calder" else "MRN" if value.startswith("MR-")
                             else "PHONE" if value.startswith("(202)") else "EMAIL" if "@" in value else "DATE")
                    spans.append(Span(match.start(), match.end(), label))
                results.append(spans)
            return results
    service = Deidentifier(detector=SmokeDetector(), secret=SECRET)
    calls = []
    class Reply(io.BytesIO):
        headers = {"Cache-Control": "no-store"}
    def urlopen(request, timeout):
        payload = json.loads(request.data) if request.data is not None else None
        calls.append((request.full_url, payload, request.get_header("X-api-key")))
        endpoint = request.full_url.split("example.test", 1)[-1]
        if endpoint == "/ready":
            result = {"status": "ready", "synthetic_enabled": True, "model_version": service.version}
        elif endpoint == "/health":
            result = {"status": "ok"}
        elif endpoint == "/v1/deidentify":
            result = service.deidentify(**payload)
            if mutate_redaction and payload.get("mode") == "redact":
                for entity in result["entities"]:
                    if entity["label"] == "MRN":
                        entity["end"] -= 1
        elif endpoint == "/v1/batch":
            result = service.deidentify_batch(payload["items"])
        elif endpoint == "/openapi.json":
            result = {"paths": {path: {} for path in ("/v1/deidentify", "/v1/batch", "/health", "/ready")}}
        else:
            raise AssertionError("Unexpected smoke route")
        return Reply(json.dumps(result).encode())
    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    return calls


def test_http_smoke_exact_spans_default_cross_note_mapping_and_env_auth(tmp_path, monkeypatch):
    module = smoke_module()
    calls = fake_smoke_http(monkeypatch, module)
    api_key = "PRIVATE-SMOKE-AUTH-KEY"
    monkeypatch.setenv("DEID_API_KEY", api_key)
    output = tmp_path / "smoke.json"
    monkeypatch.setattr(sys, "argv", ["smoke_api", "--url", "https://example.test", "--output", str(output)])
    module.main()
    report = json.loads(output.read_text())
    assert report["passed"]
    assert len(report["redact"]["entities"]) == 7
    assert api_key not in output.read_text()
    assert all(key == api_key for _, _, key in calls)
    default_calls = [payload for url, payload, key in calls if url.endswith("/v1/deidentify")
                     and payload["mode"] == "synthetic" and "date_shift_days" not in payload]
    assert len(default_calls) == 2
    assert default_calls[0]["patient_id"] == default_calls[1]["patient_id"]
    assert default_calls[0]["text"] != default_calls[1]["text"]


def test_http_smoke_catches_partial_span_despite_presence_of_correct_label(tmp_path, monkeypatch):
    module = smoke_module()
    fake_smoke_http(monkeypatch, module, mutate_redaction=True)
    output = tmp_path / "smoke.json"
    monkeypatch.setattr(sys, "argv", ["smoke_api", "--url", "https://example.test", "--output", str(output)])
    with pytest.raises(AssertionError):
        module.main()
    assert not output.exists()
