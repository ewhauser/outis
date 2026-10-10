"""API extension contracts, request limits and failure privacy without model loads."""

import base64
import json
import re
from threading import current_thread

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from outis.api import create_app
from outis.service import Deidentifier
from outis.types import Span

SECRET = b"test-only-api-extensions-secret-1234567890"
TEXT = "Alice Calder was seen on 2024-10-03. Glucose 109 mg/dL."


class StubDetector:
    def __init__(self):
        self.calls = []

    def detect_batch(self, texts):
        self.calls.append(list(texts))
        return [
            [
                Span(
                    match.start(),
                    match.end(),
                    "NAME" if match[0] == "Alice Calder" else "DATE",
                )
                for match in re.finditer(r"Alice Calder|\d{4}-\d{2}-\d{2}", text)
            ]
            for text in texts
        ]


@pytest.fixture
def service():
    return Deidentifier(detector=StubDetector(), secret=SECRET)


@pytest.mark.parametrize(
    "path", ["/v1/policies", "/v1/usage", "/metrics", "/diagnostics"]
)
def test_operational_routes_require_auth_and_do_not_cache(service, path):
    with TestClient(create_app(service, api_key="test-key")) as client:
        denied = client.get(path)
        assert denied.status_code == 401
        assert denied.headers["cache-control"] == "no-store"
        response = client.get(path, headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/v1/detect", {"text": TEXT}),
        ("/v1/detect/batch", {"items": [{"text": TEXT}]}),
        ("/v1/analyze", {"text": TEXT}),
        ("/v1/reidentify", {"text": "[NAME_1]", "mappings": []}),
        ("/v1/structured", {"data": {"note": TEXT}}),
        (
            "/v1/files",
            {
                "data": base64.b64encode(TEXT.encode()).decode(),
                "content_type": "text/plain",
            },
        ),
    ],
)
def test_new_processing_routes_require_auth(service, path, payload):
    with TestClient(create_app(service, api_key="test-key")) as client:
        response = client.post(path, json=payload)
        assert response.status_code == 401
        assert response.headers["cache-control"] == "no-store"
    assert service.detector.calls == []


def test_detection_policy_is_shared_with_deidentification_and_no_original_values(
    service,
):
    text = "Alice Calder has study ID STUDY-482901."
    policy = {"custom_entities": [{"label": "STUDY_ID", "pattern": r"\bSTUDY-\d{6}\b"}]}
    with TestClient(create_app(service)) as client:
        detected = client.post("/v1/detect", json={"text": text, "policy": policy})
        processed = client.post("/v1/deidentify", json={"text": text, "policy": policy})
    assert detected.status_code == processed.status_code == 200
    detections, replacements = detected.json(), processed.json()
    assert "text" not in detections
    assert "Alice Calder" not in detected.text
    assert "STUDY-482901" not in detected.text
    assert [
        (row["start"], row["end"], row["label"]) for row in detections["entities"]
    ] == [(row["start"], row["end"], row["label"]) for row in replacements["entities"]]
    assert detections["policy"]["custom_entity_count"] == 1
    assert "STUDY-482901" not in replacements["text"]
    assert "Alice Calder" not in replacements["text"]


def test_detection_batch_order_and_shared_context_options(service):
    payload = {
        "items": [{"text": TEXT}, {"text": "No identifiers."}],
        "shared_entities": True,
    }
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/detect/batch", json=payload)
    assert response.status_code == 200
    assert [result["entity_count"] for result in response.json()] == [2, 0]
    assert all("entity_id" in entity for entity in response.json()[0]["entities"])


def test_analysis_sensitive_normalization_is_opt_in(service):
    with TestClient(create_app(service)) as client:
        default = client.post("/v1/analyze", json={"text": TEXT})
        opted_in = client.post(
            "/v1/analyze", json={"text": TEXT, "include_sensitive": True}
        )
    assert default.status_code == opted_in.status_code == 200
    assert "2024-10-03" not in default.text
    assert "Alice Calder" not in default.text
    assert not default.json()["metadata"]["sensitive_values_included"]
    assert "2024-10-03" in opted_in.text
    assert opted_in.json()["metadata"]["sensitive_values_included"]


def test_mapping_roundtrip_requires_explicit_restoration(service):
    with TestClient(create_app(service)) as client:
        redacted = client.post(
            "/v1/deidentify",
            json={"text": TEXT, "replacement": {"include_mapping": True}},
        )
        assert redacted.status_code == 200
        result = redacted.json()
        assert result["mappings"]
        payload = {"text": result["text"], "mappings": result["mappings"]}
        denied = client.post("/v1/reidentify", json=payload)
        restored = client.post(
            "/v1/reidentify", json={**payload, "restore_sensitive": True}
        )
    assert denied.status_code in {200, 422}
    if denied.status_code == 200:
        assert "Alice Calder" not in denied.json()["text"]
    assert restored.status_code == 200
    assert restored.json()["text"] == TEXT
    assert restored.json()["restored_count"] == 2


def test_structured_json_preserves_shape_and_nulls(service):
    original = {"notes": [TEXT, "unchanged"], "count": 2, "absent": None}
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/structured", json={"data": original})
    assert response.status_code == 200
    result = response.json()
    assert result["data"]["count"] == 2
    assert result["data"]["absent"] is None
    assert result["data"]["notes"][1] == "unchanged"
    assert "Alice Calder" not in result["data"]["notes"][0]
    assert result["entity_count"] == 2
    assert all("location" in entity for entity in result["entities"])


def test_structured_json_requires_data_but_preserves_explicit_null(service):
    with TestClient(create_app(service)) as client:
        missing = client.post("/v1/structured", json={})
        assert missing.status_code == 422
        assert service.detector.calls == []
        for original in (None, {"value": None}):
            response = client.post("/v1/structured", json={"data": original})
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
            result = response.json()
            assert "data" in result
            assert result["data"] == original
            assert result["format"] == "json"
            assert result["entities"] == []
            assert result["entity_count"] == 0
            assert result["warnings"] == []


def test_file_base64_roundtrip(service):
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/files",
            json={
                "data": base64.b64encode(TEXT.encode()).decode(),
                "content_type": "text/plain",
            },
        )
    assert response.status_code == 200
    result = response.json()
    decoded = base64.b64decode(result["data"], validate=True).decode()
    assert "Alice Calder" not in decoded
    assert "Glucose 109 mg/dL" in decoded
    assert result["entity_count"] == 2


def test_usage_has_aggregate_counts_and_hashed_project_identifiers(service):
    private_project = "PRIVATE-PROJECT-MARKER"
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/detect", json={"text": TEXT, "project_id": private_project}
        )
        assert response.status_code == 200
        usage = client.get("/v1/usage")
        metrics = client.get("/metrics")
        diagnostics = client.get("/diagnostics")
    assert usage.status_code == metrics.status_code == diagnostics.status_code == 200
    assert (
        sum(project["documents"] for project in usage.json()["projects"].values()) == 1
    )
    assert "deid_documents_total" in metrics.text
    for response in (usage, metrics, diagnostics):
        assert private_project not in response.text
        assert "Alice Calder" not in response.text
        assert SECRET.decode() not in response.text


@pytest.mark.parametrize(
    "path,payload",
    [
        (
            "/v1/detect",
            {
                "text": TEXT,
                "policy": {"medical_terms": {"PRIVATE-DYNAMIC-KEY": ["PRIVATE-INPUT"]}},
            },
        ),
        (
            "/v1/deidentify",
            {"text": TEXT, "replacement": {"PRIVATE-DYNAMIC-KEY": "PRIVATE-INPUT"}},
        ),
        (
            "/v1/structured",
            {"data": {"PRIVATE-DYNAMIC-KEY": TEXT}, "options": {"PRIVATE-INPUT": 123}},
        ),
    ],
)
def test_nested_validation_locations_never_echo_submitted_keys(service, path, payload):
    with TestClient(create_app(service)) as client:
        response = client.post(path, json=payload)
    assert response.status_code == 422
    assert "PRIVATE-DYNAMIC-KEY" not in response.text
    assert "PRIVATE-INPUT" not in response.text
    assert TEXT not in response.text
    assert not service.detector.calls


@pytest.mark.parametrize("content_length", [True, False])
def test_http_body_byte_limit_includes_chunked_requests(
    service, monkeypatch, content_length
):
    import outis.api as api

    monkeypatch.setattr(api, "MAX_REQUEST_BYTES", 512)
    body = json.dumps({"text": "PRIVATE-TEXT" * 100}).encode()
    content = body if content_length else iter([body[:200], body[200:]])
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/detect", content=content, headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}
    assert response.headers["cache-control"] == "no-store"
    assert not service.detector.calls


@pytest.mark.parametrize("value", [object(), float("nan"), float("inf")])
def test_invalid_extension_output_is_handled_without_serialization_phi(value, caplog):
    class BrokenService:
        def detect(self, **kwargs):
            return {
                "entities": [],
                "entity_count": 0,
                "model_version": "injected",
                "policy": {"PRIVATE-OUTPUT-MARKER": value},
            }

    with TestClient(create_app(BrokenService())) as client:
        response = client.post("/v1/detect", json={"text": TEXT})
    assert response.status_code == 500
    assert response.json() == {"detail": "Deidentification failed; no output released"}
    assert "PRIVATE-OUTPUT-MARKER" not in response.text
    assert "PRIVATE-OUTPUT-MARKER" not in caplog.text


def test_websocket_matches_http_and_runs_detector_in_threadpool(service):
    original = service.detector.detect_batch
    thread_names = []

    def detector(texts):
        thread_names.append(current_thread().name)
        return original(texts)

    service.detector.detect_batch = detector
    with TestClient(create_app(service, api_key="test-key")) as client:
        http = client.post(
            "/v1/detect", json={"text": TEXT}, headers={"X-API-Key": "test-key"}
        )
        with client.websocket_connect(
            "/v1/ws", headers={"X-API-Key": "test-key"}
        ) as socket:
            socket.send_json({"operation": "detect", "payload": {"text": TEXT}})
            response = socket.receive_json()
    assert http.status_code == 200
    assert response == {"operation": "detect", "result": http.json()}
    assert all("AnyIO worker thread" in name for name in thread_names)


def test_websocket_rejects_missing_header_auth_query_secrets_and_foreign_origin(
    service,
):
    with TestClient(create_app(service, api_key="test-key")) as client:
        for path, headers in [
            ("/v1/ws", {}),
            ("/v1/ws?api_key=test-key", {}),
            ("/v1/ws", {"X-API-Key": "test-key", "Origin": "https://attacker.invalid"}),
        ]:
            with pytest.raises(WebSocketDisconnect) as denied:
                with client.websocket_connect(path, headers=headers):
                    pass
            assert denied.value.code == 1008
        with client.websocket_connect(
            "/v1/ws", headers={"X-API-Key": "test-key", "Origin": "http://testserver"}
        ) as socket:
            socket.send_json({"operation": "detect", "payload": {"text": "clean"}})
            assert socket.receive_json()["result"]["entity_count"] == 0


def test_websocket_invalid_envelopes_are_private_and_connection_recovers(service):
    with TestClient(create_app(service)) as client:
        with client.websocket_connect("/v1/ws") as socket:
            for payload in [
                {"operation": "PRIVATE-OPERATION", "payload": {"text": TEXT}},
                {"operation": "detect", "payload": {"PRIVATE-KEY": TEXT}},
                {"operation": "detect", "payload": {"text": "PRIVATE-TEXT" * 10000}},
            ]:
                socket.send_json(payload)
                assert socket.receive_json() == {
                    "error": {"status": 422, "detail": "Invalid request"}
                }
            socket.send_json({"operation": "detect", "payload": {"text": TEXT}})
            assert socket.receive_json()["result"]["entity_count"] == 2
    assert len(service.detector.calls) == 1


def test_websocket_runtime_failure_never_echoes_internal_values(caplog):
    class BrokenService:
        def detect(self, **kwargs):
            raise RuntimeError("PRIVATE-INTERNAL-MARKER")

    with TestClient(create_app(BrokenService())) as client:
        with client.websocket_connect("/v1/ws") as socket:
            socket.send_json({"operation": "detect", "payload": {"text": TEXT}})
            assert socket.receive_json() == {
                "error": {
                    "status": 500,
                    "detail": "Deidentification failed; no output released",
                }
            }
    assert "PRIVATE-INTERNAL-MARKER" not in caplog.text


def test_websocket_envelope_byte_limit(service, monkeypatch):
    import outis.api as api

    monkeypatch.setattr(api, "MAX_REQUEST_BYTES", 512)
    with TestClient(create_app(service)) as client:
        with client.websocket_connect("/v1/ws") as socket:
            socket.send_text("PRIVATE-TEXT" * 100)
            with pytest.raises(WebSocketDisconnect) as closed:
                socket.receive_json()
            assert closed.value.code == 1009
    assert not service.detector.calls


@pytest.mark.parametrize(
    "replacement", [{"type": "synthetic"}, {"per_type": {"DATE": "synthetic"}}]
)
def test_effective_synthetic_replacement_accepts_date_shift_without_legacy_mode(
    service, replacement
):
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/deidentify",
            json={
                "text": TEXT,
                "patient_id": "patient-1",
                "date_shift_days": 14,
                "replacement": replacement,
            },
        )
    assert response.status_code == 200
    assert "2024-10-17" in response.json()["text"]
    assert "2024-10-03" not in response.json()["text"]


@pytest.mark.parametrize(
    "replacement",
    [
        {"type": "synthetic"},
        {"per_type": {"DATE": "synthetic"}},
        {"entity_ids": "keyed"},
    ],
)
def test_effective_sensitive_replacement_requires_patient_anchor_before_inference(
    service, replacement
):
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/deidentify", json={"text": TEXT, "replacement": replacement}
        )
    assert response.status_code == 422
    assert not service.detector.calls


@pytest.mark.parametrize("label", ["MRN", "ACCOUNT", "ID"])
def test_http_synthetic_opaque_identifiers_keep_exact_identities_and_restore(label):
    originals = ["AB12", "ab12", "AB12 "]
    text = "|".join(originals)
    spans = [Span(0, 4, label), Span(5, 9, label), Span(10, 15, label)]

    class OpaqueIdentifierDetector:
        def detect_batch(self, texts):
            assert texts == [text]
            return [spans]

    service = Deidentifier(detector=OpaqueIdentifierDetector(), secret=SECRET)
    payload = {
        "text": text,
        "mode": "synthetic",
        "patient_id": "opaque-identity-patient",
        "replacement": {
            "type": "synthetic",
            "entity_ids": "keyed",
            "include_mapping": True,
        },
    }
    with TestClient(create_app(service)) as client:
        first = client.post("/v1/deidentify", json=payload)
        repeated = client.post("/v1/deidentify", json=payload)
        assert first.status_code == repeated.status_code == 200
        result = first.json()
        assert result == repeated.json()
        assert result["entity_count"] == 3
        assert len({entity["entity_id"] for entity in result["entities"]}) == 3
        replacements = [entity["replacement"] for entity in result["entities"]]
        assert len(set(replacements)) == 3
        assert set(replacements).isdisjoint(originals)
        assert result["text"] == "|".join(replacements)
        assert {row["marker"]: row["original"] for row in result["mappings"]} == dict(
            zip(replacements, originals)
        )
        assert all(row["label"] == label for row in result["mappings"])
        restored = client.post(
            "/v1/reidentify",
            json={
                "text": result["text"],
                "mappings": result["mappings"],
                "allowed_types": [label],
                "restore_sensitive": True,
            },
        )
    assert restored.status_code == 200
    assert restored.json() == {"text": text, "restored_count": 3}


@pytest.mark.parametrize(
    "dates,shift,expected",
    [
        (
            ["13/09/2024", "14/09/2024", "09/13/2024", "2024-09-13"],
            20,
            ["2024/10/03", "2024/10/04", "10/03/2024", "2024-10-03"],
        ),
        (["29/2/2024", "2024-02-29"], 1, ["2024/3/1", "2024-03-01"]),
    ],
)
def test_http_synthetic_dmy_dates_emit_unambiguous_shifted_timeline(
    dates, shift, expected
):
    text = "Visits: " + "; ".join(dates) + "."

    class DateDetector:
        def detect_batch(self, texts):
            assert texts == [text]
            return [
                [
                    Span(text.index(date), text.index(date) + len(date), "DATE")
                    for date in dates
                ]
            ]

    service = Deidentifier(detector=DateDetector(), secret=SECRET)
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/deidentify",
            json={
                "text": text,
                "mode": "synthetic",
                "patient_id": "date-timeline-patient",
                "date_shift_days": shift,
                "replacement": {"type": "synthetic"},
            },
        )
    assert response.status_code == 200
    result = response.json()
    assert result["text"] == "Visits: " + "; ".join(expected) + "."
    assert [entity["replacement"] for entity in result["entities"]] == expected
    assert all(
        result["text"][entity["processed_start"] : entity["processed_end"]]
        == entity["replacement"]
        for entity in result["entities"]
    )
    assert result["preservation"]["date_intervals_preserved"]
    assert result["preservation"]["exact_dates_shifted"] == len(dates)
    assert result["preservation"]["unshiftable_dates_redacted"] == 0
    assert any("year/month/day" in warning.lower() for warning in result["warnings"])


@pytest.mark.parametrize("project_id", ["x" * 61, "space name", "person@example.com"])
def test_project_identifier_contract_matches_service(service, project_id):
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/detect", json={"text": TEXT, "project_id": project_id}
        )
    assert response.status_code == 422
    assert project_id not in response.text
    assert not service.detector.calls


@pytest.mark.parametrize(
    "operation,payload",
    [
        ("deidentify", {"text": TEXT, "replacement": {"type": "mask"}}),
        ("analyze", {"text": TEXT}),
        (
            "reidentify",
            {
                "text": "Hello [NAME_1]",
                "mappings": [
                    {"marker": "[NAME_1]", "original": "Alice Calder", "label": "NAME"}
                ],
                "restore_sensitive": True,
            },
        ),
    ],
)
def test_websocket_extended_operations_match_http(service, operation, payload):
    with TestClient(create_app(service)) as client:
        http = client.post(f"/v1/{operation}", json=payload)
        with client.websocket_connect("/v1/ws") as socket:
            socket.send_json({"operation": operation, "payload": payload})
            response = socket.receive_json()
    assert http.status_code == 200
    assert response == {"operation": operation, "result": http.json()}


def test_unencodable_detector_response_is_caught_before_fastapi_serialization(caplog):
    class BrokenService:
        def detect(self, **kwargs):
            return {
                "entities": [],
                "entity_count": 0,
                "model_version": "PRIVATE-OUTPUT-MARKER\ud800",
            }

    with TestClient(create_app(BrokenService())) as client:
        response = client.post("/v1/detect", json={"text": TEXT})
    assert response.status_code == 500
    assert response.json() == {"detail": "Deidentification failed; no output released"}
    assert "PRIVATE-OUTPUT-MARKER" not in response.text
    assert "PRIVATE-OUTPUT-MARKER" not in caplog.text


def test_escaped_surrogate_input_cannot_trigger_unhandled_response_error(
    service, caplog
):
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/deidentify",
            content=b'{"text":"PRIVATE-INPUT-MARKER\\ud800"}',
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid request"
    assert "PRIVATE-INPUT-MARKER" not in response.text
    assert "PRIVATE-INPUT-MARKER" not in caplog.text
    assert not service.detector.calls


@pytest.mark.parametrize(
    "origin",
    [
        "http://testserver",
        "https://testserver/path",
        "https://testserver#private",
        "https://user@testserver",
        "https://testserver:444",
    ],
)
def test_websocket_tls_rejects_origins_outside_exact_origin(service, origin):
    with TestClient(create_app(service)) as client:
        with pytest.raises(WebSocketDisconnect) as denied:
            with client.websocket_connect(
                "wss://testserver/v1/ws", headers={"Origin": origin}
            ):
                pass
        assert denied.value.code == 1008
        with client.websocket_connect(
            "wss://testserver/v1/ws", headers={"Origin": "https://testserver:443"}
        ) as socket:
            socket.send_json({"operation": "detect", "payload": {"text": "clean"}})
            assert socket.receive_json()["result"]["entity_count"] == 0
