"""Exercise the expanded API against a local checkpoint using invented text only.

Writes statuses/counts rather than source text, mappings, patient IDs, or keys.
This verifies integration, not independent model accuracy or release readiness.
"""

import argparse
import base64
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from clinical_deid.api import create_app
from clinical_deid.service import Deidentifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/clinical-adapter")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("reports/extended_api_smoke.json"))
    args = parser.parse_args()
    service = Deidentifier(args.model_dir, device=args.device,
                           secret=b"integration-smoke-only-never-use-for-real-data")
    policy = {"custom_entities": [{"label": "STUDY_ID", "pattern": r"\bSTUDY-\d{6}\b"}]}
    text = "Patient Mara Calder, DOB 1974-06-13, study STUDY-482901. Creatinine 1.2 mg/dL."
    results = {}
    with TestClient(create_app(service, api_key="smoke-test-only")) as client:
        client.headers["X-API-Key"] = "smoke-test-only"

        def post(route, payload):
            response = client.post(route, json=payload)
            assert response.status_code == 200, f"Unexpected status for {route}: {response.status_code}"
            assert response.headers["cache-control"] == "no-store"
            results[route] = {"status": response.status_code}
            return response.json()

        detection = post("/v1/detect", {"text": text, "policy": policy})
        assert any(e["label"] == "STUDY_ID" for e in detection["entities"])
        assert "text" not in detection
        processed = post("/v1/deidentify", {"text": text, "policy": policy,
            "replacement": {"pattern": "[{label}_{index}]", "include_mapping": True}})
        assert "STUDY-482901" not in processed["text"]
        assert "Creatinine 1.2 mg/dL." in processed["text"]
        assert [(e["start"], e["end"], e["label"]) for e in detection["entities"]] == [
            (e["start"], e["end"], e["label"]) for e in processed["entities"]]
        for entity in processed["entities"]:
            assert processed["text"][entity["processed_start"]:entity["processed_end"]] == entity["replacement"]
        restored = post("/v1/reidentify", {"text": processed["text"],
                         "mappings": processed["mappings"], "restore_sensitive": True})
        assert restored["text"] == text
        analyzed = post("/v1/analyze", {"text": text, "policy": policy})
        assert analyzed["metadata"]["sensitive_values_included"] is False
        batch = post("/v1/batch", {"shared_entities": True, "items": [
            {"text": text, "policy": policy, "replacement": {"type": "marker"}},
            {"text": text, "policy": policy, "replacement": {"type": "marker"}}
        ]})
        assert batch[0]["text"] == batch[1]["text"]
        structured = post("/v1/structured", {"data": {"notes": [text, text]},
                                            "options": {"policy": policy}})
        assert "STUDY-482901" not in json.dumps(structured["data"])
        file = post("/v1/files", {"data": base64.b64encode(text.encode()).decode(),
                                  "content_type": "text/plain", "options": {"policy": policy}})
        assert "STUDY-482901" not in base64.b64decode(file["data"]).decode()
        with client.websocket_connect("/v1/ws") as socket:
            socket.send_json({"operation": "detect", "payload": {"text": text, "policy": policy}})
            assert socket.receive_json()["result"] == detection
            results["/v1/ws"] = {"passed": True}
        for route in ("/v1/policies", "/v1/usage", "/metrics", "/diagnostics"):
            response = client.get(route)
            assert response.status_code == 200
            assert "Mara Calder" not in response.text
            results[route] = {"status": response.status_code}
    report = {"passed": True, "model_version": service.version,
              "qualification": "integration_only_invented_inputs",
              "detected_entities": detection["entity_count"], "endpoints": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
