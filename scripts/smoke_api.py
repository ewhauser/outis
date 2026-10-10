"""Exercise the running API with invented clinical text and interval checks."""

import argparse
import json
import os
import urllib.request
from datetime import date
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="reports/api_smoke.json")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OUTIS_API_KEY"),
        help="Optional X-API-Key; defaults to OUTIS_API_KEY and is never recorded",
    )
    args = parser.parse_args()

    def request(path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if args.api_key:
            headers["X-API-Key"] = args.api_key
        req = urllib.request.Request(
            args.url + path,
            data=body,
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            assert response.headers.get("Cache-Control") == "no-store"
            return json.load(response)

    ready = request("/ready")
    assert ready["status"] == "ready" and ready["synthetic_enabled"]
    assert request("/health")["status"] == "ok"
    text = (
        "Patient Mara Calder, DOB 1974-06-13, MRN MR-724891. "
        "Admitted 2024-02-28; discharged 2024-03-02. "
        "Call (202) 555-0143 or mara.calder@example.org. "
        "Creatinine 1.2 mg/dL, potassium 4.1 mmol/L. "
        "Take 25 mg twice daily for 14 days; follow up in six weeks."
    )
    expected = (
        ("NAME", "Mara Calder"),
        ("DATE", "1974-06-13"),
        ("MRN", "MR-724891"),
        ("DATE", "2024-02-28"),
        ("DATE", "2024-03-02"),
        ("PHONE", "(202) 555-0143"),
        ("EMAIL", "mara.calder@example.org"),
    )

    def check_spans(result, original, annotations):
        gold = {
            (original.index(value), original.index(value) + len(value), label)
            for label, value in annotations
        }
        predicted = {
            (entity["start"], entity["end"], entity["label"])
            for entity in result["entities"]
        }
        assert predicted == gold and len(result["entities"]) == len(gold)
        assert result["entity_count"] == len(gold)
        assert result["model_version"] == ready["model_version"]

    redact = request("/v1/deidentify", {"text": text, "mode": "redact"})
    check_spans(redact, text, expected)
    assert all(value not in redact["text"] for _, value in expected)
    payload = {
        "text": text,
        "mode": "synthetic",
        "patient_id": "invented-smoke-patient",
        "date_shift_days": 31,
    }
    synthetic, repeat = request("/v1/batch", {"items": [payload, payload]})
    assert synthetic == repeat
    check_spans(synthetic, text, expected)
    shifted_dates = [
        entity["replacement"]
        for entity in synthetic["entities"]
        if entity["label"] == "DATE"
    ]
    assert shifted_dates == ["1974-07-14", "2024-03-30", "2024-04-02"]
    assert (
        date.fromisoformat(shifted_dates[2]) - date.fromisoformat(shifted_dates[1])
    ).days == 3
    clinical = [
        "Creatinine 1.2 mg/dL, potassium 4.1 mmol/L.",
        "Take 25 mg twice daily for 14 days; follow up in six weeks.",
    ]
    for phrase in clinical:
        assert phrase in redact["text"] and phrase in synthetic["text"]
    # Separate notes for the same patient must derive one nonzero date offset
    # without an explicit shift, including dates not present in the first note.
    default_payload = {
        "text": text,
        "mode": "synthetic",
        "patient_id": "invented-smoke-patient",
    }
    default_first = request("/v1/deidentify", default_payload)
    check_spans(default_first, text, expected)
    followup_text = (
        "Patient Mara Calder, follow-up 2024-03-16. Discharged 2024-03-02 "
        "after admission 2024-02-28. Continue 25 mg twice daily."
    )
    followup_expected = (
        ("NAME", "Mara Calder"),
        ("DATE", "2024-03-16"),
        ("DATE", "2024-03-02"),
        ("DATE", "2024-02-28"),
    )
    default_followup = request(
        "/v1/deidentify", {**default_payload, "text": followup_text}
    )
    check_spans(default_followup, followup_text, followup_expected)

    def replacements(result, original, label):
        return {
            original[entity["start"] : entity["end"]]: entity["replacement"]
            for entity in result["entities"]
            if entity["label"] == label
        }

    first_dates = replacements(default_first, text, "DATE")
    followup_dates = replacements(default_followup, followup_text, "DATE")
    assert all(
        first_dates[value] == followup_dates[value]
        for value in ("2024-02-28", "2024-03-02")
    )
    offsets = {
        (date.fromisoformat(shifted) - date.fromisoformat(original)).days
        for mapping in (first_dates, followup_dates)
        for original, shifted in mapping.items()
    }
    assert len(offsets) == 1 and next(iter(offsets)) != 0
    assert replacements(default_first, text, "NAME") == replacements(
        default_followup, followup_text, "NAME"
    )
    assert "Continue 25 mg twice daily." in default_followup["text"]
    assert all(phrase in default_first["text"] for phrase in clinical)
    assert default_first["preservation"]["date_intervals_preserved"]
    assert default_followup["preservation"]["date_intervals_preserved"]
    clinical_negatives = [
        "Creatinine 1.2 mg/dL and potassium 4.1 mmol/L. Continue 25 mg twice daily for 14 days.",
        "The infant is aged 100 days. Follow up after 120 hours.",
        "The child is aged 120 months. Continue observation for 100 minutes.",
        "Patient transport to ED for evaluation.",
        "Patient transport to ER for evaluation.",
        "Patient transport to OR for evaluation.",
        "Patient transport to NICU for evaluation.",
        "Patient transport to Ultrasound for evaluation.",
        "Patient transport to Operating Room for evaluation.",
        "Patient transport to MRI for imaging.",
        "Community support is available in Nephrology.",
        "Home care is provided in Rheumatology.",
        "Community support is available in COPD management.",
        "Home care is provided in Palliative Care.",
        "The household is in Isolation after exposure.",
        "The household is in Quarantine after exposure.",
        "The patient lives in Assisted Living.",
        "The patient moved from CPAP to BiPAP.",
        "The patient returned to Sinus Rhythm.",
        "Patient name is Unknown; continue clinical assessment.",
        "Patient name is The patient declined to provide a name.",
    ]
    for start in range(0, len(clinical_negatives), 16):
        texts = clinical_negatives[start : start + 16]
        results = request(
            "/v1/batch",
            {"items": [{"text": value, "mode": "redact"} for value in texts]},
        )
        assert len(results) == len(texts)
        for original, result in zip(texts, results):
            assert result["entities"] == [] and result["text"] == original, original
    schema = request("/openapi.json")
    assert {"/v1/deidentify", "/v1/batch", "/health", "/ready"} <= set(schema["paths"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Invented text over the running HTTP API with the actual trained model",
        "passed": True,
        "checks": [
            "redaction removes names, dates, record numbers and contacts",
            "all seven expected typed spans match exact input boundaries",
            "same patient and key produce identical replacement",
            "all exact dates shift 31 days and preserve a 3-day admission interval",
            "separate notes for the same patient use consistent default date and name replacements",
            "clinical measurements, doses and relative durations remain intact",
            "21 clinical negatives retain their exact text and have no PHI detections",
            "batch endpoint, readiness, OpenAPI and no-store headers work",
        ],
        "ready": ready,
        "redact": redact,
        "synthetic": synthetic,
        "default_synthetic": default_first,
        "default_followup": default_followup,
        "clinical_negative_count": len(clinical_negatives),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "model_version": ready["model_version"],
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
