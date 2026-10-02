"""Public Python API. All processing is local and serializes GPU access."""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock
from typing import Literal

from .replacement import replace


class Deidentifier:
    def __init__(
        self,
        model_dir="models/clinical-adapter",
        *,
        secret=None,
        device="auto",
        backend="hybrid",
        detector=None,
        base_dir=None,
    ):
        if detector is None:
            from .model import Detector

            detector = Detector(
                model_dir, device=device, backend=backend, base_dir=base_dir
            )
        self.detector = detector
        self.secret = secret
        self.lock = Lock()
        path = Path(model_dir) / "adapter.safetensors"
        self.version = (
            hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            if path.is_file() and getattr(detector, "model_dir", None) is not None
            else "injected-detector"
        )

    def deidentify(
        self,
        text: str,
        *,
        mode: Literal["redact", "synthetic"] = "redact",
        patient_id=None,
        date_shift_days=None,
    ):
        return self.deidentify_batch(
            [
                {
                    "text": text,
                    "mode": mode,
                    "patient_id": patient_id,
                    "date_shift_days": date_shift_days,
                }
            ]
        )[0]

    def deidentify_batch(self, requests):
        from .replacement import _validated_secret

        if not isinstance(requests, (list, tuple)) or not 1 <= len(requests) <= 16:
            raise ValueError("A batch must contain between 1 and 16 requests")
        total_characters = 0
        for req in requests:
            if not isinstance(req, dict):
                raise ValueError("Each batch request must be a dictionary")
            if not isinstance(req.get("text"), str) or len(req["text"]) > 100_000:
                raise ValueError("Text must contain at most 100000 characters")
            total_characters += len(req["text"])
            if total_characters > 100_000:
                raise ValueError("A batch may contain at most 100000 characters")
            if req.get("mode", "redact") not in {"redact", "synthetic"}:
                raise ValueError("mode must be redact or synthetic")
            shift = req.get("date_shift_days")
            if shift is not None:
                if req.get("mode", "redact") != "synthetic":
                    raise ValueError("Date shifts apply only in synthetic mode")
                if (
                    isinstance(shift, bool)
                    or not isinstance(shift, int)
                    or not 1 <= abs(shift) <= 3650
                ):
                    raise ValueError(
                        "date_shift_days must be a nonzero integer between -3650 and 3650"
                    )
            anchor = req.get("patient_id")
            if anchor is not None and (
                not isinstance(anchor, str) or not 1 <= len(anchor) <= 256
            ):
                raise ValueError("patient_id must contain between 1 and 256 characters")
            if req.get("mode", "redact") == "synthetic":
                _validated_secret(self.secret)
                if (
                    not isinstance(req.get("patient_id"), str)
                    or not req["patient_id"].strip()
                ):
                    raise ValueError("Synthetic mode requires a stable patient_id")
        texts = [req["text"] for req in requests]
        with self.lock:
            if hasattr(self.detector, "detect_batch"):
                predictions = self.detector.detect_batch(texts)
            else:
                predictions = [self.detector.detect(text) for text in texts]
        if len(predictions) != len(texts):
            raise RuntimeError("Detector returned invalid batch output")
        results = []
        for req, spans in zip(requests, predictions):
            result = replace(
                req["text"],
                spans,
                mode=req.get("mode", "redact"),
                patient_id=req.get("patient_id"),
                secret=self.secret,
                date_shift_days=req.get("date_shift_days"),
            )
            result["model_version"] = self.version
            result["mode"] = req.get("mode", "redact")
            result["entity_count"] = len(result["entities"])
            results.append(result)
        return results
