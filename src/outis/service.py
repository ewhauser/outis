"""Local detection, policy, transformation, and structured-processing service."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace as with_fields
from pathlib import Path
from threading import Lock

from outis.entities import checked_entities
from outis.operations import Usage
from outis.paths import resolve_model_dir
from outis.replacement.primitives import validated_secret


def _model_version(model_dir, detector):
    """Identify the locally loaded inference bundle without retaining its weights."""
    loaded_dir = getattr(detector, "model_dir", None)
    config = getattr(detector, "config", {})
    if (
        loaded_dir is not None
        and isinstance(config, dict)
        and config.get("encoder") == "hf_token_classifier"
    ):
        root = Path(loaded_dir)
        files = [root / "config.json"]
        files.extend(
            sorted(path for path in (root / "hf").rglob("*") if path.is_file())
        )
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
            with path.open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").digest())
        return digest.hexdigest()[:16]
    return "injected-detector"


class Deidentifier:
    def __init__(
        self,
        model_dir=None,
        *,
        secret=None,
        device="auto",
        detector=None,
    ):
        if detector is None:
            from outis.detection.hf_encoder import load_detector

            model_dir = resolve_model_dir(model_dir)
            detector = load_detector(model_dir, device=device)
        self.detector = detector
        self.secret = secret
        self.lock = Lock()
        self.usage = Usage()
        self.version = _model_version(model_dir, detector)

    def _requests(self, requests, *, detect=False, limit=16):
        from outis.policy import KNOWN_LABELS, validate_policy
        from outis.replacement.transform import ReplacementOptions, legacy_options

        if not isinstance(requests, (list, tuple)) or not 1 <= len(requests) <= limit:
            raise ValueError("Invalid batch size")
        normalized = []
        total = 0
        for req in requests:
            allowed = {"text", "policy", "project_id"}
            if not detect:
                allowed |= {"mode", "patient_id", "date_shift_days", "replacement"}
            if not isinstance(req, dict) or set(req) - allowed:
                raise ValueError("Invalid request fields")
            text = req.get("text")
            if not isinstance(text, str) or len(text) > 100_000:
                raise ValueError("Text must contain at most 100000 characters")
            total += len(text)
            if total > 100_000:
                raise ValueError("A batch may contain at most 100000 characters")
            project = req.get("project_id", "main")
            if not isinstance(project, str) or not re.fullmatch(
                r"[A-Za-z0-9_:-]{1,60}", project
            ):
                raise ValueError("Invalid project ID")
            policy = validate_policy(req.get("policy"))
            item = {
                **req,
                "policy": policy,
                "project_id": project,
                "_extended": req.get("policy") is not None
                or req.get("replacement") is not None,
            }
            if not detect:
                mode = req.get("mode", "redact")
                if mode not in {"redact", "synthetic"}:
                    raise ValueError("mode must be redact or synthetic")
                anchor = req.get("patient_id")
                if anchor is not None and (
                    not isinstance(anchor, str) or not 1 <= len(anchor) <= 256
                ):
                    raise ValueError("Invalid patient anchor")
                shift = req.get("date_shift_days")
                try:
                    replacement = req.get("replacement")
                    if isinstance(replacement, ReplacementOptions):
                        replacement = replacement.model_dump()
                    options = (
                        ReplacementOptions.model_validate(replacement)
                        if replacement is not None
                        else legacy_options(mode)
                    )
                except (ValueError, TypeError):
                    raise ValueError("Invalid replacement options") from None
                labels = KNOWN_LABELS | {item.label for item in policy.custom_entities}
                if set(options.per_type) - labels:
                    raise ValueError("Unknown replacement entity type")
                synthetic = (
                    options.type == "synthetic"
                    or "synthetic" in options.per_type.values()
                )
                if shift is not None and (
                    not synthetic
                    or type(shift) is not int
                    or not 1 <= abs(shift) <= 3650
                ):
                    raise ValueError("Invalid date shift")
                if synthetic or options.entity_ids == "keyed":
                    validated_secret(self.secret)
                    if not isinstance(anchor, str) or not anchor.strip():
                        raise ValueError("Replacement requires a stable patient_id")
                item.update(
                    mode=mode,
                    patient_id=anchor,
                    date_shift_days=shift,
                    replacement=options,
                )
            normalized.append(item)
        return normalized

    @staticmethod
    def _shared_options(requests):
        """Sharing context is explicit and cannot cross patient/project boundaries."""
        first = requests[0]
        for req in requests[1:]:
            if any(
                req.get(key) != first.get(key)
                for key in (
                    "policy",
                    "project_id",
                    "patient_id",
                    "mode",
                    "replacement",
                    "date_shift_days",
                )
            ):
                raise ValueError(
                    "Shared context requires identical policies and patient scope"
                )

    def _infer(self, texts):
        predictions = []
        for offset in range(0, len(texts), 16):
            batch = texts[offset : offset + 16]
            # Hold the detector only per model batch so a large structured
            # request cannot monopolize it; other requests interleave between.
            with self.lock:
                if hasattr(self.detector, "detect_batch"):
                    found = self.detector.detect_batch(batch)
                else:
                    found = [self.detector.detect(text) for text in batch]
            if not isinstance(found, (list, tuple)) or len(found) != len(batch):
                raise RuntimeError("Detector returned invalid batch output")
            predictions.extend(
                checked_entities(text, spans) for text, spans in zip(batch, found)
            )
        return predictions

    def _detect(self, requests, *, link_batch=False, shared_entities=False):
        from outis.policy import apply_policy
        from outis.replacement import transform

        if type(link_batch) is not bool or type(shared_entities) is not bool:
            raise ValueError("Invalid context options")
        texts = [req["text"] for req in requests]
        context = None
        if link_batch or shared_entities:
            self._shared_options(requests)
            context = "\n".join(texts)
            if len(context) > 100_000:
                raise ValueError("Linked input exceeds 100000 characters")
        if link_batch:
            spans = apply_policy(
                context, self._infer([context])[0], requests[0]["policy"]
            )
            predictions = None
        else:
            predictions = [
                apply_policy(text, spans, req["policy"])
                for text, spans, req in zip(texts, self._infer(texts), requests)
            ]
            if shared_entities:
                spans = []
                start = 0
                for text, detected in zip(texts, predictions):
                    spans.extend(
                        with_fields(
                            span, start=span.start + start, end=span.end + start
                        )
                        for span in detected
                    )
                    start += len(text) + 1
        if context is not None:
            req = requests[0]
            keyed = getattr(req.get("replacement"), "entity_ids", None) == "keyed"
            shared = transform.prepare_context(
                context,
                spans,
                secret=self.secret if keyed else None,
                patient_id=req.get("patient_id") if keyed else None,
            )
            prepared = [shared] * len(requests)
            context_spans = shared.spans
            predictions = []
            start = 0
            first_span = 0
            for text in texts:
                end = start + len(text)
                local = []
                while (
                    first_span < len(context_spans)
                    and context_spans[first_span].end <= start
                ):
                    first_span += 1
                for span_index in range(first_span, len(context_spans)):
                    span = context_spans[span_index]
                    if span.start >= end:
                        break
                    if span.start < end and span.end > start:
                        clipped = span.start < start or span.end > end
                        local.append(
                            with_fields(
                                span,
                                start=max(span.start, start) - start,
                                end=min(span.end, end) - start,
                                source="context_fragment" if clipped else span.source,
                            )
                        )
                predictions.append(local)
                start = end + 1
        else:
            # Each request's identity analysis is computed here once and
            # reused by rendering instead of linking the same spans again.
            prepared = []
            for text, spans, req in zip(texts, predictions, requests):
                keyed = getattr(req.get("replacement"), "entity_ids", None) == "keyed"
                prepared.append(
                    transform.prepare_context(
                        text,
                        spans,
                        secret=self.secret if keyed else None,
                        patient_id=req.get("patient_id") if keyed else None,
                    )
                )
            predictions = [list(item.spans) for item in prepared]
        return predictions, context, prepared

    def detect(self, text, *, policy=None, project_id="main"):
        return self.detect_batch(
            [{"text": text, "policy": policy, "project_id": project_id}]
        )[0]

    def detect_batch(self, requests, *, link_batch=False, shared_entities=False):
        from outis.policy import policy_metadata

        requests = self._requests(requests, detect=True)
        with self.usage.track(
            [r["project_id"] for r in requests],
            [len(r["text"]) for r in requests],
            "detect",
        ):
            predictions, _, _ = self._detect(
                requests, link_batch=link_batch, shared_entities=shared_entities
            )
            return [
                {
                    "entities": [entity.as_dict() for entity in spans],
                    "entity_count": len(spans),
                    "model_version": self.version,
                    "policy": policy_metadata(req["policy"]),
                }
                for req, spans in zip(requests, predictions)
            ]

    def deidentify(
        self,
        text,
        *,
        mode="redact",
        patient_id=None,
        date_shift_days=None,
        policy=None,
        replacement=None,
        project_id="main",
    ):
        return self.deidentify_batch(
            [
                {
                    "text": text,
                    "mode": mode,
                    "patient_id": patient_id,
                    "date_shift_days": date_shift_days,
                    "policy": policy,
                    "replacement": replacement,
                    "project_id": project_id,
                }
            ]
        )[0]

    def deidentify_batch(self, requests, *, link_batch=False, shared_entities=False):
        return self._process(
            requests, link_batch=link_batch, shared_entities=shared_entities, limit=16
        )

    def process_segments(self, requests, *, link_batch=False, shared_entities=True):
        """Bounded internal field batching for structured documents."""
        if requests == []:
            return []
        return self._process(
            requests, link_batch=link_batch, shared_entities=shared_entities, limit=4096
        )

    def _process(self, requests, *, link_batch, shared_entities, limit):
        from outis.policy import policy_metadata
        from outis.replacement.transform import legacy_response, reidentify, render

        requests = self._requests(requests, limit=limit)
        with self.usage.track(
            [r["project_id"] for r in requests],
            [len(r["text"]) for r in requests],
            "deidentify",
        ):
            predictions, context, prepared = self._detect(
                requests, link_batch=link_batch, shared_entities=shared_entities
            )
            results = []
            mention_offset = 0
            context_offset = 0
            for req, spans, context_index in zip(requests, predictions, prepared):
                result = render(
                    req["text"],
                    spans,
                    options=req["replacement"],
                    mode=req["mode"],
                    patient_id=req["patient_id"],
                    secret=self.secret,
                    date_shift_days=req["date_shift_days"],
                    prepared_context=context_index,
                    index_offset=mention_offset if context is not None else 0,
                    context_offset=context_offset if context is not None else 0,
                )
                if req["_extended"] or link_batch or shared_entities:
                    result["policy"] = policy_metadata(req["policy"])
                else:
                    result = legacy_response(result)
                result.update(
                    model_version=self.version,
                    mode=req["mode"],
                    entity_count=len(result["entities"]),
                )
                results.append(result)
                mention_offset += len(result["entities"])
                context_offset += len(req["text"]) + 1
            if sum(len(result["text"]) for result in results) > 1_000_000:
                raise ValueError("Batch replacement output exceeds the size limit")
            if (
                sum(
                    len(item["marker"]) + len(item["original"])
                    for result in results
                    for item in result.get("mappings", [])
                )
                > 2_000_000
            ):
                raise ValueError("Batch restoration mappings exceed the size limit")
            if link_batch or shared_entities:
                # Returned mappings belong to one explicitly shared namespace.
                # Distinct aliases cannot silently reuse a reversible token.
                mappings = {}
                for result in results:
                    for mapping in result.get("mappings", []):
                        old = mappings.setdefault(mapping["marker"], mapping)
                        if old != mapping:
                            raise ValueError(
                                "Ambiguous restoration mapping across shared inputs"
                            )
                reidentify("", list(mappings.values()))
            return results

    def analyze(
        self,
        text,
        *,
        policy=None,
        project_id="main",
        locale="en-US",
        include_sensitive=False,
    ):
        from outis.analysis import analyze_entities
        from outis.policy import policy_metadata

        if locale not in {"en-US", "en-GB"} or type(include_sensitive) is not bool:
            raise ValueError("Invalid analysis options")
        requests = self._requests(
            [{"text": text, "policy": policy, "project_id": project_id}], detect=True
        )
        with self.usage.track([project_id], [len(text)], "analyze"):
            predictions, _, _ = self._detect(requests)
            result = analyze_entities(
                text, predictions[0], locale=locale, include_sensitive=include_sensitive
            )
            result.update(
                model_version=self.version,
                entity_count=len(predictions[0]),
                policy=policy_metadata(requests[0]["policy"]),
            )
            return result
