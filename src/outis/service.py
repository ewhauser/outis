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


# The model_version of a replace response: no model ran on that request.
SUPPLIED_VERSION = "supplied"


class _NoDetector:
    def detect_batch(self, texts):
        raise RuntimeError("This instance replaces supplied spans only")


def _scoped_project(req):
    """The project that keys identities, or None in patient scope."""
    scope = getattr(req.get("replacement"), "scope", "patient")
    return req["project_id"] if scope == "project" else None


class Deidentifier:
    def __init__(
        self,
        model_dir=None,
        *,
        secret=None,
        device="auto",
        detector=None,
        site_dictionaries=None,
        fingerprint_secret=None,
    ):
        # Validate operator configuration before any model load.
        if site_dictionaries is not None:
            from outis.site_dictionaries import load_site_dictionaries

            site_dictionaries = load_site_dictionaries(site_dictionaries)
        if detector is None:
            from outis.detection.hf_encoder import load_detector

            model_dir = resolve_model_dir(model_dir)
            detector = load_detector(model_dir, device=device)
        self.detector = detector
        self.secret = secret
        # Text fingerprints have their own key so the surrogate key can be
        # rotated without invalidating span files; it defaults to the secret.
        self.fingerprint_secret = (
            secret if fingerprint_secret is None else fingerprint_secret
        )
        self.lock = Lock()
        self.usage = Usage()
        self.version = _model_version(model_dir, detector)
        self.site_dictionaries = site_dictionaries

    @classmethod
    def for_replacement(cls, *, secret=None, fingerprint_secret=None):
        """An instance that only replaces supplied spans and loads no model."""
        return cls(
            secret=secret, fingerprint_secret=fingerprint_secret, detector=_NoDetector()
        )

    def _requests(self, requests, *, detect=False, limit=16, supplied=False):
        from outis.policy import KNOWN_LABELS, site_selection, validate_policy
        from outis.replacement.transform import ReplacementOptions, legacy_options
        from outis.spans import (
            check_fingerprint,
            supplied_entities,
            text_fingerprint,
            validate_replace_policy,
        )

        if not isinstance(requests, (list, tuple)) or not 1 <= len(requests) <= limit:
            raise ValueError("Invalid batch size")
        normalized = []
        total = 0
        for req in requests:
            allowed = {"text", "policy", "project_id"}
            if detect:
                allowed.add("include_fingerprint")
            else:
                allowed |= {"mode", "patient_id", "date_shift_days", "replacement"}
            if supplied:
                allowed |= {"entities", "text_fingerprint"}
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
            if supplied:
                # Supplied spans were already detected: only ages still applies.
                policy = validate_replace_policy(req.get("policy"))
            else:
                policy = validate_policy(req.get("policy"))
                site_selection(self.site_dictionaries, policy)
            item = {
                **req,
                "policy": policy,
                "project_id": project,
                "_extended": req.get("policy") is not None
                or req.get("replacement") is not None,
            }
            if detect and type(req.get("include_fingerprint", False)) is not bool:
                raise ValueError("include_fingerprint must be a boolean")
            if detect and req.get("include_fingerprint"):
                # Fail before detection runs, not after.
                text_fingerprint(self.fingerprint_secret, "")
            if supplied:
                item["entities"] = supplied_entities(text, req.get("entities"))
                if req.get("text_fingerprint") is not None:
                    check_fingerprint(
                        self.fingerprint_secret, text, req["text_fingerprint"]
                    )
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
                if options.scope == "project" and "project_id" not in req:
                    # The default project would silently share one key.
                    raise ValueError("Project scope requires an explicit project_id")
                # Supplied spans may carry any valid custom label, which the
                # options model has already checked.
                if not supplied:
                    labels = KNOWN_LABELS | {
                        item.label for item in policy.custom_entities
                    }
                    if self.site_dictionaries is not None:
                        labels |= self.site_dictionaries.labels
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
            # Malformed detector output is a server defect, not a client
            # error; keep it out of the ValueError (HTTP 422) contract.
            try:
                predictions.extend(
                    checked_entities(text, spans) for text, spans in zip(batch, found)
                )
            except ValueError:
                raise RuntimeError("Detector returned invalid output") from None
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
                context,
                self._infer([context])[0],
                requests[0]["policy"],
                self.site_dictionaries,
            )
            predictions = None
        else:
            predictions = [
                apply_policy(text, spans, req["policy"], self.site_dictionaries)
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
                project_id=_scoped_project(req) if keyed else None,
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
            predictions, prepared = self._prepare(requests, predictions)
        return predictions, context, prepared

    def _prepare(self, requests, predictions):
        """Link each request's spans once; rendering reuses the analysis."""
        from outis.replacement import transform

        prepared = []
        for spans, req in zip(predictions, requests):
            keyed = getattr(req.get("replacement"), "entity_ids", None) == "keyed"
            prepared.append(
                transform.prepare_context(
                    req["text"],
                    spans,
                    secret=self.secret if keyed else None,
                    patient_id=req.get("patient_id") if keyed else None,
                    project_id=_scoped_project(req) if keyed else None,
                )
            )
        return [list(item.spans) for item in prepared], prepared

    def _supplied(self, requests):
        """Prepare supplied spans exactly as _detect prepares detected ones."""
        from outis.spans import filter_ages

        spans = [
            filter_ages(req["text"], req["entities"], req["policy"].ages)
            for req in requests
        ]
        predictions, prepared = self._prepare(requests, spans)
        return predictions, None, prepared

    def detect(
        self, text, *, policy=None, project_id="main", include_fingerprint=False
    ):
        request = {"text": text, "policy": policy, "project_id": project_id}
        if include_fingerprint:
            request["include_fingerprint"] = include_fingerprint
        return self.detect_batch([request])[0]

    def detect_batch(self, requests, *, link_batch=False, shared_entities=False):
        from outis.policy import policy_metadata
        from outis.spans import text_fingerprint

        requests = self._requests(requests, detect=True)
        with self.usage.track(
            [r["project_id"] for r in requests],
            [len(r["text"]) for r in requests],
            "detect",
        ):
            predictions, _, _ = self._detect(
                requests, link_batch=link_batch, shared_entities=shared_entities
            )
            results = []
            for req, spans in zip(requests, predictions):
                result = {
                    "entities": [entity.as_dict() for entity in spans],
                    "entity_count": len(spans),
                    "model_version": self.version,
                    "policy": policy_metadata(req["policy"], self.site_dictionaries),
                }
                if req.get("include_fingerprint"):
                    result["text_fingerprint"] = text_fingerprint(
                        self.fingerprint_secret, req["text"]
                    )
                results.append(result)
            return results

    def deidentify(
        self,
        text,
        *,
        mode="redact",
        patient_id=None,
        date_shift_days=None,
        policy=None,
        replacement=None,
        project_id=None,
    ):
        request = {
            "text": text,
            "mode": mode,
            "patient_id": patient_id,
            "date_shift_days": date_shift_days,
            "policy": policy,
            "replacement": replacement,
        }
        if project_id is not None:
            # Omitted means the default project, which project scope rejects.
            request["project_id"] = project_id
        return self.deidentify_batch([request])[0]

    def deidentify_batch(self, requests, *, link_batch=False, shared_entities=False):
        return self._process(
            requests, link_batch=link_batch, shared_entities=shared_entities, limit=16
        )

    def replace(
        self,
        text,
        entities,
        *,
        mode="redact",
        patient_id=None,
        date_shift_days=None,
        policy=None,
        replacement=None,
        project_id=None,
        text_fingerprint=None,
    ):
        """Replace caller-supplied spans with the options of ``deidentify``.

        No detection runs. Entities are ``{start, end, label}`` mappings or
        objects; score is carried through and entity_id and source are ignored.
        Given the spans ``detect`` returned, the text, entities (apart from
        source), warnings, preservation and mappings equal ``deidentify``'s.
        """
        request = {
            "text": text,
            "entities": entities,
            "mode": mode,
            "patient_id": patient_id,
            "date_shift_days": date_shift_days,
            "policy": policy,
            "replacement": replacement,
        }
        if project_id is not None:
            request["project_id"] = project_id
        if text_fingerprint is not None:
            request["text_fingerprint"] = text_fingerprint
        return self.replace_batch([request])[0]

    def replace_batch(self, requests):
        return self._process(
            requests, link_batch=False, shared_entities=False, limit=16, supplied=True
        )

    def process_segments(self, requests, *, link_batch=False, shared_entities=True):
        """Bounded internal field batching for structured documents."""
        if requests == []:
            return []
        return self._process(
            requests, link_batch=link_batch, shared_entities=shared_entities, limit=4096
        )

    def _process(self, requests, *, link_batch, shared_entities, limit, supplied=False):
        from outis.policy import policy_metadata
        from outis.replacement.transform import legacy_response, reidentify, render
        from outis.spans import replace_policy_metadata

        requests = self._requests(requests, limit=limit, supplied=supplied)
        with self.usage.track(
            [r["project_id"] for r in requests],
            [len(r["text"]) for r in requests],
            "replace" if supplied else "deidentify",
        ):
            if supplied:
                predictions, context, prepared = self._supplied(requests)
            else:
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
                    project_id=req["project_id"],
                )
                if supplied and req["_extended"]:
                    result["policy"] = replace_policy_metadata(req["policy"])
                elif req["_extended"] or link_batch or shared_entities:
                    result["policy"] = policy_metadata(
                        req["policy"], self.site_dictionaries
                    )
                else:
                    result = legacy_response(result)
                result.update(
                    model_version=SUPPLIED_VERSION if supplied else self.version,
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
                policy=policy_metadata(requests[0]["policy"], self.site_dictionaries),
            )
            return result
