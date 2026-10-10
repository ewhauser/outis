"""Bounded HTTP and WebSocket interfaces with PHI-safe failures."""

from __future__ import annotations

import hmac
import json
import os
from contextlib import asynccontextmanager, suppress
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from outis.paths import DEFAULT_MODEL_DIR
from outis.policy import DetectionPolicy
from outis.replacement.transform import ReidentifyRequest, ReplacementOptions
from outis.service import Deidentifier

MAX_REQUEST_BYTES = 16 * 1024 * 1024
_FAILURE = "Deidentification failed; no output released"


class SafeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class DeidentifyOptions(SafeModel):
    mode: Literal["redact", "synthetic"] = "redact"
    patient_id: str | None = Field(default=None, min_length=1, max_length=256)
    date_shift_days: int | None = Field(default=None, ge=-3650, le=3650, strict=True)
    policy: DetectionPolicy | None = None
    replacement: ReplacementOptions | None = None
    project_id: str = Field(
        default="main", pattern=r"^[A-Za-z0-9_:-]{1,60}$", min_length=1, max_length=60
    )

    @model_validator(mode="after")
    def validate_policy(self):
        options = self.replacement or ReplacementOptions(
            type="synthetic" if self.mode == "synthetic" else "marker"
        )
        synthetic = (
            options.type == "synthetic" or "synthetic" in options.per_type.values()
        )
        if (synthetic or options.entity_ids == "keyed") and (
            not self.patient_id or not self.patient_id.strip()
        ):
            raise ValueError("Replacement requires patient_id")
        if self.date_shift_days == 0:
            raise ValueError("date_shift_days must be nonzero")
        if not synthetic and self.date_shift_days is not None:
            raise ValueError("Date shifts apply only in synthetic replacement")
        return self


class DeidentifyRequest(DeidentifyOptions):
    text: str = Field(
        max_length=100_000, description="Clinical plain text; Unicode character offsets"
    )


class DetectRequest(SafeModel):
    text: str = Field(max_length=100_000)
    policy: DetectionPolicy | None = None
    project_id: str = Field(
        default="main", pattern=r"^[A-Za-z0-9_:-]{1,60}$", min_length=1, max_length=60
    )


class AnalyzeRequest(DetectRequest):
    locale: Literal["en-US", "en-GB"] = "en-US"
    include_sensitive: bool = False


class BatchRequest(SafeModel):
    items: list[DeidentifyRequest] = Field(min_length=1, max_length=16)
    link_batch: bool = False
    shared_entities: bool = False

    @model_validator(mode="after")
    def limit_total(self):
        if sum(len(item.text) for item in self.items) > 100_000:
            raise ValueError("A batch may contain at most 100000 characters")
        return self


class DetectBatchRequest(SafeModel):
    items: list[DetectRequest] = Field(min_length=1, max_length=16)
    link_batch: bool = False
    shared_entities: bool = False

    @model_validator(mode="after")
    def limit_total(self):
        if sum(len(item.text) for item in self.items) > 100_000:
            raise ValueError("A batch may contain at most 100000 characters")
        return self


class StructuredRequest(SafeModel):
    data: Any
    format: Literal["json", "csv", "xml"] = "json"
    paths: list[str] | None = Field(default=None, min_length=1, max_length=256)
    options: DeidentifyOptions | None = None
    link_records: bool = False


class FileRequest(SafeModel):
    data: str = Field(max_length=MAX_REQUEST_BYTES)
    content_type: str = Field(min_length=1, max_length=128)
    paths: list[str] | None = Field(default=None, min_length=1, max_length=256)
    options: DeidentifyOptions | None = None
    link_records: bool = False


class DetectedEntityResponse(SafeModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    label: str
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    source: str
    entity_id: str | None = None

    @model_validator(mode="after")
    def ordered_span(self):
        if self.end <= self.start:
            raise ValueError("Invalid output span")
        return self


class EntityResponse(DetectedEntityResponse):
    replacement: str
    processed_start: int | None = Field(default=None, ge=0)
    processed_end: int | None = Field(default=None, ge=0)


class PreservationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True)
    clinical_text_outside_spans_preserved: bool = Field(
        description="Characters outside detected spans are copied exactly; detection errors can still alter clinical meaning."
    )
    date_intervals_preserved: bool = Field(
        description="Synthetic mode preserves intervals among detected, supported DATE spans; false if a detected date is redacted. Undetected dates are not covered."
    )
    exact_dates_shifted: int = Field(
        ge=0,
        description="Number of detected supported DATE spans shifted by the patient-wide offset.",
    )
    unshiftable_dates_redacted: int = Field(
        ge=0,
        description="Number of detected DATE spans redacted because synthetic date shifting was unsupported.",
    )
    patient_consistent_replacements: bool = Field(
        description="Synthetic mode uses keyed replacements for full identities, identifiers and dates; partial-name aliases have the documented limits."
    )


class MappingResponse(SafeModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True)
    marker: str
    original: str
    label: str


class DeidentifyResponse(SafeModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True)
    text: str
    mode: str
    entities: list[EntityResponse]
    warnings: list[str]
    preservation: PreservationResponse
    model_version: str
    entity_count: int = Field(ge=0)
    mappings: list[MappingResponse] | None = None
    policy: dict[str, Any] | None = None


class DetectResponse(SafeModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True)
    entities: list[DetectedEntityResponse]
    model_version: str
    entity_count: int = Field(ge=0)
    policy: dict[str, Any] | None = None


class AnalyzedEntityResponse(DetectedEntityResponse):
    analysis: dict[str, Any] | None = None


class RelationshipResponse(SafeModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True)
    type: str
    source_entity: int = Field(ge=0)
    target_entity: int = Field(ge=0)
    source: str


class AnalyzeResponse(SafeModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True)
    entities: list[AnalyzedEntityResponse]
    relationships: list[RelationshipResponse]
    metadata: dict[str, Any]
    model_version: str
    entity_count: int = Field(ge=0)
    policy: dict[str, Any] | None = None


class ReidentifyResponse(SafeModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True)
    text: str
    restored_count: int = Field(ge=0)


class LocatedEntityResponse(EntityResponse):
    location: dict[str, Any]


class StructuredResponse(SafeModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True)
    data: Any
    format: str
    entities: list[LocatedEntityResponse]
    entity_count: int = Field(ge=0)
    warnings: list[str]
    model_version: str | None = None
    mappings: list[MappingResponse] | None = None


class FileResponse(SafeModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True)
    data: str
    content_type: str
    entities: list[LocatedEntityResponse]
    entity_count: int = Field(ge=0)
    warnings: list[str]
    model_version: str | None = None
    mappings: list[MappingResponse] | None = None


class WebSocketRequest(SafeModel):
    operation: Literal["detect", "deidentify", "analyze", "reidentify"]
    payload: dict[str, Any]


def _plain_json(value):
    """Reject non-JSON objects and nonfinite values inside handled boundaries."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        import math

        if not math.isfinite(value):
            raise RuntimeError("Invalid service response")
        return
    if type(value) is list:
        for item in value:
            _plain_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _plain_json(item)
        return
    raise RuntimeError("Invalid service response")


def _validated_result(model, value):
    _plain_json(value)
    result = model.model_validate(value)
    _plain_json(result.model_dump(exclude_none=True))
    # Exercise the actual serializer here too: JSON-shaped data can still
    # contain lone surrogates or oversized integers that fail after validation.
    try:
        result.model_dump_json(exclude_none=True)
    except Exception:
        raise RuntimeError("Invalid service serialization") from None
    return result


def _payload(model):
    # Preserve old injected-service interfaces and omit absent extensions.
    payload = model.model_dump(exclude_none=True, exclude_unset=True)
    # A required nullable value (structured JSON data) is not an absent option.
    for name, field in type(model).model_fields.items():
        if field.is_required() and getattr(model, name) is None:
            payload[name] = None
    return payload


def _safe_call(function):
    try:
        return function()
    except ValidationError:
        raise HTTPException(status_code=500, detail=_FAILURE) from None
    except ValueError:
        raise HTTPException(
            status_code=422, detail="Replacement policy cannot process this request"
        ) from None
    except Exception:
        raise HTTPException(status_code=500, detail=_FAILURE) from None


class RequestBoundaryMiddleware:
    """Limit even chunked bodies before parsing, and prohibit response caching."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def private_send(message):
            if message["type"] == "http.response.start":
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() != b"cache-control"
                ]
                message = {
                    **message,
                    "headers": [*headers, (b"cache-control", b"no-store")],
                }
            await send(message)

        lengths = [
            v for k, v in scope.get("headers", []) if k.lower() == b"content-length"
        ]
        if lengths:
            try:
                invalid = len(lengths) != 1 or int(lengths[0]) < 0
                too_large = int(lengths[0]) > MAX_REQUEST_BYTES
            except (ValueError, OverflowError):
                invalid, too_large = True, False
            if invalid or too_large:
                response = JSONResponse(
                    status_code=413 if too_large else 400,
                    content={
                        "detail": "Request body too large"
                        if too_large
                        else "Invalid request"
                    },
                )
                await response(scope, receive, private_send)
                return
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                await JSONResponse(
                    status_code=413, content={"detail": "Request body too large"}
                )(scope, receive, private_send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, private_send)


def create_app(service=None, *, api_key=None):
    service_arg = service
    configured_key = api_key if api_key is not None else os.environ.get("OUTIS_API_KEY")

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service_arg or Deidentifier(
            os.environ.get("OUTIS_MODEL_DIR", DEFAULT_MODEL_DIR),
            secret=os.environ.get("OUTIS_SECRET"),
            device=os.environ.get("OUTIS_DEVICE", "auto"),
        )
        yield

    app = FastAPI(
        title="Outis",
        version="0.2.0",
        lifespan=lifespan,
        description="Local English clinical detection, configurable redaction, structured documents and patient-consistent replacement. HTTP bodies and WebSocket envelopes are limited to 16 MiB. Performance is established only on synthetic evaluation data.",
    )
    app.add_middleware(RequestBoundaryMiddleware)

    def key_valid(key):
        return not configured_key or bool(
            key
            and hmac.compare_digest(key.encode("utf-8"), configured_key.encode("utf-8"))
        )

    def authorize(x_api_key: str | None = Header(default=None)):
        if not key_valid(x_api_key):
            raise HTTPException(status_code=401, detail="Valid X-API-Key required")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Dynamic dictionary keys and Pydantic input/context may contain PHI.
        known_fields = {
            "body",
            "query",
            "path",
            "header",
            "text",
            "mode",
            "patient_id",
            "date_shift_days",
            "items",
            "policy",
            "replacement",
            "project_id",
            "link_batch",
            "shared_entities",
            "data",
            "format",
            "paths",
            "options",
            "link_records",
            "content_type",
            "locale",
            "include_sensitive",
            "mappings",
            "allowed_types",
            "restore_sensitive",
        }
        errors = [
            {
                "loc": [
                    part if isinstance(part, int) or part in known_fields else "[field]"
                    for part in e["loc"]
                ],
                "type": e["type"],
            }
            for e in exc.errors()[:32]
        ]
        return JSONResponse(
            status_code=422,
            content={"detail": "Invalid request", "errors": errors},
            headers={"Cache-Control": "no-store"},
        )

    @app.exception_handler(Exception)
    async def runtime_error(request, exc):
        return JSONResponse(
            status_code=500,
            content={"detail": _FAILURE},
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready", dependencies=[Depends(authorize)])
    def ready(request: Request):
        instance = request.app.state.service
        from outis.replacement.surrogates import _validated_secret

        try:
            _validated_secret(instance.secret)
            synthetic_enabled = True
        except ValueError:
            synthetic_enabled = False
        return {
            "status": "ready",
            "model_version": instance.version,
            "synthetic_enabled": synthetic_enabled,
        }

    def single(instance, operation, payload):
        if operation == "reidentify":
            from outis.replacement.transform import reidentify

            return _validated_result(
                ReidentifyResponse, reidentify(**_payload(payload))
            )
        model = {
            "deidentify": DeidentifyResponse,
            "detect": DetectResponse,
            "analyze": AnalyzeResponse,
        }[operation]
        return _validated_result(
            model, getattr(instance, operation)(**_payload(payload))
        )

    def multiple(instance, operation, payload, model):
        options = {
            name: True
            for name in ("link_batch", "shared_entities")
            if getattr(payload, name)
        }
        values = getattr(instance, operation)(
            [_payload(item) for item in payload.items], **options
        )
        if not isinstance(values, list) or len(values) != len(payload.items):
            raise RuntimeError("Invalid service batch result")
        return [_validated_result(model, value) for value in values]

    @app.post(
        "/v1/deidentify",
        response_model=DeidentifyResponse,
        response_model_exclude_none=True,
        dependencies=[Depends(authorize)],
    )
    def deidentify(payload: DeidentifyRequest, request: Request):
        return _safe_call(
            lambda: single(request.app.state.service, "deidentify", payload)
        )

    @app.post(
        "/v1/batch",
        response_model=list[DeidentifyResponse],
        response_model_exclude_none=True,
        dependencies=[Depends(authorize)],
    )
    def batch(payload: BatchRequest, request: Request):
        return _safe_call(
            lambda: multiple(
                request.app.state.service,
                "deidentify_batch",
                payload,
                DeidentifyResponse,
            )
        )

    @app.post(
        "/v1/detect",
        response_model=DetectResponse,
        response_model_exclude_none=True,
        dependencies=[Depends(authorize)],
    )
    def detect(payload: DetectRequest, request: Request):
        return _safe_call(lambda: single(request.app.state.service, "detect", payload))

    @app.post(
        "/v1/detect/batch",
        response_model=list[DetectResponse],
        response_model_exclude_none=True,
        dependencies=[Depends(authorize)],
    )
    def detect_batch(payload: DetectBatchRequest, request: Request):
        return _safe_call(
            lambda: multiple(
                request.app.state.service, "detect_batch", payload, DetectResponse
            )
        )

    @app.post(
        "/v1/analyze",
        response_model=AnalyzeResponse,
        response_model_exclude_none=True,
        dependencies=[Depends(authorize)],
    )
    def analyze(payload: AnalyzeRequest, request: Request):
        return _safe_call(lambda: single(request.app.state.service, "analyze", payload))

    @app.post(
        "/v1/reidentify",
        response_model=ReidentifyResponse,
        dependencies=[Depends(authorize)],
    )
    def reidentify(payload: ReidentifyRequest, request: Request):
        return _safe_call(
            lambda: single(request.app.state.service, "reidentify", payload)
        )

    # Omit optional defaults while retaining a required data:null response.
    @app.post(
        "/v1/structured",
        response_model=StructuredResponse,
        response_model_exclude_defaults=True,
        dependencies=[Depends(authorize)],
    )
    def structured(payload: StructuredRequest, request: Request):
        from outis.formats.structured import process_structured

        return _safe_call(
            lambda: _validated_result(
                StructuredResponse,
                process_structured(request.app.state.service, **_payload(payload)),
            )
        )

    @app.post(
        "/v1/files",
        response_model=FileResponse,
        response_model_exclude_none=True,
        dependencies=[Depends(authorize)],
    )
    def files(payload: FileRequest, request: Request):
        from outis.formats.files import process_file

        return _safe_call(
            lambda: _validated_result(
                FileResponse,
                process_file(request.app.state.service, **_payload(payload)),
            )
        )

    @app.get("/v1/policies", dependencies=[Depends(authorize)])
    def policies():
        from outis.policy import EXTENDED_TYPES, MEDICAL_TYPES, POLICY_VERSION
        from outis.types import ENTITY_TYPES

        return {
            "version": POLICY_VERSION,
            "profiles": ["clinical_default", "identifiers_only", "extended"],
            "entity_types": list(ENTITY_TYPES),
            "extended_types": list(EXTENDED_TYPES),
            "medical_types": list(MEDICAL_TYPES),
            "language_scope": ["en"],
            "limits": {
                "text_characters": 100_000,
                "batch_items": 16,
                "request_bytes": MAX_REQUEST_BYTES,
                "custom_entities": 32,
            },
        }

    @app.get("/v1/usage", dependencies=[Depends(authorize)])
    def usage(request: Request):
        def result():
            value = request.app.state.service.usage.snapshot()
            _plain_json(value)
            return JSONResponse(content=value)

        return _safe_call(result)

    @app.get(
        "/metrics", dependencies=[Depends(authorize)], response_class=PlainTextResponse
    )
    def metrics(request: Request):
        def result():
            value = request.app.state.service.usage.prometheus()
            if not isinstance(value, str):
                raise RuntimeError("Invalid metrics response")
            return PlainTextResponse(value, media_type="text/plain; version=0.0.4")

        return _safe_call(result)

    @app.get("/diagnostics", dependencies=[Depends(authorize)])
    def diagnostics(request: Request):
        from outis.operations import diagnostics as diagnose

        def result():
            value = diagnose(request.app.state.service)
            _plain_json(value)
            return JSONResponse(content=value)

        return _safe_call(result)

    @app.websocket("/v1/ws")
    async def websocket(socket: WebSocket):
        origin = socket.headers.get("origin")
        allowed_origin = not origin
        if origin:
            try:
                parsed = urlsplit(origin)
                expected_scheme = "https" if socket.url.scheme == "wss" else "http"
                expected_port = socket.url.port or (
                    443 if expected_scheme == "https" else 80
                )
                origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
                allowed_origin = (
                    parsed.scheme == expected_scheme
                    and parsed.hostname == socket.url.hostname
                    and origin_port == expected_port
                    and parsed.username is None
                    and parsed.password is None
                    and not parsed.path
                    and not parsed.query
                    and not parsed.fragment
                )
            except ValueError:
                allowed_origin = False
        if not key_valid(socket.headers.get("x-api-key")) or not allowed_origin:
            await socket.close(code=1008)
            return
        await socket.accept(headers=[(b"cache-control", b"no-store")])
        schemas = {
            "deidentify": DeidentifyRequest,
            "detect": DetectRequest,
            "analyze": AnalyzeRequest,
            "reidentify": ReidentifyRequest,
        }
        while True:
            try:
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                raw = message.get("text")
                if raw is None:
                    await socket.close(code=1003)
                    return
                if len(raw.encode("utf-8")) > MAX_REQUEST_BYTES:
                    await socket.close(code=1009)
                    return
                try:
                    envelope = WebSocketRequest.model_validate(json.loads(raw))
                    payload = schemas[envelope.operation].model_validate(
                        envelope.payload
                    )
                except (ValueError, TypeError, RecursionError):
                    await socket.send_json(
                        {"error": {"status": 422, "detail": "Invalid request"}}
                    )
                    continue
                try:
                    # The closure runs before the next message is read.
                    result = await run_in_threadpool(
                        _safe_call,
                        lambda: single(
                            socket.app.state.service,
                            envelope.operation,  # noqa: B023
                            payload,  # noqa: B023
                        ),
                    )
                    value = {
                        "operation": envelope.operation,
                        "result": result.model_dump(exclude_none=True),
                    }
                    await socket.send_json(value)
                except HTTPException as exc:
                    await socket.send_json(
                        {"error": {"status": exc.status_code, "detail": exc.detail}}
                    )
                except Exception:
                    await socket.send_json(
                        {"error": {"status": 500, "detail": _FAILURE}}
                    )
            except WebSocketDisconnect:
                return
            except Exception:
                # Transport/serialization exceptions may retain the entire frame.
                # Close without passing those objects to ASGI error logging.
                with suppress(Exception):
                    await socket.close(code=1011)
                return

    return app


app = create_app()
