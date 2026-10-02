"""FastAPI HTTP interface; request text is never logged or echoed in errors."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import os
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .service import Deidentifier


class DeidentifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    text: str = Field(
        max_length=100_000, description="Clinical plain text; Unicode character offsets"
    )
    mode: Literal["redact", "synthetic"] = "redact"
    patient_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Stable patient anchor required for consistent synthetic dates",
    )
    date_shift_days: int | None = Field(
        default=None,
        ge=-3650,
        le=3650,
        strict=True,
        description="Optional explicit nonzero patient-wide shift",
    )

    @model_validator(mode="after")
    def validate_policy(self):
        if self.mode == "synthetic" and (
            not self.patient_id or not self.patient_id.strip()
        ):
            raise ValueError("Synthetic mode requires patient_id")
        if self.date_shift_days == 0:
            raise ValueError("date_shift_days must be nonzero")
        if self.mode == "redact" and self.date_shift_days is not None:
            raise ValueError("Date shifts apply only in synthetic mode")
        return self


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    items: list[DeidentifyRequest] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def limit_total(self):
        if sum(len(item.text) for item in self.items) > 100_000:
            raise ValueError("A batch may contain at most 100000 characters")
        return self


class EntityResponse(BaseModel):
    start: int
    end: int
    label: str
    replacement: str
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    source: str


class PreservationResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid", hide_input_in_errors=True, strict=True
    )
    clinical_text_outside_spans_preserved: bool = Field(
        description="Characters outside detected spans are copied exactly; detection errors can still alter clinical meaning."
    )
    date_intervals_preserved: bool = Field(
        description="Synthetic mode preserves intervals among detected, supported DATE spans; false if a detected date is redacted. Undetected dates are not covered."
    )
    exact_dates_shifted: int = Field(
        ge=0, description="Number of detected supported DATE spans shifted by the patient-wide offset."
    )
    unshiftable_dates_redacted: int = Field(
        ge=0, description="Number of detected DATE spans redacted because synthetic date shifting was unsupported."
    )
    patient_consistent_replacements: bool = Field(
        description="Synthetic mode uses keyed replacements for full identities, identifiers and dates; partial-name aliases have the documented limits."
    )


class DeidentifyResponse(BaseModel):
    text: str
    mode: str
    entities: list[EntityResponse]
    warnings: list[str]
    preservation: PreservationResponse
    model_version: str
    entity_count: int


def create_app(service=None, *, api_key=None):
    service_arg = service
    configured_key = api_key if api_key is not None else os.environ.get("DEID_API_KEY")

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service_arg or Deidentifier(
            os.environ.get("DEID_MODEL_DIR", "models/clinical-adapter"),
            secret=os.environ.get("DEID_SECRET"),
            device=os.environ.get("DEID_DEVICE", "auto"),
            base_dir=os.environ.get("DEID_BASE_DIR"),
        )
        yield

    app = FastAPI(
        title="Clinical Deidentifier",
        version="0.1.0",
        lifespan=lifespan,
        description="Local clinical NER, redaction and patient-consistent synthetic replacement. "
        "Performance is established only on synthetic evaluation data.",
    )

    def authorize(x_api_key: str | None = Header(default=None)):
        if configured_key and (
            not x_api_key
            or not hmac.compare_digest(
                x_api_key.encode("utf-8"), configured_key.encode("utf-8")
            )
        ):
            raise HTTPException(status_code=401, detail="Valid X-API-Key required")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Pydantic errors include request inputs by default, potentially PHI.
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
        }
        errors = [
            {
                "loc": [
                    part if isinstance(part, int) or part in known_fields else "[field]"
                    for part in e["loc"]
                ],
                "type": e["type"],
            }
            for e in exc.errors()
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
            content={"detail": "Deidentification failed; no output released"},
            headers={"Cache-Control": "no-store"},
        )

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready", dependencies=[Depends(authorize)])
    def ready(request: Request):
        instance = request.app.state.service
        from .replacement import _validated_secret

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

    @app.post(
        "/v1/deidentify",
        response_model=DeidentifyResponse,
        dependencies=[Depends(authorize)],
    )
    def deidentify(payload: DeidentifyRequest, request: Request):
        try:
            result = request.app.state.service.deidentify(**payload.model_dump())
            return DeidentifyResponse.model_validate(result)
        except ValidationError:
            # Validate inside this handled boundary: FastAPI's later response
            # validator can otherwise re-raise an exception containing PHI.
            raise HTTPException(
                status_code=500, detail="Deidentification failed; no output released"
            ) from None
        except ValueError:
            raise HTTPException(
                status_code=422, detail="Replacement policy cannot process this request"
            ) from None
        except Exception:
            # Handled HTTP errors avoid ServerErrorMiddleware re-raising the
            # original detector exception to server logs, which could contain PHI.
            raise HTTPException(
                status_code=500, detail="Deidentification failed; no output released"
            ) from None

    @app.post(
        "/v1/batch",
        response_model=list[DeidentifyResponse],
        dependencies=[Depends(authorize)],
    )
    def batch(payload: BatchRequest, request: Request):
        try:
            results = request.app.state.service.deidentify_batch(
                [item.model_dump() for item in payload.items]
            )
            if not isinstance(results, list) or len(results) != len(payload.items):
                raise RuntimeError("Invalid service batch result")
            return [DeidentifyResponse.model_validate(result) for result in results]
        except ValidationError:
            raise HTTPException(
                status_code=500, detail="Deidentification failed; no output released"
            ) from None
        except ValueError:
            raise HTTPException(
                status_code=422, detail="Replacement policy cannot process this request"
            ) from None
        except Exception:
            raise HTTPException(
                status_code=500, detail="Deidentification failed; no output released"
            ) from None

    return app


app = create_app()
