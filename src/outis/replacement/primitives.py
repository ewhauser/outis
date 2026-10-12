"""Keyed digests and span checks shared by every replacement strategy."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import unicodedata

from outis.types import Span


def canonical_form(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def keyed_digest(
    secret: bytes, patient: str, label: str, value: str, salt: int = 0
) -> bytes:
    # The pre-rename key keeps surrogates stable for an existing secret.
    message = json.dumps(
        ["clinical-deid-replacement-v1", patient, label, value, salt],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).digest()


# Versioned tags for project scope. Changing either re-keys every project release.
_PROJECT_KEY_DOMAIN = b"outis-project-key-v1"
_PROJECT_DIGEST_DOMAIN = "outis-project-replacement-v1"


def project_key(secret: bytes, project_id: str) -> bytes:
    """Derive an independent per-project key with single-block HKDF-SHA256.

    Distinct project IDs give unrelated keys, so their surrogates cannot be linked
    without the server secret. The result is key material: never return or log it.
    """
    prk = hmac.new(_PROJECT_KEY_DOMAIN, secret, hashlib.sha256).digest()
    info = json.dumps(
        [_PROJECT_KEY_DOMAIN.decode(), project_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(prk, info + b"\x01", hashlib.sha256).digest()


def project_digest(
    key: bytes, label: str, value: str, salt: int = 0, patient: str | None = None
) -> bytes:
    """Keyed digest under a project key; patient is set only for date offsets."""
    message = json.dumps(
        [_PROJECT_DIGEST_DOMAIN, patient, label, value, salt],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).digest()


def validated_secret(secret: bytes | str | None) -> bytes:
    if isinstance(secret, str):
        secret = secret.encode("utf-8")
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError(
            "Synthetic replacement requires a secret containing at least 32 bytes"
        )
    return secret


def validate_spans(text: str, spans: list[Span]) -> list[Span]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    previous_end = 0
    for span in ordered:
        if (
            not isinstance(span.start, int)
            or isinstance(span.start, bool)
            or not isinstance(span.end, int)
            or isinstance(span.end, bool)
            or span.start < previous_end
            or span.start < 0
            or span.end <= span.start
            or span.end > len(text)
        ):
            raise ValueError("PHI spans must be nonoverlapping valid character ranges")
        previous_end = span.end
    return ordered


def safe_label(label: str) -> str:
    # Unknown model labels can be redacted, but never interpolated as free text.
    return (
        label
        if isinstance(label, str) and re.fullmatch(r"[A-Z_]{1,40}", label)
        else "PHI"
    )
