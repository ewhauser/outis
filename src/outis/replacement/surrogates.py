"""Confidentiality-safe redaction and patient-consistent fictional replacement.

All mappings are stateless keyed HMACs.  Neither the mapping key, patient anchor,
source substrings, nor the date shift are returned to callers.  Exact dates are
shifted by one patient-wide offset; imprecise or unsupported dates are redacted
instead of inventing chronology that the source did not establish.
"""

from __future__ import annotations

from typing import Any

from outis.replacement.transform import legacy_replace
from outis.types import Span


def replace(
    text: str,
    spans: list[Span],
    *,
    mode: str = "redact",
    patient_id: str | None = None,
    secret: bytes | str | None = None,
    date_shift_days: int | None = None,
    style: str = "obvious",
    keep_region: bool = True,
) -> dict[str, Any]:
    """Replace supplied PHI spans while copying every other character exactly.

    Entity ``start``/``end`` values index the *input* Unicode string.  Synthetic
    mode requires an application-held key and a stable patient anchor.  The
    patient_id is an exact, case-sensitive opaque anchor, including whitespace.
    Full canonical identities and structured identifiers remain stable across
    requests.  Unambiguous first/last aliases are linked within a note; standalone
    partial names cannot be linked to full identities across different notes.
    Numeric dates use US month/day/year when ambiguous, day/month/year when the
    first component exceeds 12, and year/month/day when year comes first.
    A shifted day/month/year date is emitted year-first if its meaning would
    otherwise become ambiguous. Case-sensitive identifiers use exact keys.
    ``style="realistic"`` selects format-preserving surrogates in reserved
    ranges; ``keep_region`` then keeps a place's state and country.

    This is the legacy entry point; rendering is done by
    ``outis.replacement.transform.render`` through ``legacy_replace``.
    """
    return legacy_replace(
        text,
        spans,
        mode=mode,
        patient_id=patient_id,
        secret=secret,
        date_shift_days=date_shift_days,
        style=style,
        keep_region=keep_region,
    )
