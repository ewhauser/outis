"""Executable examples for transcript decisions T1 and T2 (2026-10-06).

See docs/annotation-policy-v2.md and docs/transcript-bank-brief.md. Kept in
its own module so source hashes recorded by earlier generators stay
reproducible. Every name and code is invented.
"""

from __future__ import annotations

from clinical_deid.data.annotation_policy import _example

TRANSCRIPT_FIXTURES = (
    # T1: a name in a speaker label is NAME; honorifics, roles, punctuation and time-only stamps stay outside.
    _example(
        "speaker_label_names",
        "[00:08:22] dr [[NAME|Marwick]]: any palpitations?\nPatient: no.\n"
        "CLINICIAN ([[NAME|Brannock]]): Take it with food.\nInterpreter: She agrees.",
        ["speaker_label", "time_only_stamp"],
    ),
    # T1: a timestamp with a date keeps its date part as DATE; the clock time stays outside.
    _example(
        "timestamp_with_date",
        "Call started [[DATE|2025-08-24]] 14:02. Note signed [[DATE|Mon, 18 Dec 2023]] at 10:43.",
        ["date_in_timestamp", "date_boundaries"],
    ),
    # T2: session, visit and call identifiers and patient-tied access codes are ID; meeting links stay URL.
    _example(
        "session_and_access_codes",
        "Video session [[ID|VX-4471]] joined from [[URL|https://visit.example.test/r/8812]]; "
        "portal activation code [[ID|K7TQ-2RMB]]. App version 4.12.0.",
        ["session_id", "access_code"],
    ),
)
