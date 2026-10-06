"""Executable examples for lab, pathology and radiology decisions L1 to L4 (2026-10-06).

See docs/annotation-policy-v2.md, docs/lab-pathology-bank-brief.md and
docs/radiology-bank-brief.md. Kept in its own module so source hashes recorded
by earlier generators stay reproducible. Every name and number is invented.
"""

from __future__ import annotations

from clinical_deid.data.annotation_policy import _example

REPORT_FIXTURES = (
    # L1: lab identifiers (CLIA and licence numbers) are ID; the performing lab is FACILITY.
    _example(
        "lab_identifiers",
        "Performed at [[FACILITY|Ostravel Reference Laboratory]], CLIA [[ID|32D9917046]]. "
        "Specimen [[ID|SP-26-118734]].",
        ["lab_identifier", "specimen_id"],
    ),
    # L2: a person referred to only by initials is NAME.
    _example(
        "initials_only_person",
        "Tech: [[NAME|JKL]]. Verified by: [[NAME|MKT]]. Read by Dr. [[NAME|Quennell]].",
        ["initials_only_name"],
    ),
    # L3: series and image numbers, LOINC and CPT codes, reference ranges, lot and instrument IDs are not annotated.
    _example(
        "report_codes_not_phi",
        "Series 3 image 45. LOINC 2345-7, CPT 71046. Glucose 112 mg/dL (ref 70-99) H. "
        "Reagent lot 4471B, analyzer AU-680.",
        ["non_phi_codes"],
    ),
    # L4: collection, received and reported date-times: date is DATE, time outside; accession is ID.
    _example(
        "report_datetimes",
        "Accession [[ID|RA-7720931]]. Collected [[DATE|03/14/2026]] 07:42; reported [[DATE|Mar 15, 2026]] 13:05.",
        ["date_in_timestamp", "accession_id"],
    ),
)
