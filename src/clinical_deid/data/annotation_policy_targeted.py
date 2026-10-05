"""Executable examples for the tk14 targeted-gaps bank (decisions of 2026-10-05).

See docs/annotation-policy-v2.md and docs/targeted-gaps-bank-brief.md. Kept in
its own module so source hashes recorded by earlier generators stay
reproducible. These are authoring and evaluation examples, not a detector
dictionary. Every name is invented.
"""

from __future__ import annotations

from clinical_deid.data.annotation_policy import _example

TARGETED_FIXTURES = (
    # Patient-specific years alone in history and screening lists are DATE.
    _example(
        "history_list_years",
        "HEALTH MAINTENANCE: DEXA [[DATE|2013]]: T-score -1.8. Td booster [[DATE|2021]]. "
        "Appendectomy ([[DATE|1995]]). Colonoscopy [[DATE|Jul 2017]], normal; repeat in 10 years.",
        ["history_year", "date_boundaries"],
    ),
    _example(
        "public_years_and_durations_are_not_dates",
        "Per the 2021 USPSTF update, screen every 10 years; statin for 5 years in the 2019 trial.",
        ["public_dates", "durations"],
    ),
    # A ZIP code after a pharmacy name is a separate LOCATION; the pharmacy is FACILITY.
    _example(
        "pharmacy_then_zip",
        "PHARMACY: [[FACILITY|Quarrington Drug]], [[LOCATION|45706]]. Preferred pharmacy: "
        "[[FACILITY|Felsham Apothecary]], [[LOCATION|04101]].",
        ["pharmacy_zip", "facility_boundary"],
    ),
    # A written-out extension is inside the PHONE span.
    _example(
        "phone_written_extension",
        "Report to the charge nurse at [[PHONE|(404) 555-0142, extension 1183]]; "
        "office [[PHONE|505.555.0169 EXT. 1175]].",
        ["phone_extension"],
    ),
    # A named funeral home handling the patient's arrangements is FACILITY (service organisation).
    _example(
        "funeral_home_is_facility",
        "Family chose [[FACILITY|Felsham Funeral Chapel]]; [[FACILITY|Orlinde & Sons Funeral Home]] notified at 0930. "
        "The funeral home will call the family.",
        ["service_organisation", "funeral_home"],
    ),
)
