"""Executable examples for the tk14 targeted-gaps bank (decisions of 2026-10-05).

See docs/annotation-policy-v2.md and docs/targeted-gaps-bank-brief.md. Kept in
its own module so source hashes recorded by earlier generators stay
reproducible. These are authoring and evaluation examples, not a detector
dictionary. Every name is invented.
"""

from __future__ import annotations

from outis.data.annotation_policy import _example

TARGETED_FIXTURES = (
    # Patient-specific years alone in history and screening lists are DATE.
    _example(
        "history_list_years",
        "Prior studies: bone density scan [[DATE|2011]], osteopenia. Shingles vaccine [[DATE|2019]]. "
        "Cholecystectomy ([[DATE|1988]]). Upper endoscopy [[DATE|Mar 2015]], benign; recheck in 5 years.",
        ["history_year", "date_boundaries"],
    ),
    _example(
        "public_years_and_durations_are_not_dates",
        "Per the 2018 society guideline, recheck every 3 years; aspirin for 6 years in the 2009 cohort study.",
        ["public_dates", "durations"],
    ),
    # A ZIP code after a pharmacy name is a separate LOCATION; the pharmacy is FACILITY.
    _example(
        "pharmacy_then_zip",
        "PHARMACY: [[FACILITY|Quarrington Drug]], [[LOCATION|53728]]. Preferred pharmacy: "
        "[[FACILITY|Felsham Apothecary]], [[LOCATION|36184]].",
        ["pharmacy_zip", "facility_boundary"],
    ),
    # A written-out extension is inside the PHONE span.
    _example(
        "phone_written_extension",
        "Call the unit clerk at [[PHONE|(612) 555-0193, extension 2276]]; "
        "billing [[PHONE|719.555.0104 EXT. 318]].",
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

# Follow-up decisions of 2026-10-05 (cases left open by the tk14 author).
TARGETED_FOLLOWUP_FIXTURES = (
    _example(
        "onset_range_and_future_years",
        "Asthma diagnosed in [[DATE|1997]]; smoked [[DATE|1984-2003]]; colon screening next due [[DATE|2028]].",
        ["history_year", "date_range", "future_year"],
    ),
    _example(
        "zip_plus_four",
        "Mailing ZIP: [[LOCATION|53728-4416]].",
        ["zip_plus_four"],
    ),
    _example(
        "prescription_number_is_id",
        "Rx #[[ID|8830167]] refilled; prescription number [[ID|RX-55102]] transferred.",
        ["prescription_number"],
    ),
    _example(
        "pager_is_phone_bare_extension_is_not",
        "Page Dr. [[NAME|Kell Ravelston]] at pager [[PHONE|(612) 555-0177]]; ward phone ext. 4412.",
        ["pager_number", "bare_extension"],
    ),
)
