"""Executable examples for places decisions P3a and P5b (2026-10-08).

P5b: an official care-site name that ends in its own town, joined by a hyphen
or nothing, is one FACILITY span. P3a: a short care-site name that is also a
place name is FACILITY when it names the site and LOCATION when it names the
place. Every name is invented. These are authoring and evaluation examples,
not a detector dictionary.
"""

from __future__ import annotations

from clinical_deid.data.annotation_policy import _example

FACILITY_NAME_FIXTURES = (
    _example(
        "official_name_ending_in_town",
        "Transferred from [[FACILITY|Northfield Health-Larkin Bay]] on [[DATE|March 3, 2026]].",
        ["p5b_official_name_with_town"],
    ),
    _example(
        "short_name_equal_to_town",
        "Transferred to [[FACILITY|Tessford]] for dialysis; her daughter lives in [[LOCATION|Tessford]].",
        ["p3a_short_name_role"],
    ),
)
