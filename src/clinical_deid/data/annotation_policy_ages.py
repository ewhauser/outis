"""Executable examples for the all-ages decision (2026-10-06).

Detection annotates every age of the patient or a relative or household
member; the request policy decides what is removed. Supersedes the age
examples in ``POLICY_FIXTURES`` that leave ages of 89 and under unannotated.
Every value is invented.
"""

from __future__ import annotations

from clinical_deid.data.annotation_policy import _example

AGE_FIXTURES = (
    _example(
        "ages_any_value",
        "A [[AGE|72]]-year-old woman; [[AGE|89]] years old at the earlier visit; [[AGE|93]]F; "
        "a [[AGE|six]]-month-old sibling.",
        ["age_boundary", "all_ages"],
    ),
    _example(
        "ages_of_relatives",
        "Mother died at [[AGE|seventy-two]]; brother, [[AGE|58]], is well.",
        ["age_relative", "all_ages"],
    ),
    _example(
        "public_ages_and_durations",
        "Screen from age 45 per guideline; 5-year survival; 10-year history of asthma.",
        ["public_age", "durations"],
    ),
)
