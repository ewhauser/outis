"""Executable examples for the all-ages decision (2026-10-06).

Detection annotates every age of the patient or a relative or household
member; the request policy decides what is removed. Supersedes the age
examples in ``POLICY_FIXTURES`` that leave ages of 89 and under unannotated.
Decisions of 2026-10-07: a hypothetical, example or template person's age is
not annotated, and a decade of any age is AGE on the decade word. Every value
is invented.
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
    _example(
        "hypothetical_ages",
        "A comparison exercise described a person aged 46; for example, a 50-year-old "
        "would qualify; if a patient were 70, the dose would change.",
        ["hypothetical_age"],
    ),
    _example(
        "decades",
        "A woman in her [[AGE|seventies]]; her brother is in his [[AGE|40s]]; "
        "father in his early [[AGE|sixties]]; the 1970s were hard.",
        ["decade_age", "all_ages"],
    ),
)
