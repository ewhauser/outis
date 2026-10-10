"""Executable examples for decision E1 (2026-10-08): eponymous instruments.

A person's or place's name inside a named instrument, score, criterion, sign,
test, manoeuvre, classification or device is not PHI, extending the rule for
disease eponyms (``Wilson disease``) and places addendum N1. A person who
shares the surname is still NAME. These are authoring and evaluation examples,
not a detector dictionary. Every person is invented.
"""

from __future__ import annotations

from outis.data.annotation_policy import _example

EPONYM_FIXTURES = (
    _example(
        "eponymous_instruments_shorthand",
        "wells 4, ddimer neg. bishop 7 -> induce. apgar 8/9. murphy sign neg. romberg neg.",
        ["eponym_instrument", "shorthand"],
    ),
    _example(
        "eponymous_instruments_fields",
        "Braden: 14\nMallampati: II\nChild-Pugh: B (8)\nGlasgow Coma Scale: 15\nOttawa ankle rules: negative",
        ["eponym_instrument", "structured_field"],
    ),
    _example(
        "same_surname_two_roles",
        "Dr. [[NAME|Corin Wells]] reviewed the Wells score; pt [[NAME|Ana Murphy]], murphy sign neg.",
        ["eponym_instrument", "same_surname_person"],
    ),
)
