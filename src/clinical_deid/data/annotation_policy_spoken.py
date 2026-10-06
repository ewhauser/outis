"""Executable examples for spoken and spelled PHI, decisions S1 to S5 (2026-10-06).

See docs/annotation-policy-v2.md and docs/spoken-phi-bank-brief.md. Kept in
its own module so source hashes recorded by earlier generators stay
reproducible. Every name and number is invented.
"""

from __future__ import annotations

from clinical_deid.data.annotation_policy import _example

SPOKEN_FIXTURES = (
    # S1: identifiers said as words are one span of their type, fillers inside; captions outside; a later part is its own span.
    _example(
        "spoken_phone_and_area_code",
        "patient: my callback is [[PHONE|five five five, uh, oh one four two]], area code [[PHONE|six one two]].",
        ["spoken_identifier"],
    ),
    _example(
        "spoken_ssn_mrn_zip",
        "caller: my social is [[SSN|nine one two, four four, eight eight one zero]]. "
        "record number [[MRN|seven seven two zero nine three one]]. zip is [[LOCATION|five three seven two eight]].",
        ["spoken_identifier"],
    ),
    # S2: letter-by-letter spelling takes the type of what is spelled.
    _example(
        "spelled_name",
        "agent: last name? caller: [[NAME|Brannock]], [[NAME|b r a n n o c k]].",
        ["spelled_name"],
    ),
    # S3: spoken email and web addresses are one EMAIL or URL span.
    _example(
        "spoken_email",
        "patient: the email is [[EMAIL|dally dot brannock at example dot com]].",
        ["spoken_email"],
    ),
    # S4: spoken patient-specific dates in any form are DATE; a leading "the" of an ordinal date is inside.
    _example(
        "spoken_dates",
        "patient: date of birth [[DATE|oh three, oh three, forty-one]]. birthday's [[DATE|march third]]. "
        "surgery was back in [[DATE|oh nine]]. born in [[DATE|forty-one]]. got the shot on [[DATE|the twelfth]]. "
        "felt worse last week.",
        ["spoken_date", "relative_date"],
    ),
    # S6: street-level address parts tied to the patient are ADDRESS; a spoken ZIP is LOCATION.
    _example(
        "partial_address_verification",
        "agent: still on [[ADDRESS|wenlow lane]]? patient: yes, [[ADDRESS|four twelve]], [[ADDRESS|apartment two b]]. "
        "agent: zip? patient: [[LOCATION|five three seven two eight]].",
        ["partial_address", "spoken_identifier"],
    ),
    # S7: PHI split across speakers: each piece annotated with the same type; spans never cross a speaker label.
    _example(
        "split_across_turns",
        "agent: date of birth [[DATE|march]]... patient: [[DATE|third, forty-one]]. "
        "agent: so that's [[PHONE|five five five, oh one four two]]? patient: correct.",
        ["split_turns", "read_back", "spoken_date"],
    ),
    # S5: every attempt in a correction is annotated.
    _example(
        "spoken_correction",
        "caller: it's [[PHONE|five five six]], sorry, [[PHONE|five five seven, oh one four two]].",
        ["spoken_identifier", "correction"],
    ),
)
