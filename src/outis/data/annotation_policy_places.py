"""Executable examples for the clinical-phi-v2 places addendum.

See docs/annotation-policy-v2-places.md. Kept separate from
``annotation_policy`` so source hashes recorded by earlier generators remain
reproducible. Like the base fixtures, these are authoring and evaluation
examples, not a detector dictionary.
"""

from __future__ import annotations

import hashlib
import unicodedata

from outis.data.annotation_policy import _example


ADDENDUM = "clinical-phi-v2-places"
VALUE_POOLS = ("train", "heldout")


def place_value_pool(value: str, expansion: str | None = None) -> str:
    """Assign a real place or institution value to its train or heldout pool.

    An abbreviation passes its expansion so both share one pool.
    """
    key = expansion if expansion is not None else value
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Place value must be a nonempty string")
    digest = hashlib.sha256(unicodedata.normalize("NFC", key.strip()).casefold().encode("utf-8")).hexdigest()
    return "train" if int(digest[:8], 16) % 10 < 8 else "heldout"


PLACE_FIXTURES = (
    _example("p1_patient_locality", "Pt from [[LOCATION|Chicago]]; daughter lives in [[LOCATION|Tacoma, WA]].",
             ["p1_patient_locality"]),
    _example("p2_care_site_abbreviation", "Seen at [[FACILITY|UCSF]] in May; transferred from [[FACILITY|MGH]].",
             ["p2_care_site_abbreviation"]),
    _example("p3_care_site_without_type", "Followed at [[FACILITY|Mt. Sinai]]; ED visit at [[FACILITY|Houston Memorial]].",
             ["p3_care_site_without_type"]),
    _example("p4_full_care_site", "Admitted to [[FACILITY|St. Francis Hospital]] after a fall.",
             ["p4_full_care_site", "leading_abbreviation"]),
    _example("p5_care_site_then_locality", "Treated at [[FACILITY|Mayo Clinic]] in [[LOCATION|Rochester, MN]].",
             ["p5_care_site_then_locality"]),
    _example("p6_clinician_affiliation", "Dr. [[NAME|Priya Testier]] from [[FACILITY|UCSF]] reviewed the scan.",
             ["p6_clinician_affiliation", "name_boundaries"]),
    _example("n1_named_instruments", "Manometry per the Chicago Classification; Ottawa ankle rules negative; Framingham Risk Score 12%.",
             ["n1_named_instrument"]),
    _example("n2_place_diseases", "Tested for Lyme disease, West Nile virus and Rocky Mountain spotted fever.",
             ["n2_place_disease"]),
    _example("n3_institution_as_source", "Per Mayo Clinic guidelines and a Johns Hopkins study, continue the statin.",
             ["n3_institution_source"]),
    _example("n4_state_or_country_alone", "Valley fever is common in Arizona; she plans travel to Mexico.",
             ["n4_state_or_country"]),
    _example("n5_place_word_as_person_or_word", "[[NAME|Florence Testier]] needs reading glasses and a mobile phone charger.",
             ["n5_place_word_other_role", "name_boundaries"]),
    _example("mixed_care_beats_citation", "Seen at [[FACILITY|Mayo Clinic]]; per Mayo Clinic guidelines, start aspirin.",
             ["mixed_roles", "p3_care_site_without_type", "n3_institution_source"]),
    _example("state_inside_locality", "Lives in [[LOCATION|Tacoma, WA]], not elsewhere in Washington.",
             ["p1_patient_locality", "n4_state_or_country"]),
)
