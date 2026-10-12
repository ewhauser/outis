"""Patient- or study-consistent fictional surrogates from stateless keyed HMACs.

Patient scope passes the server secret and a patient anchor. Study scope passes
a derived study key and ``patient=None``; it draws from wider fictional spaces
because one surrogate must stay distinct across a whole release.
"""

from __future__ import annotations

import re

from outis.replacement.ages import synthetic_age
from outis.replacement.names.aliases import full_identity_aliases
from outis.replacement.names.keys import name_identity, name_words
from outis.replacement.primitives import (
    canonical_form,
    keyed_digest,
    safe_label,
    study_digest,
)

_FIRST_NAMES = (
    "Avery",
    "Morgan",
    "Rowan",
    "Quinn",
    "Casey",
    "Riley",
    "Taylor",
    "Jordan",
    "Alex",
    "Drew",
    "Robin",
    "Cameron",
    "Sage",
    "Harper",
    "Skyler",
    "Emery",
    "Reese",
    "Parker",
    "Finley",
    "Dakota",
    "Blair",
    "Charlie",
    "Jamie",
    "Kendall",
    "Logan",
    "Hayden",
    "Ellis",
    "Remy",
    "Arden",
    "Lane",
    "Marley",
    "Sidney",
)
_LAST_NAMES = (
    "Alder",
    "Aspen",
    "Birch",
    "Cedar",
    "Cobalt",
    "Cypress",
    "Elm",
    "Fern",
    "Flint",
    "Grove",
    "Haven",
    "Juniper",
    "Laurel",
    "Linden",
    "Maple",
    "Meadow",
    "Moss",
    "North",
    "Oak",
    "Orchard",
    "Pine",
    "Quartz",
    "Reed",
    "River",
    "Stone",
    "Vale",
    "West",
    "Willow",
    "Wren",
    "Brook",
    "Cove",
    "Sable",
)

# Study scope only. Patient-scope outputs keep the vocabulary above unchanged.
_STUDY_FIRST_NAMES = _FIRST_NAMES + (
    "Addison",
    "Ainsley",
    "Ari",
    "Ashton",
    "Bailey",
    "Bellamy",
    "Blake",
    "Bowie",
    "Briar",
    "Carson",
    "Corey",
    "Dallas",
    "Devon",
    "Eden",
    "Elliot",
    "Emerson",
    "Frankie",
    "Gray",
    "Hollis",
    "Indigo",
    "Jessie",
    "Jules",
    "Kai",
    "Kerry",
    "Lennon",
    "Monroe",
    "Noel",
    "Peyton",
    "Rory",
    "Shea",
    "Tatum",
    "Wynn",
)
_STUDY_SURNAME_SUFFIXES = (
    "",
    *(
        "brook field ford wood ridge dale hill gate well more worth land shaw ley "
        "by ton wick stead mont crest bury combe hurst mere holt thorne bridge "
        "cliff haven row vale burn croft fell garth glen hall ham heath hollow "
        "howe lake leigh lock marsh mead mill moor port rock shire side stow "
        "thwaite view wold yard bank beck bourne cote down fold"
    ).split(),
)
_STUDY_LAST_NAMES = tuple(
    dict.fromkeys(
        stem + suffix
        for stem in _LAST_NAMES
        for suffix in _STUDY_SURNAME_SUFFIXES
        if suffix != stem.casefold()
    )
)
# Every NANP area code keeps 555-0100 through 555-0199 for fictional use. Study
# scope spreads that block over geographic-format codes, excluding N11 codes.
_STUDY_AREA_CODES = tuple(
    code for code in range(200, 1000) if code // 10 % 10 != 9 and code % 100 != 11
)
# RFC 5737 documentation networks.
_STUDY_IP_NETWORKS = ("192.0.2", "198.51.100", "203.0.113")

_EXACT_IDENTITY_LABELS = frozenset({"MRN", "ACCOUNT", "ID", "EMAIL", "URL"})
_FACILITY_SPECIALTIES = (
    (
        "Pediatric",
        re.compile(r"\b(?:children(?:['’]s|s)?|pediatric|paediatric)\b", re.I),
    ),
    ("Family", re.compile(r"\bfamily\b", re.I)),
    ("Rehabilitation", re.compile(r"\brehabilitation\b", re.I)),
    ("Dialysis", re.compile(r"\bdialysis\b", re.I)),
    ("Infusion", re.compile(r"\binfusion\b", re.I)),
    ("Oncology", re.compile(r"\boncology\b", re.I)),
    ("Cardiology", re.compile(r"\bcardiology\b", re.I)),
    ("Psychiatric", re.compile(r"\b(?:psychiatric|mental\s+health)\b", re.I)),
    ("Hospice", re.compile(r"\bhospice\b", re.I)),
    ("Wound Care", re.compile(r"\bwound\s+care\b", re.I)),
    ("Wound", re.compile(r"\bwound\b(?!\s+care\b)", re.I)),
)
_FACILITY_TYPES = (
    ("House", re.compile(r"\bhospice\s+house\b", re.I)),
    ("Care Residence", re.compile(r"\bcare\s+(?:residence|lodge)\b", re.I)),
    ("Nursing Home", re.compile(r"\bnursing\s+home\b", re.I)),
    ("Hospital", re.compile(r"\bhospital\b", re.I)),
    ("Infirmary", re.compile(r"\binfirmary\b", re.I)),
    (
        "Pavilion",
        re.compile(
            r"\b(?:medical|outpatient|diagnostic|infusion|wound(?:\s+care)?)\s+pavilion\b",
            re.I,
        ),
    ),
    ("Clinic", re.compile(r"\bclinic\b", re.I)),
    ("Unit", re.compile(r"\bunit\b", re.I)),
    ("Medical Center", re.compile(r"\bmedical\s+cent(?:er|re)\b", re.I)),
    ("Center", re.compile(r"\bcent(?:er|re)\b", re.I)),
    ("Practice", re.compile(r"\bpractice\b", re.I)),
)


def _digest(
    secret: bytes, patient: str | None, label: str, value: str, salt: int
) -> bytes:
    if patient is None:
        # Study scope: the key is already study-specific; no patient is mixed in.
        return study_digest(secret, label, value, salt)
    return keyed_digest(secret, patient, label, value, salt)


def _fictional_name(
    secret: bytes, patient: str | None, identity: str
) -> tuple[str, str]:
    original_words = name_words(identity)
    expressed_parts = full_identity_aliases(identity)
    family_words = name_words(expressed_parts[-1]) if len(expressed_parts) == 2 else []
    original_first = (
        canonical_form(expressed_parts[0])
        if len(expressed_parts) == 2
        else original_words[0]
        if original_words
        else None
    )
    original_last = (
        family_words[-1]
        if family_words
        else original_words[-1]
        if original_words
        else None
    )
    first_names, last_names = (
        (_STUDY_FIRST_NAMES, _STUDY_LAST_NAMES)
        if patient is None
        else (_FIRST_NAMES, _LAST_NAMES)
    )
    for salt in range(100):
        digest = _digest(secret, patient, "NAME", identity, salt)
        first = first_names[int.from_bytes(digest[:4], "big") % len(first_names)]
        last = last_names[int.from_bytes(digest[4:8], "big") % len(last_names)]
        # Partial follow-up mentions must also change, even if only one original
        # component happened to collide with a candidate fictional component.
        if (
            canonical_form(first + " " + last) != identity
            and first.casefold() != original_first
            and last.casefold() != original_last
        ):
            return first, last
    # This path is unreachable with the current vocabulary, but remains safe if
    # a downstream caller replaces the vocabulary with a single original name.
    return "Synthetic", "Person"


def _facility_setting(value: str) -> str:
    """Emit only fixed public setting terms, never source proper-name tokens."""
    specialties = [
        safe_term
        for safe_term, pattern in _FACILITY_SPECIALTIES
        if pattern.search(value)
    ]
    facility_type = next(
        (safe_term for safe_term, pattern in _FACILITY_TYPES if pattern.search(value)),
        None,
    )
    if facility_type is None:
        if specialties == ["Hospice"]:
            return "Hospice"
        facility_type = "Medical Center"
    return " ".join(specialties + [facility_type])


def synthetic_value(
    value: str,
    label: str,
    secret: bytes,
    patient: str | None,
    aliases: dict[str, tuple[str, str]],
    *,
    name_binding=None,
    name_warnings=None,
) -> str:
    canonical = canonical_form(value)
    # Opaque identifiers, email local parts and URL paths can distinguish case,
    # whitespace or Unicode spelling. Match the exact identity-linking keys;
    # applying name-style normalization here silently merges distinct values.
    identity = value if label in _EXACT_IDENTITY_LABELS else canonical
    if label == "NAME":
        title, identity = name_identity(value)
        full_identity, role = (
            (name_binding.owner, name_binding.role)
            if name_binding is not None
            else aliases.get(identity, (identity, "full"))
        )
        first, last = _fictional_name(secret, patient, full_identity)
        rendered = title + (
            first if role == "first" else last if role == "last" else first + " " + last
        )
        if name_binding is not None:
            original_tokens = name_words(identity)

            def safe_nickname(candidate):
                tokens = name_words(name_identity(candidate)[1])
                width = len(original_tokens)
                return bool(
                    tokens
                    and width
                    and not any(
                        tokens[i : i + width] == original_tokens
                        for i in range(len(tokens) - width + 1)
                    )
                )

            if not safe_nickname(rendered):
                # Keep the canonical person's key/mapping unchanged. Its family
                # name is a consistent alias only if no original nickname token
                # sequence survives; expanding to the full name would echo it.
                rendered = title + last
            if not safe_nickname(rendered):
                if name_warnings is not None:
                    name_warnings.append(
                        "Synthetic owner name contains the original nickname; NAME redacted."
                    )
                return "[NAME]"
        return rendered
    if label == "AGE":
        return synthetic_age(value)
    study = patient is None
    for salt in range(100):
        digest = _digest(secret, patient, label, identity, salt)
        token = digest.hex()[:12]
        if label == "PHONE":
            area = (
                _STUDY_AREA_CODES[
                    int.from_bytes(digest[4:8], "big") % len(_STUDY_AREA_CODES)
                ]
                if study
                else 202
            )
            replacement = (
                f"({area}) 555-01{int.from_bytes(digest[:4], 'big') % 100:02d}"
            )
        elif label == "EMAIL":
            replacement = f"person.{token}@example.invalid"
        elif label == "ADDRESS":
            city = f"Example City {token[:6].upper()}" if study else "Example City"
            replacement = f"{100 + int.from_bytes(digest[:4], 'big') % 900} Synthetic Way, {city}, ZZ 00000"
        elif label == "SSN":
            replacement = f"000-{int.from_bytes(digest[:4], 'big') % 100:02d}-{int.from_bytes(digest[4:8], 'big') % 10000:04d}"
        elif label in {"MRN", "ACCOUNT", "ID"}:
            replacement = f"SYN-{label}-{token.upper()}"
        elif label == "URL":
            replacement = f"https://synthetic-{token}.invalid"
        elif label == "IP":
            network = (
                _STUDY_IP_NETWORKS[
                    int.from_bytes(digest[4:8], "big") % len(_STUDY_IP_NETWORKS)
                ]
                if study
                else "192.0.2"
            )
            replacement = f"{network}.{1 + int.from_bytes(digest[:4], 'big') % 254}"
        elif label == "LOCATION":
            replacement = f"Example City {token[:6].upper()}"
        elif label == "FACILITY":
            replacement = f"Synthetic {_facility_setting(value)} {token[:6].upper()}"
        else:
            return f"[{safe_label(label)}]"
        if canonical_form(replacement) != canonical:
            return replacement
    return f"[{safe_label(label)}]"
