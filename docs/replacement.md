# Consistent synthetic replacements

In `synthetic` mode, detected PHI is replaced with fictional values that stay
consistent for a patient across notes. Redaction mode needs none of this setup.

## Secret and patient anchor

Set `OUTIS_SECRET` to a secret with at least 32 bytes and keep it stable. Synthetic
requests also require a stable `patient_id`; it anchors date shifts across notes.
This anchor is exact and case-sensitive, including whitespace. Reuse the same
opaque identifier for the patient in every request.
Rotate the secret when a new mapping universe is desired. Independent secrets
produce independent mappings. Example startup:

```sh
export OUTIS_SECRET="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(48))')"
uv run outis serve
```

```json
{
  "text": "Mara Calder was admitted on 2024-02-28 and discharged on 2024-03-02. Creatinine 1.2 mg/dL.",
  "mode": "synthetic",
  "patient_id": "internal-patient-42"
}
```

All detected, supported exact dates for a patient share a nonzero shift. Calendar arithmetic
preserves event order, intervals, and leap days. A caller can explicitly provide
`date_shift_days` for a study-wide patient mapping, but must keep it consistent
for that patient. Incomplete or unsupported dates are redacted with warnings
instead of inventing precision. Numeric dates use month/day/year when ambiguous
and emit a warning; an initial value above 12 establishes day/month/year.
If shifting a day/month/year date would make its meaning ambiguous, the output
uses year/month/day with the original separator and a warning.
Year-first dates, dotted numeric dates, hyphenated English month names, and ISO date-times are also
supported. A date range detected as one span (`03/02/2024 to 03/09/2024`,
`March 2-9, 2024`, `March 28 - April 4, 2024`) has both endpoints shifted by the
same offset, keeping its connector; a compressed range stays compressed only
while both shifted endpoints share a month and year. A range whose endpoints
cannot both be read, or one written with a single year that crosses a year
boundary, is redacted. Numeric dates with a two-digit year (`3/20/26`) are
shifted and keep two digits; the century is not needed except for year `00`
around February 29, which is redacted, as is a day/month date whose shifted
form would be ambiguous. Date-times retain the original time of day and timezone text.
Date shifting changes calendar/seasonal context even when intervals are preserved.

## Names, identifiers and other types

Names and identifiers are replaced deterministically. Unambiguous first/last
name aliases within a note follow a detected full name. Supported explicit
nickname declarations also link detected nickname occurrences to their detected
owner, so both use one synthetic identity. Ambiguous ownership remains separate;
an unsafe nickname replacement falls back to redaction with a warning. Standalone aliases
across different notes cannot be linked reliably without a full identity.
MRN, ACCOUNT, ID, EMAIL and URL replacements use the exact detected value,
including case, whitespace and Unicode spelling, to avoid merging distinct
identifiers. Earlier versions normalized these values; reprocess related notes
when upgrading if they must share one replacement mapping.
Synthetic facilities retain only whitelisted generic setting and specialty
terms, such as hospice, rehabilitation hospital, pediatric clinic, dialysis
unit, or infusion center. Identifying names and unknown terms are replaced.
Contacts use reserved fictional domains/phone ranges. Ages above 89 are
bucketed to `90+`; other ages, diagnoses, medications, doses, labs, vitals,
relative durations, and clinical narrative remain outside replacements unless
detected as PHI. The replacement engine preserves every character outside
detected spans. Fictional identities may coincidentally resemble real people;
they do not establish anonymity or resistance to reidentification.

## Patient and project scope

By default (`replacement.scope: "patient"`) every surrogate is keyed by the
server secret, the patient anchor, the label and the value. One provider,
facility, phone number or MRN therefore gets a different fictional value for
each patient, and a released corpus cannot follow it across patients. This is
the default and its outputs are unchanged from earlier releases.

Project scope is an opt-in alternative for building one release, such as a
research data set. Set `replacement.scope: "project"` and the request's
`project_id`:

```json
{
  "text": "Seen by Dr. Mara Calder at Exampleford Rehabilitation Hospital on 2024-02-28.",
  "mode": "synthetic",
  "patient_id": "internal-patient-42",
  "project_id": "sepsis-2026-release-1",
  "replacement": {
    "type": "synthetic",
    "scope": "project"
  }
}
```

In project scope:

- NAME, FACILITY, LOCATION, ADDRESS, PHONE, EMAIL, URL, IP, MRN, SSN, ACCOUNT
  and ID surrogates are keyed by a project key, the label and the value, with
  no patient. The same value gets the same surrogate in every patient's notes
  under one `project_id`. Keyed `entity_ids` follow the same scope.
- Dates still shift by one nonzero offset per patient, so `patient_id` stays
  required. The offset is derived from the project key and the patient, so a
  patient's dates also shift differently in different projects. An explicit
  `date_shift_days` still overrides it.
- Keyed `entity_ids` for DATE spans stay per patient (project key plus patient).
  A DATE ID shared across patients would group one original date's
  differently shifted copies, and the bounded shift range would then reveal
  the original date and each patient's offset.
- The project key is derived from `OUTIS_SECRET` and the project ID with
  HKDF-SHA256 under the versioned tag `outis-project-key-v1`. Different project
  IDs give unrelated keys, so their surrogates and keyed IDs cannot be linked
  without the server secret. Project surrogates are also unrelated to
  patient-scope surrogates for the same secret. Neither the secret nor any
  derived key appears in responses or logs; responses do not echo the project
  ID.
- `project_id` must be set explicitly. Requests that omit it fall back to the
  default `main`, so project scope rejects them instead of putting every such
  caller into one shared key. An explicit `"project_id": "main"` is accepted.
- `project_id` is the same 1 to 60 characters from `A-Z a-z 0-9 _ : -` that
  usage accounting already accepts. It names a key but is not itself secret.
  Usage counters hash it, and renaming a project re-keys its surrogates. Use a
  separate project ID for each release that must not be linkable to another.
- Within-note alias and nickname linking work as in patient scope. A
  standalone partial name in another note (`Calder` alone) is still not linked
  to a full name, so it draws its own surrogate.
- `shared_entities` and `link_batch` still require one patient per batch.
  Shared context resolves partial names and nicknames, and that evidence should
  not cross patients. Cross-patient consistency comes from the project key
  instead, so send each patient's notes as a separate batch with the same
  `project_id`.

### Threat model

Project scope trades unlinkability for analytic utility. Before choosing it:

- Anyone holding the release can link a surrogate across patients and count
  how often it occurs. Frequency analysis then becomes possible: the most
  common surrogate surname or facility in a regional corpus likely maps to a
  common real surname or the largest local hospital.
- Surrogates are keyed hashes of the source value. Anyone who learns the
  server secret, or a derived project key, can recompute the surrogate for any
  candidate value (for example a list of local physicians) and confirm which
  ones appear in that project's release.
- Rotating `OUTIS_SECRET` or choosing a new `project_id` protects future
  releases only. Releases already issued keep their surrogates, and anyone
  holding an old key can still test candidates against them.
- Use one `project_id` per release or recipient that should not be linkable to
  another, and keep `OUTIS_SECRET` outside the release environment.

### Surrogate collisions

Patient-scope surrogates only need to be distinct within one patient, so they
draw from small fictional spaces. In project scope two distinct real values that
draw the same surrogate merge across the whole release, so project scope uses
wider reserved spaces:

| Label | Patient scope | Project scope |
| --- | --- | --- |
| NAME | 32 × 32 = 1,024 full names | 64 given names × 2,045 compound surnames = 130,880 |
| PHONE | `(202) 555-0100` to `555-0199`: 100 numbers | `555-01xx` under 712 geographic-format area codes (N11 excluded): 71,200 |
| IP | `192.0.2.1` to `.254`: 254 | all three RFC 5737 documentation networks: 762 |
| ADDRESS | 900 house numbers | 900 house numbers × a 6-hex-digit city token |
| SSN | `000-xx-xxxx`: 1,000,000 | same |
| FACILITY, LOCATION | 6 hex digits (about 16.8 million) per setting | same |
| EMAIL, URL, MRN, ACCOUNT, ID | 12 hex digits (48 bits) | same |

The table below gives the share of distinct invented values whose project-scope
surrogate is shared with at least one other value, measured by
`uv run python scripts/surrogate_collisions.py --sizes 200 2000 20000`:

| Label | 200 values | 2,000 values | 20,000 values |
| --- | ---: | ---: | ---: |
| NAME | 0.00% | 1.35% | 14.29% |
| PHONE | 0.00% | 2.90% | 24.44% |
| IP | 23.00% | 91.95% | 100.00% |
| SSN | 0.00% | 0.30% | 1.93% |
| FACILITY | 0.00% | 0.00% | 0.15% |
| LOCATION | 0.00% | 0.00% | 0.10% |
| ADDRESS, EMAIL, URL, MRN (6, 8 or 10 digits), ACCOUNT, ID | 0.00% | 0.00% | 0.00% |

With the patient-scope spaces, 2,000 distinct values would collide for 85% of
names and every phone number (`--scope patient`). Phone numbers and IP
addresses are limited by the size of the reserved fictional ranges; no larger
reserved range exists in the same format. A partial mention rendered as a
given or family name alone collides more often than the full names above.
Where these merges matter, mark PHONE and IP with `per_type` instead of
synthesizing them, and use keyed `entity_ids` (128-bit, collision-free in
practice) when an analysis must tell identities apart.
