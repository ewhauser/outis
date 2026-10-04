# Consistent synthetic replacements

In `synthetic` mode, detected PHI is replaced with fictional values that stay
consistent for a patient across notes. Redaction mode needs none of this setup.

## Secret and patient anchor

Set `DEID_SECRET` to a secret with at least 32 bytes and keep it stable. Synthetic
requests also require a stable `patient_id`; it anchors date shifts across notes.
This anchor is exact and case-sensitive, including whitespace. Reuse the same
opaque identifier for the patient in every request.
Rotate the secret when a new mapping universe is desired. Independent secrets
produce independent mappings. Example startup:

```sh
export DEID_SECRET="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(48))')"
.venv/bin/clinical-deid serve
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
