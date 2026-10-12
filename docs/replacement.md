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
Contacts use reserved fictional domains/phone ranges in both
[replacement styles](#replacement-styles). Ages above 89 are
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

## Replacement styles

Synthetic replacement has two styles. The default, `obvious`, marks every
surrogate as fake. The opt-in `realistic` style keeps the format of the
original and draws values from reserved ranges and bundled fictional word
lists. Both styles share the same keys, determinism and invariants: the same
secret, patient and value always give the same surrogate. Characters outside
spans are copied exactly. A surrogate never canonically equals its original,
and no secret material appears in output. Names, dates and ages are identical
in both styles. The style is independent of [scope](#patient-and-project-scope):
in project scope, realistic surrogates are keyed by the project key instead of
the patient, so one value keeps one realistic surrogate across a project.

| Label | `obvious` (default) | `realistic` |
| --- | --- | --- |
| MRN, ACCOUNT, ID | `SYN-MRN-393B59E5AB0A` | `MR-004521` → `US-631169`: same length, character class per position and punctuation |
| SSN | `000-57-9817` | `123-45-6789` → `902-15-1948`: same layout, never-issued area |
| PHONE | `(202) 555-0169` | `+1 617-555-0193 ext. 22` → `+1 927-555-0158 ext. 95`: same layout, fictional 555-01XX line |
| EMAIL | `person.f980…@example.invalid` | `mara.calder@example.org` → `mark.lakedale@example.org` |
| URL | `https://synthetic-68c2….invalid` | `https://portal.northwood.org/patients/4417?view=summary` → `https://portal.millbrook.example.org/account/5193?notes=records` |
| IP | `192.0.2.75` | IPv4 in one of three documentation /24s; IPv6 in `2001:db8::/32` |
| LOCATION | `Example City 1589A8` | `Brackenford, NM` → `New Bayland, NM` |
| ADDRESS | `850 Synthetic Way, Example City, ZZ 00000` | `12 Tessford Lane, Apt 4B, Brackenford, NM 87101` → `55 Pondhaven Road, Apt 8L, New Bayland, NM 87917` |
| FACILITY | `Synthetic Pediatric Clinic 34E95E` | `Lakeshore Pediatric Clinic` → `Pineview Springs Regional Pediatric Clinic` |

Select the style in the `replacement` object (HTTP API or
`Deidentifier.deidentify(replacement=...)`), with `style=` on
`outis.replacement.surrogates.replace`, or with `outis text --mode synthetic
--style realistic` (add `--replace-region` for `keep_region: false`). Setting
`--style` sends a replacement object, so the CLI output then has the configured
response fields, such as entity IDs and processed offsets:

```json
{
  "mode": "synthetic",
  "patient_id": "internal-patient-42",
  "replacement": {"type": "synthetic", "style": "realistic", "keep_region": true}
}
```

### Realistic surrogates

- **MRN, ACCOUNT and ID** keep their length, the class of each character
  (digit, uppercase or lowercase letter) and every punctuation character in
  place. The surrogate is a keyed permutation of all strings with that exact
  format, chosen so that no value maps to itself. Within one secret, patient,
  label and format, two distinct identifiers never share a surrogate.
  Non-ASCII letters and digits become ASCII of the same class, from keyed draws
  that can collide.
- **SSN** keeps its layout. Nine-digit values use only areas the SSA never
  issues: `000`, `666` and `900`–`999`. In the 900s, the groups the IRS uses
  for ITINs (50–65, 70–88, 90–92, 94–99) are avoided, so no output is a valid
  SSN or ITIN. Partial values, such as `XXX-XX-6789`, keep their mask and
  replace only the digits.
- **PHONE** keeps the original punctuation, spacing, `+1` country code,
  extension and generic words such as `ext` or `Tel`. The number becomes
  `NXX-555-01XX`. NANP reserves 555-0100 to 555-0199 for fictional use in
  every area code, so no output can reach a real line. The area code is a
  keyed choice among 602 well-formed geographic-style codes; the original area
  code is not kept. This gives 60,200 distinct ten-digit numbers, and only 100
  seven-digit ones. Numbers that are not NANP-shaped (other country codes,
  vanity letters, unknown words) become `(NXX) 555-01XX`, because Outis uses no
  reserved fictional range for other countries.
- **EMAIL** keeps the local part's shape: letter runs become a fictional first
  name, surname or initial, digit runs keep their length, and separators stay.
  The domain is always `example.com`, `example.org` or `example.net`, matching
  a `.org` or `.net` original. Generic subdomains such as `mail` or `www` are
  kept; others become keyed words.
- **URL** keeps the scheme, port, generic subdomains, path depth, punctuation,
  query layout and a whitelisted file extension. Words become keyed generic
  words and digit runs keep their length. The site label becomes a keyed
  fictional name, so the host is always a subdomain of `example.com`,
  `example.org` or `example.net`, such as `portal.millbrook.example.org`.
- **IP** uses only documentation ranges: `192.0.2.0/24`, `198.51.100.0/24`,
  `203.0.113.0/24` and `2001:db8::/32`. A CIDR suffix or port is kept, and an
  exploded IPv6 address stays exploded. There are only 762 IPv4 surrogates,
  so many distinct addresses share one.
- **LOCATION** replaces each locality with a keyed town name such as
  `Ashford`, `Cedar Falls`, `Ashford Landing` or `New Bayland`. A trailing `County`,
  `Township`, `Parish` or `Borough` is kept. A trailing state (postal code or
  full name), ZIP code and country are recognized. With `keep_region: true`,
  the default, the state and country are kept verbatim. This keeps
  region-level covariates usable, which HIPAA Safe Harbor allows. With
  `keep_region: false`, they become a different keyed state or country. A span
  that is only a state or country is always replaced, because Outis never
  echoes a detected span.
- **ADDRESS** keeps the layout: line breaks, commas, a leading or trailing
  directional (`N.`), unit and box designators (`Apt`, `Unit`, `Suite`, `#`,
  `PO Box`) and the ZIP+4 shape. The house number keeps its digit count
  without a leading zero. The street name and suffix are keyed, and an
  abbreviated suffix stays abbreviated. The locality uses the same keyed town
  as LOCATION, so `Brackenford` becomes the same town in both labels for one
  patient. State and country follow `keep_region`.
- **ZIP codes** in a LOCATION or ADDRESS start with a three-digit prefix
  allocated to the state that is emitted. With no state, they start with
  `001`–`004`, prefixes USPS has never allocated. This is the documented
  placeholder; it looks like a ZIP code but belongs to no place.
- **FACILITY** becomes a fictional town-style name with an optional generic
  modifier (`Regional`, `Community`, `Memorial`, `Valley`). It keeps the same
  whitelisted setting and specialty terms as the obvious style, for example
  `Pediatric Clinic`, `Dialysis Unit` or `Hospice`.
- Case follows the original: an all-capitals place, street or facility stays
  in capitals.

### Why identifiers use a keyed permutation, not FF1 or FF3

Stanford's TIDE 2.0 uses FF3 format-preserving encryption. FF3 needs a minimum
domain, so it pads short identifiers and truncates the result, and TIDE
measured 35–37% collisions on three- to five-digit IDs. NIST's second public
draft of SP 800-38G Rev. 1 drops FF3 and FF3-1. FF1 remains, but its draft
also requires a domain of at least one million values. A three-digit MRN would
still need padding and truncation, and many originals would map to one
surrogate.

Outis therefore builds the permutation directly over the exact format domain.
It uses a balanced 16-round Feistel network whose round function is the same
keyed HMAC-SHA256 digest that keys every other surrogate. Cycle walking keeps
it inside the domain. The emitted value is `P⁻¹(P(x) + 1)`, a single cycle
through the domain, so it is injective and never returns the original. This
needs no new dependency. Python FF1 packages exist, but none was a maintained
dependency we could verify, and FF1 would not fix short domains.

This construction is not a standardized cipher and has no formal security
proof for small domains. Its security rests on the HMAC round function. As
with any deterministic mapping, anyone who can submit values to the service
can build a dictionary of the identifier space. Someone who learns many
original–surrogate pairs for one patient and format learns which surrogates
the remaining originals cannot have. Keyed draws, as used for SSNs and
phones, reveal nothing beyond equality, but they collide.

### Collisions

A collision merges two distinct originals into one surrogate. That loses
linkage but does not expose either value. The table gives the share of
distinct invented values whose surrogate is shared with at least one other
value, all keyed in one project scope: the worst case, where every value must
stay distinct across a release. It is measured by
`uv run python scripts/surrogate_collisions.py --sizes 2000 20000 --style realistic`
(and `--style obvious`). Realistic spaces do not depend on scope, so one
patient with as many values would collide at the same rates. Real patients
have few distinct values per label, so collisions within a patient are rare.

| Label | Obvious, 2,000 | Obvious, 20,000 | Realistic, 2,000 | Realistic, 20,000 |
| --- | ---: | ---: | ---: | ---: |
| MRN, 3–12 digits; `AA-999999`; ACCOUNT; ID | 0.00% | 0.00% | 0.00% | 0.00% |
| SSN | 0.30% | 1.93% | 0.00% | 0.06% |
| ADDRESS | 0.00% | 0.00% | 0.00% | 0.06% |
| EMAIL | 0.00% | 0.00% | 0.40% | 3.39% |
| URL | 0.00% | 0.00% | 2.00% | 15.31% |
| FACILITY | 0.00% | 0.15% | 1.10% | 7.72% |
| PHONE | 2.90% | 24.44% | 2.05% | 27.87% |
| LOCATION | 0.00% | 0.10% | 4.70% | 26.28% |
| IP | 91.95% | 100.00% | 93.10% | 100.00% |

The three-digit MRN row covers the whole 1,000-value domain: every value
gets its own surrogate, where FF3 with padding and truncation merged more than
a third. Realistic identifiers never collide within a label and format.
Identifiers of different labels, such as an MRN and an ACCOUNT with the same
digits, are permuted independently and can share a surrogate. Realistic SSNs
collide less than obvious ones because the never-issued space is 58 times
larger. Phone numbers use 602 area codes against the 712 that project scope uses with the obvious style,
because realistic style also excludes toll-free, personal-communication and
other non-geographic codes. Email, URL, facility and place names come from
vocabularies far smaller than the 24- and 48-bit tokens of the obvious style;
host-only URLs are limited by about 4,000 site names. Where these merges
matter, mark those labels with `per_type`, or use keyed `entity_ids` to tell
identities apart.

### Trade-offs

Realistic output is harder to tell apart from real data, and this cuts both
ways:

- A missed identifier sits among realistic surrogates instead of next to
  obvious `SYN-` and `Example City` markers. Prior work cited by TIDE 2.0
  argues this "hiding in plain sight" makes leaked PHI harder to find. **This
  project has not tested that claim.** Do not rely on it as a privacy control.
- Realistic values are more likely to resemble real people, places and
  records by accident. A format-preserving MRN can be a real MRN of another
  patient at the same site. A fictional town or street, built from common
  English words, may be a real place. Names in email local parts may match
  real people. Kept states and countries are real by design. Downstream
  systems may wrongly join realistic surrogates with real records. Treat
  realistic output as de-identified data and never as a source of real
  identifiers.
- Reviewers can no longer see at a glance which spans were replaced. Use the
  returned entities and processed offsets to audit replacements.

The bundled town, street, facility, first-name and URL word lists were
written for this project and ship under its licence. State names, postal codes
and ZIP prefix ranges are public USPS facts, transcribed by hand. See
`src/outis/replacement/realistic_data.py`.
