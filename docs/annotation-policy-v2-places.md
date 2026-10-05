# Annotation policy v2 addendum: real places and institutions

**Status: approved on 2026-10-04**, including the provenance amendment,
decision N4 and the blind bank thresholds. Addendum identifier:
`clinical-phi-v2-places`. Existing corpora, annotations and results are
unchanged; the addendum applies to new generators and banks. It amends the
"Synthetic provenance" section of the [v2 policy](annotation-policy-v2.md), as
described below.

This addendum clarifies `clinical-phi-v2` for real place names, care-site
abbreviations and public references to places. It keeps the policy identifier
`clinical-phi-v2`: it decides cases the v2 table leaves open rather than
changing any decided case. It does not establish legal compliance.

## Why

External verification on ASQ-PHI showed that the detector misses real,
well-known places in terse text: standalone cities ("Chicago", "Seattle"),
institution abbreviations ("UCSF", "UCLA") and short institution names
("Mt. Sinai", "Houston Memorial"). Training data contains only invented places,
and the v2 table does not say how to treat the same names when they are public
references ("Chicago Classification", "per Mayo Clinic guidelines"). Recall
work on places needs these decisions first, or it will trade misses for
eponym false positives. See [external verification](external-verification.md).

## Decisions

| Case | Decision | Example (gold in brackets) |
|---|---|---|
| P1. Locality tied to the patient, relatives or household | LOCATION, with or without an explicit residence cue | `pt from [Chicago]`; `daughter lives in [Tacoma, WA]`; `returned from [Cedar Rapids] last week` |
| P2. Care site named by abbreviation | FACILITY | `seen at [UCSF] in May`; `transferred from [MGH]` |
| P3. Care site named without a care-type word | FACILITY | `followed at [Mt. Sinai]`; `admitted to [Cedars-Sinai]`; `ED visit at [Houston Memorial]` |
| P4. Full care-site name | FACILITY, including a leading `St.`, `Ste.`, `Mt.` or `Ft.` | `[St. Francis Hospital]`; `[Mt. Sinai Hospital]` |
| P5. Locality after a care site | Separate FACILITY and LOCATION spans, preposition outside (unchanged v2 rule) | `[Mayo Clinic] in [Rochester, MN]`; `[Brigham and Women's Hospital], [Boston]` |
| P5a. Care site followed by `at` | The v2 FACILITY rule: `at <name>` continues the FACILITY; an `at` locality with a state or postal qualifier is a separate LOCATION; a named unit `at` its parent is two FACILITY spans | `[Lakeshore Family Medicine at Ellery Falls]`; `[Brackenwheel Health Center] at [Brackenford, NM]` |
| P6. Clinician affiliation in the patient's care | FACILITY | `Dr. Patel from [UCSF] reviewed the scan` |
| P7. Locality that modifies an unnamed care site | LOCATION on the locality only; the generic word stays outside. A named site is still FACILITY (P3, P4) | `seen at our [Vellmont] clinic`; `the [Corriston] office`; but `[Vellmont Clinic]` |
| N1. Place inside a named instrument, criterion, classification, score, rule or trial | Not PHI | `Chicago Classification`; `Boston criteria`; `Seattle Heart Failure Model`; `Ottawa ankle rules`; `San Francisco Syncope Rule`; `Framingham Risk Score` |
| N2. Place inside a disease, pathogen or eponymous condition | Not PHI | `Lyme disease`; `West Nile virus`; `Rocky Mountain spotted fever`; `Norwalk virus` |
| N3. Institution as the author or source of public guidance or research | Not PHI | `per Mayo Clinic guidelines`; `a Johns Hopkins study found`; `Cleveland Clinic recommends` |
| N4. State, country or larger region alone | Not PHI (Safe Harbor permits states) | `common in Texas`; `travel to Mexico` |
| N5. A place-name word used as a person's name or an ordinary word | Not LOCATION; a person's name is NAME under v2 | `[Florence] Testier` is NAME; `reading glasses`; `mobile phone` |

Rules for mixed and ambiguous text:

- **Care beats citation.** If a sentence ties an institution to the patient's
  care and also cites it, annotate the care mention only:
  `seen at [Mayo Clinic]; per Mayo Clinic guidelines, start ...`.
- **Occurrence, not spelling.** The same spelling can be PHI in one occurrence
  and a public reference in another in the same note (v2 occurrence rule).
- **No dictionary-derived gold.** A name's presence in a gazetteer never makes
  an occurrence PHI. Gold comes from the authored role of the occurrence.
- **N4 is unchanged inside a locality.** A state that is part of a locality
  expression stays inside the LOCATION span (`[Tacoma, WA]`), as in v2.
- **Out of scope.** Employer and other organisation names not acting as a care
  site are unchanged by this addendum.
- **P7 was added on 2026-10-05** by the owner for the care-site and locality
  bank. Its fixtures are in `annotation_policy_care_sites.py`, not in
  `PLACE_FIXTURES`, so earlier generator source hashes stay valid. A
  content-free check found 191 training spans already following it and none
  conflicting.
- **P5a was added on 2026-10-04** to point at the v2 `at` rule adopted in commit
  `d9bbd85`. A content-free check found no conflicting gold in the places
  curriculum (no FACILITY `at` place pairs) or in `blind_places_v1`
  (`atrule_check.json`).

Decision N4 was confirmed with the addendum. The current detector is not yet
consistent with it: a spot check on 2026-10-04 redacted `Ohio` in
`lives in Ohio` but not `Texas` in `moved to Texas`.

## Provenance amendment

The v2 policy requires invented facilities and addresses. The following is
also permitted:

- **Real public place names** (cities, towns, counties, neighbourhoods) from
  the US Census Bureau Gazetteer Files.
- **Real public care-site names** from the CMS Hospital General Information
  file, and a curated list of common institution abbreviations, each entry
  recorded with a public source.

Both government sources are US federal works in the public domain. Neither
contains patient information. Conditions:

1. Real place values are surface strings only. People, dates, ages and
   identifiers stay invented under the existing reserved schemes. No row may
   reproduce or resemble a real person's record.
2. Each source file is pinned by URL and SHA-256 in the generator manifest.
3. **Value pools.** Every real place or institution value is assigned to a pool
   by `int(sha256(nfc(casefold(value)))[:8], 16) % 10`: values below 8 are the
   `train` pool and the rest are the `heldout` pool. Abbreviations and their
   expansions share the expansion's pool. Training and development generators
   use only the `train` pool. Blind banks report `heldout` and `train` pools
   separately, so recall on unseen real names is measured apart from recall on
   memorised ones.
4. Every positive phenomenon P1–P6 needs a matched negative from N1–N5 in the
   same layout, using a name from the same pool, so the model learns the role
   and not the name.
5. The detector itself still contains no place or institution dictionary.
   Real values are training and evaluation data only.
6. The project README's provenance statement is updated to say real public
   place and institution names may appear as surface values.

## Executable contract

The decisions are exact-offset examples in `PLACE_FIXTURES`, and the value
pool rule is `place_value_pool`, both in
[`annotation_policy_places.py`](../src/clinical_deid/data/annotation_policy_places.py).
They live in a separate module so that source hashes recorded by earlier
generators for `annotation_policy.py` stay reproducible. Generators and banks
using this addendum record `clinical-phi-v2` as their policy and
`clinical-phi-v2-places` in their manifest.
