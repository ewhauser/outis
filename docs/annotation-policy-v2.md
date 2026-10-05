# Clinical PHI annotation policy

**This is the project's only annotation policy.** Every labelling decision in
force is on this page or in the [places addendum](annotation-policy-v2-places.md),
which is part of it. If another document, prompt or older corpus disagrees with
this page, this page wins.

Policy identifier: `clinical-phi-v2`. There is no v1 document. The "v2" in the
identifier and filename distinguishes this written policy from the unwritten
conventions of the earliest corpora; see
[Corpora that predate this policy](#corpora-that-predate-this-policy). The
identifier is stamped into corpus rows and manifests, so it is not renamed.

This is the authoring contract for development and training corpora and for
new evaluation banks. It defines occurrence-level annotation and span
boundaries. It does not establish clinical accuracy, clinical truth, or legal
compliance.

## What this policy consists of

| Part | Role |
| --- | --- |
| This page | Span conventions, cohorts, provenance, metadata and audit rules |
| [Places addendum](annotation-policy-v2-places.md) (`clinical-phi-v2-places`) | Real localities, care-site abbreviations and public references to places |
| [`annotation_policy.py`](../src/clinical_deid/data/annotation_policy.py) and [`annotation_policy_places.py`](../src/clinical_deid/data/annotation_policy_places.py) | Executable contract with exact character-offset examples |

Not part of the annotation policy: the API's request policies in
`src/clinical_deid/policy.py`, whose own version is `"1"`. They control what a
caller redacts and have nothing to do with how gold spans are annotated.

The executable contract and exact character-offset examples are
[`annotation_policy.py`](../src/clinical_deid/data/annotation_policy.py). The original
corpora, annotations, generators, checkpoints, manifests and reports remain
historical evidence. Legacy generators are superseded for new remediation
training, but retained to reproduce their original artifacts. Corrected labels
must be published in new files with source hashes and an annotation-change
record; historical scores must not be relabeled as v2 results.

## Occurrences and span boundaries

Annotate an occurrence in its sentence and document context. A spelling can be
PHI in one occurrence and ordinary language in another. Do not derive gold
annotations from a word blacklist, model prediction, title case alone, or a
dictionary of common names.

| Label or case | Annotate | Leave outside the span |
| --- | --- | --- |
| Ordinary pronouns | No NAME in `He answered`, `I, the patient, agreed`, or `She returned` | An ordinary pronoun remains non-PHI even when a full patient name occurs nearby |
| Explicit naming | `He` in `Patient name: He`; `Care` in `the patient named Care` | `He` used subsequently as a pronoun; `Care` in `Care history` |
| Plausible personal names | `Zoë O’Rían-Soto`, `Ana María de la Cruz`, the personal `Wilson` in a named person | Honorifics such as `Dr.`, credentials such as `MD`, a disease eponym such as `Wilson disease` |
| Aliases | A preferred or otherwise explicitly established personal alias, including an occurrence before its full identity when the document establishes the relationship | Ambiguous ordinary words without personal naming evidence |
| DATE | Complete patient-specific dates, partial dates and patient-specific years; `October 2, 2024`, `March 2021`, `1930` in `born in 1930`. A written date range is one span including its connector: `03/02/2024 to 03/09/2024`, `March 2-9, 2024`, `March 28 - April 4, 2024` | Public study years, publication dates, durations, doses and clinical stages; a leading `from` or `between` before a range; dates written as a separate list (`on 03/02/2024 and 03/09/2024`) are separate spans |
| AGE | Only the numeric or spelled-out age value when greater than 89 | Prefixes, units, suffixes, sex markers and ages at or below 89 |
| LOCATION | The maximal authored locality expression, e.g. `Cedar Haven, UT` or `North Cedar Haven` | Prepositions and adjacent facility/address fields |
| FACILITY | The authored facility name, e.g. `Silver Finch Medical Center`, including an `at <name>` continuation such as `Lakeshore Family Medicine at Ellery Falls` | An adjacent locality introduced by `in` or a comma; an `at` locality with a state or postal qualifier (`Brackenwheel Health Center` at `Brackenford, NM`); an adjacent address; the parent facility of a named unit (`Larkspur Ward` at `Mossmere Health Hub` is two FACILITY spans) |
| ADDRESS | The complete authored address as one span | Nested LOCATION annotations, field labels, trailing sentence punctuation |

AGE boundaries are deliberately consistent across surface forms: annotate
`92` in `92F`, `92yo`, `92 y/o`, `92-year-old`, and `92 years old`; annotate
`ninety-two` in `ninety-two years old`. The suffix is context for recognizing
age, not part of its gold span. A hyphen internal to a spelled-out number stays
inside the span. This convention may differ from historical annotations;
comparisons must disclose that policy difference.

Offsets use Unicode characters, not bytes or tokens; ends are exclusive. Preserve
internal apostrophes, accents, particles, spaces and hyphens in proper names.
Preserve internal commas in complete dates/localities/addresses. Exclude leading
or trailing whitespace and sentence punctuation unless punctuation belongs to
the authored value. Nested or overlapping gold spans are invalid. The remaining
labels retain the protocol's vocabulary: PHONE, EMAIL, MRN, SSN, ACCOUNT, ID,
URL and IP. Preserve typed distinctions in the primary metrics; merged-type
diagnostics do not replace exact typed-span evaluation.

## Conventions settled by the convention census

The table above leaves some cases open. In early October 2026, ten independent readers
were given only the policy, protocol, types and fixtures and asked to choose a
convention for 35 such cases. They agreed 10 of 10 on every case in the table
below. The one split case, employer names, is decided after the table. The rows for a leading `The` and for service organisations were decided by the owner on 2026-10-05 for the care-site and locality bank; their fixtures are `CARE_SITE_FIXTURES` in `src/clinical_deid/data/annotation_policy_care_sites.py`. Training banks were normalised to these choices, and they
are in force for all new authoring and annotation. The record is
[`convention_census_001.json`](../reports/takeover/convention_census_001.json).
Not every row has an executable fixture yet.

| Label | Case | Convention |
| --- | --- | --- |
| ID | Insurance member, policy, group and claim numbers | ID |
| ID | Encounter, visit and CSN numbers | ID |
| ID | Accession and specimen numbers | ID |
| ID | Device serial numbers | ID |
| ID | Driver's licence, passport and similar licence numbers | ID |
| ID | Clinician NPI and licence numbers | ID |
| ID | Prescription numbers (`Rx #8830167`) | ID on the number; the `Rx #` caption stays outside |
| ACCOUNT | Billing, patient and guarantor account numbers | ACCOUNT |
| MRN | A record number written with a prefix, such as `QX-0041122` | One MRN span including the prefix |
| MRN | A field caption before the value, such as `MRN:` or `MRN#` | Outside the span |
| NAME | Generational suffix, such as `Jr.` | Inside the span |
| NAME | Possessive `'s` | Outside the span |
| NAME | Initials | Inside the span |
| NAME | Clinician names | NAME, without the honorific or credentials |
| NAME | Nickname written in quote marks | NAME, without the quote marks |
| DATE | A weekday written before a date | Inside the span |
| DATE | A clock time next to a date | Outside the span |
| DATE | Relative expressions, such as `three days ago` | Not annotated |
| DATE | A patient-specific year alone, including years in history and screening lists (`bone density scan 2011`, `shingles vaccine 2019`, `cholecystectomy (1988)`) | DATE. Stricter than Safe Harbor, which permits a year alone; kept so years can be shifted consistently with full dates. Owner decision, 2026-10-05 |
| DATE | A written date range | One DATE span, as in the table above |
| DATE | A diagnosis-onset, year-range or future patient-specific year (`diagnosed in 1997`, `smoked 1984-2003`, `next due 2028`) | DATE; a year range is one span |
| AGE | A decade over 89, such as `nineties` | AGE |
| PHONE | Extension | Inside the span |
| PHONE | Leading `+1` country code | Inside the span |
| PHONE | Fax numbers | PHONE |
| PHONE | Pager numbers | PHONE |
| PHONE | An extension written with no phone number (`ward phone ext. 4412`) | Not annotated |
| ADDRESS | Street, unit, city, state and postal code written together | One ADDRESS span |
| URL | Scheme, such as `https://` | Inside the span |
| LOCATION | City with its state | One LOCATION span including the state |
| LOCATION | A ZIP code alone | LOCATION |
| LOCATION | A ZIP+4 code (`53728-4416`) | One LOCATION span including the +4 |
| LOCATION | A county name | One LOCATION span including `County` (`Kestrine County`, `Tollund County, OR`); followed by a generic site word, the word stays outside (`the [Kestrine County] office`, places P7); written without `County`, an ordinary locality (`grew up in [Kestrine]`) |
| LOCATION | A state or a country alone | Not annotated |
| FACILITY | A leading `the` used as an article (`seen at the Mayo Clinic`, sentence-initial `The Larkspur Pharmacy called`) | Outside the span |
| FACILITY | A capitalised `The` that is part of the proper name, including a service organisation's name (`The Marlowe at Fennick Ridge`, `The Hearth at Ombrey`) | Inside the span |
| FACILITY, LOCATION | All-caps or lowercase text, where capitals give no signal (`THE HEARTH AT OMBREY`, `OMBREY CLINIC`, `harlow vane drug`) | The authored role decides, exactly as in normal case: a named site or pharmacy is FACILITY, `The` is inside only when part of the name, and a city modifying a generic site is LOCATION (places P7) |
| Any | A place-word used as an ordinary word in all-caps or lowercase text (`READING GLASSES`) | Not annotated; case does not change the role |
| FACILITY | A department word next to an institution name | Outside the span; FACILITY covers the institution name |
| FACILITY | A named ward | FACILITY |
| FACILITY | A named pharmacy | FACILITY |
| FACILITY | A named organisation that provides care, services, supplies or housing to the patient: home health agency or VNA, hospice, durable medical equipment supplier, laboratory, adult day program, senior-living community, mail-order or specialty pharmacy, funeral home handling the patient's arrangements | FACILITY, including its type words (`Vossmere VNA`, `Saint Odile Adult Day Program`) |
| FACILITY | The same type words without a name (`home health referral sent`, `the hospice team`) | Not annotated |

**Employer names.** Do not annotate an employer or other organisation that is
not acting as a care site or serving the patient's care. Decide per occurrence: in `Works as a nurse at Ombrey General Hospital; admitted to Ombrey General Hospital`, the first mention is the employer role and is not annotated, the second is FACILITY. Six census readers chose this and four chose
FACILITY; the owner decided it on 2026-10-04. An employer that is itself the
patient's care site is FACILITY under the ordinary rule.

**Known risk.** Independent authors do not always follow the ID and ACCOUNT
split. One proxy-bank author labelled insurance member and claim numbers
ACCOUNT. A bank that does so will show ID and ACCOUNT type errors that are not
detector misses. Under [GOALS.md](../GOALS.md) such a span is still masked.

## Realistic and stress cohorts

Use `cohort=realistic` for plausible clinical prose and fictional identities.
Explicit unusual names such as a person named `He` or `Care` remain legitimate
policy examples, but belong to `cohort=unusual_name` and must be reported
separately. Their validity does not imply that their frequency is representative.
Use `cohort=long_document` for multi-window diagnostic notes, with actual model
token lengths and known ownership-boundary placement. Explicitly disclose
authored repeated-progress/seam stress notes as such; the cohort label does not
imply naturally sampled or clinically validated prose. Publish each cohort's
support and results rather than pooling away a failing slice.

Ordinary ages, pronouns, publication references and durations should occur in
both PHI-bearing and PHI-free clinical prose. Avoid annotation explanations such
as “Age below 90 remains clinical information,” “no identified individual,” or
“this is not PHI.” Labels belong in metadata, not in the note. Context cues such
as a genuine age field are allowed; differences in their frequency across
positive and negative notes are audit diagnostics requiring review.

## Synthetic provenance

Use independently invented people, scenarios, facilities and addresses; do not
copy real clinical records. Real public place and care-site names are permitted
as surface values only under the
[places addendum](annotation-policy-v2-places.md), which defines their sources,
held-out value pools and matched negatives. Retain reserved network identifiers while varying
formats: `.test`, `.example`, and example.com/example.net/example.org domains,
TEST-NET IPv4 addresses, and fictional US 555-0100 through 555-0199 phone lines.
Other supported identifier families require an explicitly documented reserved
or invented scheme. Broadening domain spellings is not permission to use
arbitrary live domains. MRNs, accounts, SSNs and IDs must be invented under the
generator's recorded synthetic scheme. Provenance checks are distinct from
annotation validity.

## New-row metadata and split construction

All new rows retain `id`, `text`, `spans` and `split` (`train` or `validation`)
and add these top-level fields:

| Field | Contract |
| --- | --- |
| `annotation_policy` | Exactly `clinical-phi-v2` |
| `identity_group` | SHA-256 of NFC-normalized, case-folded canonical full identity; a descriptive nonempty key is allowed for PHI-free notes with no identity |
| `identity_groups` | Optional list of all full-person identity hashes, including `identity_group`, clinicians and relatives; unique values only; an empty list is allowed only on PHI-free rows |
| `pair_id` | A stable nonempty string shared by paired contrasts; a unique group key for an unpaired note |
| `template_family` | A stable nonempty clinical-context family key; separate families across training and development |
| `phenomena` | Nonempty list of distinct strings describing the intended diagnostic phenomena |
| `cohort` | `realistic`, `unusual_name`, or `long_document` |

Canonical identities and their aliases stay in one group. Assign a digest to
training when `int(identity_group[:8], 16) % 10 < 8`, and validation otherwise.
Every full-person identity listed on a row must belong to that same partition.
Do not generate training names from one alphabet range and validation names
from another. Grouping and shared spelling distributions reduce known shortcuts;
they do not establish independence by themselves. A paired positive/negative
note inherits the same identity and pair keys even if the negative member has
no NAME annotation. Report lengths, scripts, punctuation, case, tokenization,
families and layouts to assess comparable distributions. Development authors
must independently author their contexts rather than import training templates.

## Corpora that predate this policy

A corpus follows this policy only if its rows and manifest declare
`annotation_policy` as `clinical-phi-v2`. Corpora without that declaration were
written before the policy existed and follow older, unwritten conventions. That
includes the original training, validation and challenge corpora, the early
coverage, context, narrative and contrastive corpora, and the qualification
banks v4, v5 and v6. Banks from v7 onward were authored with this policy as
their contract.

Rules for those corpora:

- They remain valid historical evidence and are kept to reproduce recorded
  results. Do not relabel them in place.
- Do not treat their annotations as examples of the current conventions. Where
  they disagree with this page, this page is correct for new work.
- A known difference is AGE boundaries, described above. Others include the
  census cases, which older corpora decided inconsistently.
- Banks whose filename carries a `__census1` suffix were rewritten to the
  census conventions by `scripts/takeover/normalize_conventions.py`. Use those
  versions, not the originals, when training.

The short convention summary that used to sit in the
[evaluation protocol](../reports/evaluation_protocol.md) was removed on
2026-10-04 in favour of this page.

## Audit and qualification boundary

Run the read-only auditor on explicit development paths, for example:

```sh
.venv/bin/python scripts/audit_corpus_quality.py \
  --corpus train=data/remediation_train.jsonl \
  --corpus validation=data/remediation_development.jsonl \
  --output reports/remediation_quality.json
```

The auditor rejects protected path names and symlink paths before opening input
bytes. Strict mode requires an explicitly adjacent `.jsonl.manifest.json` or
`.manifest.json` declaring the matching split, v2 policy and corpus SHA-256.
It preflights every input's manifest for protected final/test/qualification
declarations before reading any corpus payload, then verifies the corpus hash.
It never discovers files or reads generators or attempt ledgers. Rows declaring
a final/test/sealed role are refused. It checks
annotation offsets, labels, overlap, duplicate IDs and exact texts, policy
metadata, explanatory label cues, and cross-split identity/pair/template-family
contamination. Shared identities within a split are expected in paired examples.
Metadata disjointness is only as good as the author's grouping: authors must
enumerate secondary personal identities in `identity_groups`.

The JSON report binds each input's absolute path, role, SHA-256 and manifest
hashes. Masked document/sentence overlap and ordinary context-cue rates are
diagnostic. Cue gaps of at least 25 percentage points are flagged for review
when both document strata contain at least 20 notes; these are review heuristics,
not universal pass/fail thresholds. `--legacy` allows absent manifests and
performs structural checks without claiming
v2 policy compliance. Structural defects still fail in that mode. No audit
rewrites annotations or proves semantic correctness.

Token lengths use the actual `o200k_base` model tokenizer by default. The
explicit `--tokenizer none` option reports token lengths unavailable instead of
substituting character counts. Per-label, per-cohort and per-phenomenon support
must accompany results. The existing overall and per-label accuracy gates are
unchanged. If a new slice is required to demonstrate a 99% Wilson lower bound,
even 200/200 recalled spans is insufficient; 381/381 is the minimum with a
two-sided 95% Wilson interval. Correlated templates also limit interpretation.

The sealed v6 qualification bank remains unopened and excluded from this work.
Compatibility between that bank and this clarified policy has **not** been
certified. Resolve compatibility by independent policy review before deciding
whether it can qualify a v2 candidate; a fresh independently authored bank may
be needed. Do not inspect sealed predictions or examples and then repair the
bank around a candidate. All remediation corpora and results are development
evidence only.
