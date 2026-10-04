# Clinical PHI annotation policy v2

Policy identifier: `clinical-phi-v2`. This is the authoring contract for the new
development and training corpora. It clarifies occurrence-level annotation and
span boundaries from the [evaluation protocol](../reports/evaluation_protocol.md).
It does not establish clinical accuracy, clinical truth, or legal compliance.

The executable contract and exact character-offset examples are
[`annotation_policy.py`](../src/clinical_deid/annotation_policy.py). The original
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
