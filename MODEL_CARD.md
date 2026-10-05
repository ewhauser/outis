---
license: apache-2.0
language:
  - en
base_model: answerdotai/ModernBERT-base
pipeline_tag: token-classification
tags:
  - de-identification
  - phi
  - clinical
  - ner
  - privacy
  - synthetic-data
---

# Clinical Deidentifier: ModernBERT mix5

A clinical PHI token classifier: ModernBERT-base fully fine-tuned as a BIOES
tagger over 14 PHI types. It is meant to run inside the
[Clinical Deidentifier](https://github.com/ewhauser/deidentifier) library,
which adds the decoder, policies and synthetic replacement. Detection is
model-only; there is no rule layer.

> [!WARNING]
> Research software. It has never been evaluated on real clinical records, and
> no sealed qualification has been run for it. It is not a substitute for a
> HIPAA Safe Harbor review or an Expert Determination.

## Model details

| | |
|---|---|
| Base model | `answerdotai/ModernBERT-base` (149M parameters), fully fine-tuned |
| Checkpoint | `models/takeover-mbbase-mix5-001/step-003803` (mix5, epoch 1) |
| Weights | `hf/model.safetensors`, 599 MB, not stored in git |
| Context window | 512 tokens, overlapping windows (overlap 128) for longer text |
| Labels | BIOES tags over 14 types: NAME, DATE, ADDRESS, PHONE, EMAIL, MRN, SSN, ACCOUNT, ID, URL, IP, AGE, LOCATION, FACILITY |
| Decoding | Constrained Viterbi over legal BIOES transitions, exact character offsets |
| License | Apache-2.0 (fine-tuned weights and code) |

Promoted to the default on 2026-10-04 after passing the
[GOALS.md](GOALS.md) regression rules and the blind places diagnostic. The
selection record is `reports/takeover/mix5_001/selection.json`.

## Intended use

- Detecting and redacting PHI in English clinical free text before secondary
  use, with human review.
- Generating patient-consistent synthetic surrogates for development and test
  data.
- Research on synthetic-data training for de-identification.

### Out of scope

- Unreviewed release of clinical records, or any claim of regulatory
  de-identification based on this model alone.
- Languages other than English, OCR output, scanned images, and audio or call
  transcripts (not yet measured).
- Use as a calibrated risk score. Output scores are not probabilities of
  safety.

## Training data

All task-specific text is synthetic: written by this project's generators or
by independently authored synthetic banks. No external clinical dataset was
used. People, dates and identifiers are invented. The
[places addendum](docs/annotation-policy-v2-places.md) curriculum also uses
real public place and care-site names from US government sources as surface
values.

The mixture (`reports/takeover/mix5_001/protocol.json`) has 34 banks, 93,805
notes and 927,603 PHI spans: the earlier v12, remediation and v13 corpora,
19 census-normalized author banks, 4 round-3 author banks, targeted
supplements (correspondence, handoffs, spoken dates, long observations,
identifier, facility, clinical-year, name, record-year and provider-organization
contexts) and the places curriculum. Every bank passed a sanctioned overlap
audit against the sealed qualification bank, and none has text, ID or identity
overlap with the development banks. Span conventions follow the
[annotation policy](docs/annotation-policy-v2.md).

## Training procedure

Weights-only initialization from the mix3 checkpoint `step-003081`, then two
epochs with a fresh optimizer: learning rate 2e-5, batch size 32, warmup 6%,
weight decay 0.01, seed 20261003, Apple silicon (MPS), float32. Epoch 1 was
selected; epoch 2 was blocked by a small loss of direct-identifier and NAME
recall on the held-out gate bank.

## Evaluation

All numbers are model-only and come from synthetic banks; none estimates
accuracy on real notes. "Masked" means every character of a gold span is
covered by some predicted span, whatever its label.

### Held-out synthetic authors

| Bank | Direct identifiers masked | NAME masked (document 95% lower bound) | Indirect masked | Any-label precision | Exact typed F1 |
|---|---:|---:|---:|---:|---:|
| Geriatric (gate bank) | 99.78% | 99.75% (99.65%) | 99.49% | 99.80% | 99.24% |
| Primary care | 99.71% | 99.62% (99.35%) | 98.10% | 99.77% | 99.16% |

Banks written by the training generators score higher and are not evidence of
generalisation.

### Sealed places bank

`blind_places_v1` (2,464 documents, independently authored and reviewed,
scored once): 97.36% of held-out city, abbreviation and short care-site name
characters were masked, against 79.58% for the previous checkpoint. All
PHI-free public-reference documents (named instruments, place-named diseases,
cited institutions, states, place-words) were left untouched.

The complete record is in `reports/takeover/mix5_001/RESULTS.md` and the
[evaluation history](docs/evaluation-history.md). Results for the removed
Privacy-Filter adapter detector are kept there as history.

## Limitations and risks

- **Missed PHI is the primary risk.** About 0.2–0.3% of direct identifiers and
  0.5–1.9% of indirect identifiers on held-out synthetic authors are not fully
  masked.
- **Synthetic-to-real gap.** Real institutional notes are unmeasured and may
  be harder.
- **Terse text.** Very short search-query style text is the weakest layout for
  real place names (about 95% character recall).
- **Over-redaction.** Some PHI-free notes still receive a span; on one
  familiar bank 110 of 120 were left untouched.
- **Identifier typing.** One familiar template ("request … record <code>")
  types some MRNs as ID. The span stays masked.
- **Surrogates are not anonymity.** Synthetic replacements can resemble real
  people, and date shifting changes seasonal context.

## How to use

```python
from clinical_deid import Deidentifier

deid = Deidentifier()  # loads the default checkpoint
print(deid.deidentify("Mara Calder was seen on 2024-02-28.", mode="redact")["text"])
```

## Citation

See [CITATION.cff](CITATION.cff).
