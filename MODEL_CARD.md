---
license: apache-2.0
language:
  - en
base_model: microsoft/deberta-v3-base
pipeline_tag: token-classification
tags:
  - de-identification
  - phi
  - clinical
  - ner
  - privacy
  - synthetic-data
---

# Clinical Deidentifier: DeBERTa-v3 mix9c

A clinical PHI token classifier: DeBERTa-v3-base fully fine-tuned as a BIOES
tagger over 14 PHI types. It is meant to run inside the
[Clinical Deidentifier](https://github.com/ewhauser/deidentifier) library,
which adds the decoder, policies and synthetic replacement. Detection is
model-only; there is no rule layer.

> [!WARNING]
> Research software. It has never been evaluated on real clinical records, and
> no full sealed qualification has been run for it. It is not a substitute for
> a HIPAA Safe Harbor review or an Expert Determination.

## Model details

| | |
|---|---|
| Base model | `microsoft/deberta-v3-base` (184M parameters), fully fine-tuned |
| Checkpoint | `models/takeover-deberta-mix9c-001/step-000845` (`deberta_mix9c_001`) |
| Weights | `hf/model.safetensors`, 736 MB, not stored in git |
| Context window | 512 tokens, overlapping windows (overlap 128) for longer text |
| Labels | BIOES tags over 14 types: NAME, DATE, ADDRESS, PHONE, EMAIL, MRN, SSN, ACCOUNT, ID, URL, IP, AGE, LOCATION, FACILITY |
| Ages | Every age is detected; the request policy's `ages` setting decides what is removed (default `over_89`) |
| Decoding | Constrained Viterbi over legal BIOES transitions, exact character offsets |
| Runtime | DeBERTa's relative-attention bias uses a gather-free implementation with identical outputs (`clinical_deid.detection.deberta_attention`) |
| License | Apache-2.0 (fine-tuned weights and code); base model MIT |

Promoted to the default on 2026-10-08 after passing the
[GOALS.md](GOALS.md) regression rules and the sealed `blind_places_v3`
diagnostic with independent reviews
(`reports/takeover/deberta_mix9c_001/promotion_v3/`). It replaces the
ModernBERT-base mix5 checkpoint.

## Intended use

- Detecting and redacting PHI in English clinical free text before secondary
  use, with human review.
- Generating patient-consistent synthetic surrogates for development and test
  data.
- Research on synthetic-data training for de-identification.

### Out of scope

- Unreviewed release of clinical records, or any claim of regulatory
  de-identification based on this model alone.
- Languages other than English, OCR output and scanned images.
- Use as a calibrated risk score. Output scores are not probabilities of
  safety.

## Training data

All task-specific text is synthetic: written by this project's generators or
by independently authored synthetic banks. No external clinical dataset was
used, and the public benchmarks below are evaluation only. People, dates and
identifiers are invented; the [places addendum](docs/annotation-policy-v2-places.md)
allows real US public place and care-site names as surface values.

Training ran in two stages:

1. `deberta_mix8_001`: three epochs from `microsoft/deberta-v3-base` on 45
   banks (114,115 notes, 147,104 windows): the earlier mixture with every age
   annotated (`__allages2`), plus banks for care-site boundaries (tk21),
   names that are ordinary words (tk22) and ages in context (tk23).
2. `deberta_mix9c_001`: a one-epoch weights-only continuation of stage 1's
   epoch 3 at learning rate 1e-5 on tk24 (named instruments, scores and
   eponyms; 1,500 notes) plus a seeded 20,000-note replay sample of stage 1's
   banks.

Every training bank passed a sealed-overlap audit, has no text or identity
overlap with the development banks or the held-out bank, and is filtered
against the blind banks' hash-only exclusion lists. Span conventions follow
the [annotation policy](docs/annotation-policy-v2.md).

## Evaluation

All numbers come from synthetic or public synthetic banks; none estimates
accuracy on real notes. "Masked" means every character of a gold span is
covered by some predicted span, whatever its label.

### Held-out synthetic author (`heldout_dev_v2`, 2,400 notes)

| Model | Direct identifiers masked | NAME masked (document 95% lower bound) | Indirect masked | Any-label precision | Exact typed F1 | PHI notes fully masked |
|---|---:|---:|---:|---:|---:|---:|
| **This model** | **99.27%** | **99.55% (99.41%)** | **99.22%** | **99.66%** | **97.29%** | **81.71%** |
| mix5 (previous default) | 93.69% | 95.34% (94.81%) | 94.91% | 98.77% | 91.43% | 23.15% |

Ages of 89 and under are detected 91.6% of the time; ages over 89, 100%.
PHI-free notes left untouched: 92.5%.

### Sealed places bank (`blind_places_v3`, 2,352 notes, scored once)

Held-out real city, abbreviation and short care-site names: 98.69% of place
characters masked (lower bound 98.48%) against mix5's 96.87%. Public
references to instruments, diseases, institutions, states and place words
were left untouched in at least 99.29% of documents in every family. Held-out
care-site names are about 48 values, so the bound overstates certainty.

### Public synthetic benchmarks (policy-aligned view 3)

| Benchmark | PHI characters masked | Non-PHI redaction | Direct masked | PHI notes fully masked |
|---|---:|---:|---:|---:|
| ASQ-PHI | 99.56% | 0.16% | 99.85% | 97.60% |
| MedDeID English | 99.34% | 0.02% | 99.91% | 96.00% |
| Technetium-I (5k sample) | 99.65% | 0.57% | 100.00% | 84.60% |

MedDeID's own model scores 99.66%, 99.19% and 99.65% of PHI characters on
the same views, with more non-PHI redaction. ASQ-PHI and MedDeID English have
been inspected and are development evidence. Full tables:
`experiments/external_benchmarks_20261004/results/COMPARISON.md`.

## Limitations and risks

- **Missed PHI is the primary risk.** About 0.7% of direct identifiers and
  0.8% of indirect identifiers on the held-out synthetic author are not fully
  masked; identifiers (ID) and care-site names are the weakest types.
- **Clinician shorthand.** On `blind_places_v3` it masks fewer care-site
  abbreviations in shorthand than mix5 (91.84% against 100%).
- **Synthetic-to-real gap.** Real institutional notes are unmeasured and may
  be harder.
- **Over-redaction.** Some PHI-free notes still receive a span (7.5% of
  PHI-free held-out notes).
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
