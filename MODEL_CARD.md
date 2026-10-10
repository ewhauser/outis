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

# Outis: DeBERTa-v3 mix9

A clinical PHI token classifier: DeBERTa-v3-base fully fine-tuned as a BIOES
tagger over 14 PHI types. It is meant to run inside the
[Outis](https://github.com/ewhauser/outis) library,
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
| Checkpoint | `models/takeover-deberta-mix9-001/step-009288` (`deberta_mix9_001`, epoch 2) |
| Weights | `hf/model.safetensors`, 736 MB, not stored in git |
| Context window | 512 tokens, overlapping windows (overlap 128) for longer text |
| Labels | BIOES tags over 14 types: NAME, DATE, ADDRESS, PHONE, EMAIL, MRN, SSN, ACCOUNT, ID, URL, IP, AGE, LOCATION, FACILITY |
| Ages | Every age is detected; the request policy's `ages` setting decides what is removed (default `over_89`) |
| Decoding | Constrained Viterbi over legal BIOES transitions, exact character offsets |
| Runtime | DeBERTa's relative-attention bias uses a gather-free implementation with identical outputs (`outis.detection.deberta_attention`) |
| License | Apache-2.0 (fine-tuned weights and code); base model MIT |

Promoted to the default on 2026-10-09 by the owner. It passed the
[GOALS.md](GOALS.md) regression rules and the place-recall rules of the sealed
`blind_places_v3` diagnostic, but **failed that bank's false-positive rule on
region-only documents**; the owner overrode the failure
(`reports/takeover/deberta_mix9_001/promotion_v3/OWNER_DECISION.md`). It
replaces `deberta_mix9c_001` step 845, which replaced ModernBERT-base mix5.

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

`deberta_mix9_001` is a single run from `microsoft/deberta-v3-base`: three
epochs (13,932 steps, learning rate 3e-5, batch 32) on 46 banks, the earlier
mixture with every age annotated (`__allages2`) plus banks for care-site
boundaries (tk21), names that are ordinary words (tk22), ages in context
(tk23) and named instruments, scores and eponyms (tk24). Epoch 2 (step 9288)
was chosen by the privacy-first selection rule on the development banks.

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
| **This model** | **99.64%** | **99.79% (99.71%)** | **98.97%** | **99.63%** | **97.34%** | **81.90%** |
| `deberta_mix9c_001` (previous default) | 99.27% | 99.55% (99.41%) | 99.22% | 99.66% | 97.29% | 81.71% |
| mix5 | 93.69% | 95.34% (94.81%) | 94.91% | 98.77% | 91.43% | 23.15% |

Ages of 89 and under are detected 90.5% of the time; ages over 89, 100%.
PHI-free notes left untouched: 92.1%.

### Sealed places bank (`blind_places_v3`, 2,352 notes, scored once)

Held-out real city, abbreviation and short care-site names: 98.81% of place
characters masked (lower bound 98.61%) against the previous default's 98.69%.
Public references to instruments, diseases, institutions and place words were
left untouched in at least 99% of documents per family, but **bare states,
countries or regions only in 96%** (previous default 100%): the bank's
false-positive rule failed there. Held-out care-site names are about 48
values, so the bound overstates certainty.

### Public synthetic benchmarks (policy-aligned view 3)

| Benchmark | PHI characters masked | Non-PHI redaction | Direct masked | PHI notes fully masked |
|---|---:|---:|---:|---:|
| ASQ-PHI | 99.94% | 0.18% | 99.93% | 99.52% |
| MedDeID English | 99.27% | 0.01% | 99.83% | 95.00% |
| Technetium-I (5k sample) | 99.65% | 0.57% | 100.00% | 84.60% |

MedDeID's own model scores 99.66%, 99.19% and 99.65% of PHI characters on
the same views, with more non-PHI redaction. ASQ-PHI and MedDeID English have
been inspected and are development evidence. Full tables:
`experiments/external_benchmarks_20261004/results/COMPARISON.md`.

## Limitations and risks

- **Missed PHI is the primary risk.** About 0.4% of direct identifiers and
  1.0% of indirect identifiers on the held-out synthetic author are not fully
  masked; identifiers (ID) and care-site names are the weakest types.
- **Region over-redaction.** On `blind_places_v3` it tags a bare state,
  country or region in 4 of 100 PHI-free documents (previous default 0).
- **Care sites.** Care-site names are masked slightly less often than by the
  previous default on the held-out author (FACILITY 96.75%).
- **Synthetic-to-real gap.** Real institutional notes are unmeasured and may
  be harder.
- **Over-redaction.** Some PHI-free notes still receive a span (7.9% of
  PHI-free held-out notes).
- **Surrogates are not anonymity.** Synthetic replacements can resemble real
  people, and date shifting changes seasonal context.

## How to use

```python
from outis import Deidentifier

deid = Deidentifier()  # loads the default checkpoint
print(deid.deidentify("Mara Calder was seen on 2024-02-28.", mode="redact")["text"])
```

## Citation

See [CITATION.cff](CITATION.cff).
