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

# Outis: DeBERTa-v3-base

A clinical PHI token classifier: DeBERTa-v3-base fully fine-tuned as a BIOES
tagger over 14 PHI types. It is meant to run inside the
[Outis](https://github.com/ewhauser/outis) library,
which adds the decoder, policies and synthetic replacement. Detection is
model-only; there is no rule layer.

> [!WARNING]
> Research software. On the one set of real clinical notes it has been tested
> on (PhysioNet nursing notes) it masked 77% of PHI characters, far below its
> synthetic scores, and no full sealed qualification has been run for it. It is not a substitute for
> a HIPAA Safe Harbor review or an Expert Determination.

## Model details

| | |
|---|---|
| Base model | `microsoft/deberta-v3-base` (184M parameters), fully fine-tuned |
| Checkpoint | `models/deid-deberta-v3-base` |
| Weights | `hf/model.safetensors`, 736 MB, on Hugging Face: [ewhauser/outis-deberta-v3-base](https://huggingface.co/ewhauser/outis-deberta-v3-base) |
| Context window | 512 tokens, overlapping windows (overlap 128) for longer text |
| Labels | BIOES tags over 14 types: NAME, DATE, ADDRESS, PHONE, EMAIL, MRN, SSN, ACCOUNT, ID, URL, IP, AGE, LOCATION, FACILITY |
| Ages | Every age is detected; the request policy's `ages` setting decides what is removed (default `over_89`) |
| Decoding | Constrained Viterbi over legal BIOES transitions, exact character offsets |
| Runtime | DeBERTa's relative-attention bias uses a gather-free implementation with identical outputs (`outis.detection.deberta_attention`) |
| License | Apache-2.0 (fine-tuned weights and code); base model MIT |

It meets the [comparison rules](docs/evaluation.md) against the previous
model and the place-recall checks of the blind places bank, but tags bare
states and regions slightly more often than its predecessor (see
Limitations).

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
allows real US public place and care-site names as surface values. The
training and selection corpora are published as
[ewhauser/outis-synthetic-phi](https://huggingface.co/datasets/ewhauser/outis-synthetic-phi).

The model is fine-tuned from `microsoft/deberta-v3-base` for three epochs
(learning rate 3e-5, batch 32) on 46 synthetic banks covering general
clinical notes and targeted cases: care-site names and boundaries, names that
are ordinary words, ages in context, and named instruments, scores and
eponyms that are not PHI. The checkpoint after the second epoch was chosen by
a privacy-first rule on separate selection banks: direct-identifier recall
first, then indirect recall, then precision.

Every training bank passed a sealed-overlap audit, has no text or identity
overlap with the development banks or the held-out bank, and is filtered
against the blind banks' hash-only exclusion lists. Span conventions follow
the [annotation policy](docs/annotation-policy-v2.md).

## Evaluation

One benchmark is real clinical text (PhysioNet); every other number comes from
synthetic text.
"Masked" means every character of an identifier is covered by some detected
span, whatever type it was given.

### Real clinical notes

[PhysioNet Deidentified Medical Text](https://physionet.org/content/deidentifiedmedicaltext/1.0/):
2,434 real MIMIC-II nursing notes with every identifier replaced by a realistic
fictional value. Credentialed, research-only data, run locally; only aggregate
scores are kept.

| Measure | Outis | [MedDeID](https://huggingface.co/stighellemans/meddeid-english-synth) | [GLiNER-PII](https://huggingface.co/nvidia/gliner-PII)\* |
|---|---:|---:|---:|
| PHI characters masked ↑ | **77.30%** | 66.54% | 75.33% |
| Non-PHI text removed ↓ | **1.06%** | 3.91% | 1.35% |
| Notes fully masked ↑ | **54.94%** | 39.81% | 51.50% |
| Names masked ↑ | **75.44%** | 72.06% | 70.46% |

\* `nvidia/gliner-PII` with one added zero-shot label for hospitals.

PhysioNet's rule-based `deid` tool reported 96.7% instance recall on these
notes (tuned on them, run on the original text; about 94% held out). See
[against published figures](docs/external-verification.md#against-published-figures).

### Synthetic public benchmarks

Every model is scored the same way, under this project's annotation policy
([details](docs/external-verification.md)). ↑ means higher is better, ↓ lower
is better; the best result in each row is in bold.

| Benchmark | Measure | Outis | [MedDeID](https://huggingface.co/stighellemans/meddeid-english-synth) | [GLiNER-PII](https://huggingface.co/nvidia/gliner-PII)\* |
|---|---|---:|---:|---:|
| [ASQ-PHI](https://doi.org/10.17632/csz5dzp7nx.1)<br>1,051 search queries | PHI characters masked ↑ | **99.94%** | 99.66% | 98.57% |
|  | Non-PHI text removed ↓ | **0.18%** | 0.91% | 2.43% |
|  | Notes fully masked ↑ | **99.52%** | 97.60% | 94.47% |
| [MedDeID English](https://huggingface.co/datasets/stighellemans/meddeid-english-synthetic-benchmark)<br>300 notes | PHI characters masked ↑ | **99.27%** | 99.19% | 94.87% |
|  | Non-PHI text removed ↓ | **0.01%** | 0.15% | 0.64% |
|  | Notes fully masked ↑ | **95.00%** | 93.00% | 69.67% |
| [Technetium-I](https://huggingface.co/datasets/temlm-foundation/Technetium-I)<br>5,000-note sample | PHI characters masked ↑ | 99.65% | 99.65% | 98.83% |
|  | Non-PHI text removed ↓ | **0.57%** | 1.14% | 0.78% |
|  | Notes fully masked ↑ | 84.60% | 84.60% | 43.32% |

\* `nvidia/gliner-PII` with one added zero-shot label for hospitals.

We have looked at Outis's failures on ASQ-PHI and MedDeID English while
developing it, so those two are not fully independent tests; Technetium-I has
not been inspected. Per-identifier results are in the
[full comparison](benchmarks/results/COMPARISON.md).

### Held-out synthetic notes (2,400 notes)

Written by an author whose text was never used in training.

| Measure | Result |
|---|---:|
| Direct identifiers masked | 99.64% |
| Names masked (95% lower bound) | 99.79% (99.71%) |
| Indirect identifiers masked | 98.97% |
| Detected spans that are real PHI | 99.63% |
| Exact type and boundary match (F1) | 97.34% |
| Notes with every identifier masked, ages over 89* | 86.25% |
| Notes with every identifier masked, every age | 81.90% |
| PHI-free notes left untouched | 92.1% |
| Ages 89 and under detected | 90.5% |

\* Ages of 89 and under stay in the output under the default request policy
(`ages: over_89`), following HIPAA Safe Harbor's age rule; `ages: all` removes
them too.

### Real place names (2,352 notes)

98.81% of real city, abbreviation and care-site name characters masked (95%
lower bound 98.61%). PHI-free mentions of instruments, diseases,
institutions and place words were left alone in at least 99% of notes, but a
bare state, country or region was tagged in 4% of notes that contained no
PHI. The care-site names rest on about 48 distinct values, so the bound
overstates certainty.

## Limitations and risks

- **Missed PHI is the primary risk.** About 0.4% of direct identifiers and
  1.0% of indirect identifiers on the held-out synthetic author are not fully
  masked; identifiers (ID) and care-site names are the weakest types.
- **Region over-redaction.** It sometimes tags a bare state, country or
  region in notes with no PHI (4% on the place-name test).
- **Care sites.** Care-site names are the weakest indirect type on the
  held-out notes (96.75% masked).
- **Synthetic-to-real gap.** On real nursing notes 77% of PHI characters were
  masked and 55% of notes with PHI were fully masked, against 99% or more on
  synthetic text. Other real note types and institutions are unmeasured.
- **Over-redaction.** Some PHI-free notes still receive a span (7.9% of
  PHI-free held-out notes).
- **Surrogates are not anonymity.** Synthetic replacements can resemble real
  people, and date shifting changes seasonal context.

## How to use

```python
from outis import Deidentifier

deid = Deidentifier()  # downloads this model once, pinned to a tested revision
print(deid.deidentify("Mara Calder was seen on 2024-02-28.", mode="redact")["text"])
```

## Citation

See [CITATION.cff](CITATION.cff).
