---
license: apache-2.0
language:
  - en
base_model: openai/privacy-filter
pipeline_tag: token-classification
tags:
  - de-identification
  - phi
  - clinical
  - ner
  - privacy
  - synthetic-data
---

# Clinical Deidentifier adapter

A clinical PHI token classifier built on the frozen
[OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) encoder.
It is meant to run inside the
[Clinical Deidentifier](https://github.com/ewhauser/deidentifier) library,
which adds structural extractors and synthetic replacement.

> [!WARNING]
> Research software. It has not met its 99% qualification target and has
> never been evaluated on real clinical records. It is not a substitute for a
> HIPAA Safe Harbor review or an Expert Determination.

## Model details

| | |
|---|---|
| Base model | `openai/privacy-filter` at revision `7ffa9a043d54d1be65afb281eddf0ffbe629385b` (frozen) |
| Trained components | Residual bidirectional clinical adapter (hidden 640, bottleneck 160) and a linear BIOES head |
| Trainable parameters | ~450K (1.8 MB `adapter.safetensors`) |
| Context window | 512 tokens, with overlapping windows for longer text |
| Labels | BIOES tags over 14 types: NAME, DATE, ADDRESS, PHONE, EMAIL, MRN, SSN, ACCOUNT, ID, URL, IP, AGE, LOCATION, FACILITY |
| License | Apache-2.0 (adapter and code) |

The default checkpoint (`models/clinical-adapter`) is the fourth stage of the
lineage in the [training guide](docs/training.md). Later candidates are kept
in the repository but have not been promoted.

## Intended use

- Detecting and redacting PHI in English clinical free text before secondary
  use, with human review.
- Generating patient-consistent synthetic surrogates for development and test
  data.
- Research on synthetic-data training for de-identification.

### Out of scope

- Unreviewed release of clinical records, or any claim of regulatory
  de-identification based on this model alone.
- Languages other than English, OCR output, scanned images and audio.
- Use as a calibrated risk score. Output scores are not probabilities of
  safety.

## Training data

All task-specific data is synthetic and was authored by the generators in
`src/clinical_deid/`. No external clinical dataset was downloaded or used.
People, dates and identifiers are invented. The
[places addendum](docs/annotation-policy-v2-places.md) lets newer generators
also use real public place and care-site names from US government sources as
surface values; the default checkpoint was not trained on them. The default
checkpoint saw 22,800 training notes:

| Corpus | Notes | PHI spans |
|---|---:|---:|
| Base synthetic notes | 12,000 | 73,655 |
| Development notes converted to training | 800 | 9,413 |
| Dense augmentation | 4,000 | 52,610 |
| Authored context (geography, ages, contacts) | 3,000 | 30,750 |
| Authored coverage (roles, headers, name styles, care settings) | 3,000 | 51,000 |

Checkpoint selection used 3,200 separate synthetic validation notes. The
corpus text and annotations are released under CC0-1.0. Span conventions
follow the [annotation policy](docs/annotation-policy-v2.md).

## Training procedure

The backbone stays frozen and its features are cached. The adapter and head
are trained with token-level cross-entropy, resuming across four stages. The
final stage used learning rate 1e-4, 16 epochs and seed 1729. It selected
epoch 6 by mean validation cross-entropy, on Apple silicon (MPS) with float16
features.

## Evaluation

The library runs this model in a **hybrid** detector together with
deterministic structural extractors and context guards. Model-only and hybrid
results are always reported separately. Metrics are exact typed-span
precision, recall and F1 unless noted.

### Default checkpoint: V4 frozen final test

1,200 synthetic notes and 14,121 PHI spans, evaluated once:

| Detector | Precision | Recall | F1 |
|---|---:|---:|---:|
| Model only | 96.18% | 95.68% | 95.93% |
| Hybrid | 97.39% | 97.28% | 97.34% |

Hybrid PHI-character recall was 99.87%, and 97.25% of documents were
leak-free. The validation set scored 99.82% F1, but that did not transfer to
the final test. Later, independently authored banks were harder: on the V5
blind bank a later checkpoint reached 79.74% hybrid F1.

### Latest candidate: V14 qualification

This run used the `clinical-adapter-v12-warm-001` epoch-2 weights with the V14
hybrid layer. It was one frozen run on 2,160 synthetic notes and 16,632 spans,
and passed 2 of 7 gates.

| Detector | Precision | Recall | F1 | PHI-char recall | Leak-free PHI notes |
|---|---:|---:|---:|---:|---:|
| Model only | 96.46% | 94.33% | 95.38% | 98.19% | 60.17% |
| Hybrid | 96.37% | 98.88% | 97.61% | 99.54% | 91.77% |

### External benchmark

On [ASQ-PHI](https://doi.org/10.17632/csz5dzp7nx.1) (1,051 synthetic clinical
search queries, MIT licence), the hybrid detector covered 95.25% of PHI
characters and redacted 1.03% of non-PHI characters. MedDeID scored 99.14% and
6.33% on the same measures. The full protocol and comparison are in
[external verification](docs/external-verification.md).

The complete record is in the [evaluation history](docs/evaluation-history.md).

## Limitations and risks

- **Missed PHI is the primary risk.** NAME recall is weakest, especially for
  nicknames, aliases and single-token names in narrative text. On one
  160-note development challenge, NAME recall was 38.4%.
- **Synthetic-to-real gap.** Every internal score comes from text written by
  this project's generators. Performance on real institutional notes is
  unknown and probably lower.
- **Location coverage.** On ASQ-PHI, the detector misses standalone city
  names and institution abbreviations such as "UCSF".
- **Over-redaction.** Bare years and medical eponyms ("Chaddock reflex")
  sometimes cause false positives.
- **Surrogates are not anonymity.** Synthetic replacements can resemble real
  people, and date shifting changes seasonal context.

## How to use

```python
from clinical_deid import Deidentifier

deid = Deidentifier("models/clinical-adapter")
print(deid.deidentify("Mara Calder was seen on 2024-02-28.", mode="redact")["text"])
```

## Citation

See [CITATION.cff](CITATION.cff).
