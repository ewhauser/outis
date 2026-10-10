# Outis

[![CI](https://github.com/ewhauser/outis/actions/workflows/ci.yml/badge.svg)](https://github.com/ewhauser/outis/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Local PHI detection, typed redaction and patient-consistent synthetic
replacement for English clinical text.

Outis is Greek for "nobody": the name Odysseus gave the Cyclops.

The detector is a fully fine-tuned
[DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) token
classifier with a constrained BIOES decoder. There is no rule layer: every
detected span comes from the model. Everything runs on your machine; no text
is sent to a hosted service. It ships as a Python library, a CLI and a FastAPI
server.

> [!IMPORTANT]
> This is research software. It has **not** met all of its [evaluation targets](docs/evaluation.md), and it has been evaluated only on
> synthetic and public benchmark text, never on real clinical records. It does
> not by itself provide HIPAA Safe Harbor or Expert Determination
> de-identification. See [Limitations](#limitations).

## Features

- **14 PHI types**: NAME, DATE, AGE, ADDRESS, LOCATION, FACILITY, PHONE, EMAIL,
  URL, IP, MRN, SSN, ACCOUNT and ID.
- **Redaction or replacement**: typed markers or masks, or fictional values that
  stay consistent per patient. Dates shift by a stable per-patient offset that
  preserves intervals.
- **Model-only detection**: one fine-tuned encoder, decoded to exact original
  character offsets.
- **Configurable policies**: per-type actions, custom identifier rules, linked
  messages, explicit restoration, and JSON, CSV, XML and plain-text file input.
- **Local by design**: loopback-only by default, optional API key, no request
  logging, `Cache-Control: no-store` on responses.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/ewhauser/outis.git
cd outis
uv sync
```

The default model directory is `models/deid-deberta-v3-base`.
Its configuration and tokenizer are in the repository; the 736 MB weight file
(`hf/model.safetensors`) is distributed separately with the model card and
must be placed in `hf/`. Add the `pdf` extra (`uv sync --extra pdf`) for PDF file input.

### Python

Run it with `uv run python`.

```python
from outis import Deidentifier

deid = Deidentifier(secret=b"application-secret-at-least-32-bytes")

result = deid.deidentify("Mara Calder was seen on 2024-02-28.", mode="redact")
print(result["text"])

synthetic = deid.deidentify(
    "Mara Calder was seen on 2024-02-28.",
    mode="synthetic",
    patient_id="internal-patient-42",
)
print(synthetic["text"])
```

Each result contains the transformed text and, for every entity, its type,
original character offsets, replacement and score. It also lists any warnings
and the model version.

### HTTP API

```sh
uv run scripts/serve_local.py --port 8000
```

```sh
curl http://127.0.0.1:8000/v1/deidentify \
  -H 'Content-Type: application/json' \
  -d '{"text":"Patient Mara Calder, DOB 1974-06-13. Creatinine 1.2 mg/dL.","mode":"redact"}'
```

Interactive docs are served at `http://127.0.0.1:8000/docs`, and the OpenAPI
schema at `/openapi.json`. The local launcher creates and reuses a replacement
secret in `.cache/local-api-secret`. For other deployments, see
[deployment](docs/deployment.md).

### CLI

```sh
uv run outis text --input note.txt --mode redact
```

## How it works

```
text ─▶ DeBERTa-v3-base token classifier (overlapping 512-token windows)
          └─▶ constrained BIOES decoder ─▶ spans at original offsets ─▶ redact / replace
```

The default model was fine-tuned on about 116,000 synthetic notes from 46
banks written by this project's generators and independent synthetic authors. No real
clinical data was used for training. See the [model card](MODEL_CARD.md).

## Results

All results come from synthetic text. None of them measures accuracy on real
clinical notes.

### Public benchmarks

Share of PHI characters masked, and share of non-PHI text removed
unnecessarily, scored the same way for every model
([details](docs/external-verification.md)):

| Benchmark | This model | [MedDeID](https://huggingface.co/stighellemans/meddeid-english-synth) | [GLiNER-PII](https://huggingface.co/nvidia/gliner-PII)* |
|---|---:|---:|---:|
| [ASQ-PHI](https://doi.org/10.17632/csz5dzp7nx.1), 1,051 search queries | **99.94%** masked, 0.18% over | 99.66% masked, 0.91% over | 98.57% masked, 2.43% over |
| [MedDeID English](https://huggingface.co/datasets/stighellemans/meddeid-english-synthetic-benchmark), 300 notes | **99.27%** masked, 0.01% over | 99.19% masked, 0.15% over | 94.87% masked, 0.64% over |
| [Technetium-I](https://huggingface.co/datasets/temlm-foundation/Technetium-I), 5,000-note sample | 99.65% masked, **0.57%** over | 99.65% masked, 1.14% over | not run |

\* `nvidia/gliner-PII` with one added zero-shot label for hospitals.

We have looked at this model's failures on ASQ-PHI and MedDeID English while
developing it, so those two are not fully independent tests; Technetium-I has
not been inspected.

Per-identifier results are in the [full comparison](benchmarks/results/COMPARISON.md).

### Held-out synthetic notes

On 2,400 synthetic notes from an author whose writing was never used in
training:

| Measure | Result |
|---|---:|
| Direct identifiers masked (names, record numbers, phone numbers, emails, addresses and similar) | 99.64% |
| Names masked (95% lower bound) | 99.79% (99.71%) |
| Indirect identifiers masked (dates, ages over 89, places, facilities) | 98.97% |
| Detected spans that are real PHI | 99.63% |
| Exact type and boundary match (F1) | 97.34% |
| Notes with every identifier masked* | 86.25% |

\* Counts ages over 89, the ages HIPAA's Safe Harbor method lists as
identifiers; ages of 89 and under stay in the output by default. The model
detects every age, so setting the request policy's `ages` option to `all`
removes younger ages too (see the [API guide](docs/api.md)). Counting every
age, 81.90% of notes are fully masked.

On these same notes MedDeID masked 86.08% of direct identifiers and GLiNER-PII
79.09%. The notes follow this project's annotation conventions, which this
model was trained on, so that gap overstates its advantage; the public
benchmarks above are the fair comparison.

A separate test of real place names (2,352 notes) masked 98.81% of city and
care-site name characters, but tagged a bare state or country in 4% of notes
that contained no PHI.

How these numbers are computed is described in
[evaluation methodology](docs/evaluation.md).

## Limitations

- **Not yet measured on real notes.** Held-out synthetic authors are the best
  available evidence; real institutional notes may differ substantially.
- **Synthetic evaluation only.** Training and internal test text comes from
  this project's own generators. Real institutional formats, rare identifying
  narratives and unseen identity styles are not measured.
- **Scope.** English plain text only. OCR, scanned images, audio and
  multilingual text are out of scope.
- **Some options return PHI.** Restoration mappings, sensitive entity analysis
  and per-type `keep` deliberately return original values. Treat those
  responses as PHI.
- **Scores are not calibrated.** Model scores are not safety probabilities.
- **Replacement is not anonymity.** Fictional values can resemble real people,
  and date shifting changes seasonal context. Surrogates do not guarantee
  resistance to re-identification.

Use human review and an appropriate privacy determination before releasing
any output.

## Documentation

| Guide | Contents |
|---|---|
| [Evaluation methodology](docs/evaluation.md) | Targets, any-label masking and how models are compared |
| [API guide](docs/api.md) | Endpoints, policies, masks, structured data, files, WebSockets |
| [Synthetic replacement](docs/replacement.md) | Secrets, patient anchors, date shifting, identity linking |
| [Deployment](docs/deployment.md) | Launcher, default model, batch limits, security controls |
| [Model card](MODEL_CARD.md) | Architecture, training data, evaluation, risks |
| [Training](docs/training.md) | Training candidates and generating corpora |
| [External verification](docs/external-verification.md) | Public benchmark protocol and results |
| [Annotation policy](docs/annotation-policy-v2.md) | Span conventions for every entity type |
| [Places addendum](docs/annotation-policy-v2-places.md) | Real localities, care-site abbreviations and public references |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues privately, as
described in [SECURITY.md](SECURITY.md). Never put real PHI in an issue, pull
request or test fixture.

## Citation

If you use this project, please cite it using [CITATION.cff](CITATION.cff).

## License

Source code and the fine-tuned model are licensed under
[Apache-2.0](LICENSE). Generated synthetic corpus text and annotations are
dedicated to the public domain under CC0-1.0. The DeBERTa-v3-base base model
is MIT-licensed; see [NOTICE](NOTICE).
