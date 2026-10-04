# Clinical Deidentifier

[![CI](https://github.com/ewhauser/deidentifier/actions/workflows/ci.yml/badge.svg)](https://github.com/ewhauser/deidentifier/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Local PHI detection, typed redaction and patient-consistent synthetic
replacement for English clinical text.

The detector adds a small trainable clinical adapter and BIOES NER head to the
frozen [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter)
encoder. Deterministic structural extractors for emails, phones, identifiers
and other formats run alongside it. Everything runs on your machine; no text
is sent to a hosted service. It ships as a Python library, a CLI and a FastAPI
server.

> [!IMPORTANT]
> This is research software. It has **not** met its own 99% qualification
> target, and it has been evaluated only on synthetic and public benchmark
> text, never on real clinical records. It does not by itself provide HIPAA
> Safe Harbor or Expert Determination de-identification. See
> [Limitations](#limitations).

## Features

- **14 PHI types**: NAME, DATE, AGE, ADDRESS, LOCATION, FACILITY, PHONE, EMAIL,
  URL, IP, MRN, SSN, ACCOUNT and ID.
- **Redaction or replacement**: typed markers or masks, or fictional values that
  stay consistent per patient. Dates shift by a stable per-patient offset that
  preserves intervals.
- **Hybrid detection**: learned NER plus structural extractors, merged with the
  original character offsets intact.
- **Configurable policies**: per-type actions, custom identifier rules, linked
  messages, explicit restoration, and JSON, CSV, XML and plain-text file input.
- **Local by design**: loopback-only by default, optional API key, no request
  logging, `Cache-Control: no-store` on responses.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/ewhauser/deidentifier.git
cd deidentifier
uv venv --python 3.13
uv pip install --python .venv/bin/python -e .
.venv/bin/python scripts/bootstrap.py   # downloads the pinned base encoder
```

The trained adapter is included in `models/clinical-adapter`.

### Python

```python
from clinical_deid import Deidentifier

deid = Deidentifier(
    "models/clinical-adapter", secret=b"application-secret-at-least-32-bytes"
)

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
.venv/bin/python scripts/serve_local.py --port 8000
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
.venv/bin/clinical-deid text --input note.txt --mode redact
```

## How it works

```
text ─▶ OpenAI Privacy Filter encoder (frozen)
          └─▶ residual bidirectional clinical adapter ─▶ BIOES NER head ─┐
text ─▶ structural extractors (email, phone, IDs, dates, …) ─────────────┼─▶ merge ─▶ redact / replace
                                                                         │
          clinical context guards (care sites, names, record fields) ───┘
```

The default adapter has about 450K trainable parameters (1.8 MB). It was
trained on 22,800 synthetic notes written by the generators in this
repository. No real clinical data was used for training. See the
[model card](MODEL_CARD.md) for details.

## Results

All figures are exact typed-span metrics unless noted. They come from
synthetic or public benchmarks and are **not** estimates of accuracy on real
clinical notes.

**Internal qualification (V14).** This was one frozen run on 2,160
independently authored synthetic notes with 16,632 PHI spans. It used the
`clinical-adapter-v12-warm-001` epoch-2 weights, which are not yet the default
checkpoint, and passed 2 of 7 gates.

| Detector | Precision | Recall | F1 | PHI-char recall | Leak-free PHI notes |
|---|---:|---:|---:|---:|---:|
| Model only | 96.46% | 94.33% | 95.38% | 98.19% | 60.17% |
| Hybrid | 96.37% | 98.88% | 97.61% | 99.54% | 91.77% |

**External: [ASQ-PHI](https://doi.org/10.17632/csz5dzp7nx.1)** (1,051
synthetic clinical search queries, label-agnostic):

| System | Char recall | Non-PHI redaction | Exact-span F1 | Clean hard negatives |
|---|---:|---:|---:|---:|
| This project (hybrid, pre-fix) | 95.25% | 1.03% | 83.19% | 84.47% |
| MedDeID (`meddeid-english-synth`) | 99.14% | 6.33% | 79.79% | 14.61% |
| GLiNER-PII + facility label | 96.81% | 8.27% | 70.65% | 27.40% |

This detector over-redacts the least but misses more PHI, mostly standalone
cities and institution abbreviations. ASQ-PHI has since guided a fix, so it is
now development evidence. See [external verification](docs/external-verification.md)
and the [full evaluation history](docs/evaluation-history.md).

## Limitations

- **NAME recall is the main weakness.** On a 160-note development challenge,
  the current system found 38.4% of NAME spans and 5.7% of nicknames and
  aliases.
- **Synthetic evaluation only.** Training and internal test text comes from
  this project's own generators. Real institutional formats, rare identifying
  narratives and unseen identity styles are not measured.
- **Scope.** English plain text only. OCR, scanned images, audio and
  multilingual text are out of scope.
- **Some options return PHI.** Restoration mappings, sensitive entity analysis
  and per-type `keep` deliberately return original values. Treat those
  responses as PHI.
- **Scores are not calibrated.** Model and rule scores are not safety
  probabilities.
- **Replacement is not anonymity.** Fictional values can resemble real people,
  and date shifting changes seasonal context. Surrogates do not guarantee
  resistance to re-identification.

Use human review and an appropriate privacy determination before releasing
any output.

## Documentation

| Guide | Contents |
|---|---|
| [API guide](docs/api.md) | Endpoints, policies, masks, structured data, files, WebSockets |
| [Synthetic replacement](docs/replacement.md) | Secrets, patient anchors, date shifting, identity linking |
| [Deployment](docs/deployment.md) | Launcher, backends, batch limits, security controls |
| [Model card](MODEL_CARD.md) | Architecture, training data, evaluation, risks |
| [Training](docs/training.md) | Reproducing the adapter and generating corpora |
| [Evaluation history](docs/evaluation-history.md) | All qualification and development results |
| [External verification](docs/external-verification.md) | Public benchmark protocol and results |
| [Annotation policy](docs/annotation-policy-v2.md) | Span conventions for every entity type |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues privately, as
described in [SECURITY.md](SECURITY.md). Never put real PHI in an issue, pull
request or test fixture.

## Citation

If you use this project, please cite it using [CITATION.cff](CITATION.cff).

## License

Source code and the fine-tuned adapter are licensed under
[Apache-2.0](LICENSE). Generated synthetic corpus text and annotations are
dedicated to the public domain under CC0-1.0. The OpenAI Privacy Filter base
model is Apache-2.0; see [NOTICE](NOTICE).
