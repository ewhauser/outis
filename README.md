# Clinical Deidentifier

[![CI](https://github.com/ewhauser/deidentifier/actions/workflows/ci.yml/badge.svg)](https://github.com/ewhauser/deidentifier/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Local PHI detection, typed redaction and patient-consistent synthetic
replacement for English clinical text.

The detector is a fully fine-tuned
[ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) token
classifier with a constrained BIOES decoder. There is no rule layer: every
detected span comes from the model. Everything runs on your machine; no text
is sent to a hosted service. It ships as a Python library, a CLI and a FastAPI
server.

> [!IMPORTANT]
> This is research software. It has **not** met its [project goal](GOALS.md)
> or its 99% exact-span qualification target, and it has been evaluated only on
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
git clone https://github.com/ewhauser/deidentifier.git
cd deidentifier
uv venv --python 3.13
uv pip install --python .venv/bin/python -e .
```

The default model is `models/takeover-mbbase-mix5-001/step-003803`. Its
configuration and tokenizer are in the repository; the 599 MB weight file
(`hf/model.safetensors`) is not, and must be placed there from the project's
model store. Add the `pdf` extra (`-e '.[pdf]'`) for PDF file input.

### Python

```python
from clinical_deid import Deidentifier

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
text ─▶ ModernBERT-base token classifier (overlapping 512-token windows)
          └─▶ constrained BIOES decoder ─▶ spans at original offsets ─▶ redact / replace
```

The default model was fine-tuned on 93,805 synthetic notes from 34 banks
written by the generators and independent authors in this repository. No real
clinical data was used for training. See the [model card](MODEL_CARD.md).

## Results

These figures come from synthetic or public benchmarks and are **not**
estimates of accuracy on real clinical notes.

**Default model (mix5 epoch 1), held-out synthetic authors.** Two
development banks written by authors the training data never saw. "Masked"
means every character of a gold span is covered under any label.

| Bank | Direct identifiers masked | NAME masked (document 95% lower bound) | Indirect masked | Any-label precision | Exact typed F1 |
|---|---:|---:|---:|---:|---:|
| Geriatric | 99.78% | 99.75% (99.65%) | 99.49% | 99.80% | 99.24% |
| Primary care | 99.71% | 99.62% (99.35%) | 98.10% | 99.77% | 99.16% |

**Sealed places bank.** Held-out real city, abbreviation and short care-site
names: 97.36% of characters masked; all public-reference negative families
(instruments, diseases, cited institutions, states, place-words) left
untouched. See [mix5 results](reports/takeover/mix5_001/RESULTS.md).

Historical results for the removed adapter detector, including the V14
qualification and the ASQ-PHI comparison, are in the
[evaluation history](docs/evaluation-history.md) and
[external verification](docs/external-verification.md).

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
| [Goals](GOALS.md) | The accepted project goal, targets and regression rules |
| [API guide](docs/api.md) | Endpoints, policies, masks, structured data, files, WebSockets |
| [Synthetic replacement](docs/replacement.md) | Secrets, patient anchors, date shifting, identity linking |
| [Deployment](docs/deployment.md) | Launcher, default model, batch limits, security controls |
| [Model card](MODEL_CARD.md) | Architecture, training data, evaluation, risks |
| [Training](docs/training.md) | Training candidates and generating corpora |
| [Evaluation history](docs/evaluation-history.md) | All qualification and development results |
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
dedicated to the public domain under CC0-1.0. The ModernBERT-base base model
is Apache-2.0; see [NOTICE](NOTICE).
