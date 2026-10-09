# Clinical Deidentifier

[![CI](https://github.com/ewhauser/deidentifier/actions/workflows/ci.yml/badge.svg)](https://github.com/ewhauser/deidentifier/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Local PHI detection, typed redaction and patient-consistent synthetic
replacement for English clinical text.

The detector is a fully fine-tuned
[DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) token
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

The default model is `models/takeover-deberta-mix9-001/step-009288`. Its
configuration and tokenizer are in the repository; the 736 MB weight file
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
text ─▶ DeBERTa-v3-base token classifier (overlapping 512-token windows)
          └─▶ constrained BIOES decoder ─▶ spans at original offsets ─▶ redact / replace
```

The default model was fine-tuned on about 116,000 synthetic notes from 46
banks written by the generators and independent authors in this repository. No real
clinical data was used for training. See the [model card](MODEL_CARD.md).

## Results

These figures come from synthetic or public benchmarks and are **not**
estimates of accuracy on real clinical notes.

**Default model (`deberta_mix9_001`), held-out synthetic author.**
`heldout_dev_v2`, 2,400 notes by an author the training data never saw.
"Masked" means every character of a gold span is covered under any label.

| Model | Direct identifiers masked | NAME masked (document 95% lower bound) | Indirect masked | Any-label precision | Exact typed F1 | PHI notes fully masked |
|---|---:|---:|---:|---:|---:|---:|
| Default (DeBERTa-v3, mix9) | 99.64% | 99.79% (99.71%) | 98.97% | 99.63% | 97.34% | 81.90% |
| Previous default (DeBERTa-v3, mix9c) | 99.27% | 99.55% (99.41%) | 99.22% | 99.66% | 97.29% | 81.71% |
| ModernBERT, mix5 | 93.69% | 95.34% (94.81%) | 94.91% | 98.77% | 91.43% | 23.15% |

**Sealed places bank (`blind_places_v3`).** Held-out real city,
abbreviation and short care-site names: 98.81% of place characters masked
(previous default: 98.69%). Region-only PHI-free documents were left
untouched in 96% of cases against 100%, which failed the bank's
false-positive rule; the owner promoted the model anyway. See the
[result](reports/takeover/deberta_mix9_001/promotion_v3/RESULT.md) and the
[owner decision](reports/takeover/deberta_mix9_001/promotion_v3/OWNER_DECISION.md).

**Public synthetic benchmarks** (policy-aligned):
[external comparison](experiments/external_benchmarks_20261004/results/COMPARISON.md).

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
dedicated to the public domain under CC0-1.0. The DeBERTa-v3-base base model
is MIT-licensed; see [NOTICE](NOTICE).
