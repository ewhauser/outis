# Outis

[![CI](https://github.com/ewhauser/outis/actions/workflows/ci.yml/badge.svg)](https://github.com/ewhauser/outis/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](pyproject.toml)

Local PHI detection, typed redaction and patient-consistent synthetic
replacement for English clinical text.

Outis is Greek for "nobody": the name Odysseus gave the Cyclops.

The detector is a fully fine-tuned
[DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) token
classifier with a constrained BIOES decoder. There is no rule layer: every
detected span comes from the model. Everything runs on your machine; no text
is sent to a hosted service. It ships as a Python library, a CLI and a FastAPI
server.

- **Model:** [ewhauser/outis-deberta-v3-base](https://huggingface.co/ewhauser/outis-deberta-v3-base) on Hugging Face
- **Training data:** [ewhauser/outis-synthetic-phi](https://huggingface.co/datasets/ewhauser/outis-synthetic-phi), the synthetic
  corpora the model was trained and selected on

> [!IMPORTANT]
> This is research software. It has **not** met all of its [evaluation targets](docs/evaluation.md), and it has been evaluated only on
> synthetic and public benchmark text, never on real clinical records. It does
> not by itself provide HIPAA Safe Harbor or Expert Determination
> de-identification. See [Limitations](#limitations).

## Features

- **14 PHI types**: NAME, DATE, AGE, ADDRESS, LOCATION, FACILITY, PHONE, EMAIL,
  URL, IP, MRN, SSN, ACCOUNT and ID.
- **Redaction or replacement**: typed markers or masks, or fictional values that
  stay consistent per patient, or opt-in across a whole project release. Dates
  shift by a stable per-patient offset that preserves intervals.
- **Model-only detection**: one fine-tuned encoder, decoded to exact original
  character offsets.
- **Configurable policies**: per-type actions, custom identifier rules, linked
  messages, explicit restoration, and JSON, CSV, XML and plain-text file input.
- **Site dictionaries** (opt-in): load a site's own facility names, departments
  and identifier formats once at startup and match them in a single pass. They
  are site-supplied, and their effect on accuracy has not been measured.
- **Local by design**: loopback-only by default, optional API key, no request
  logging, `Cache-Control: no-store` on responses.

## Quickstart

Requires Python 3.13+.

```sh
pip install outis
```

The first run downloads the model (736 MB) from
[Hugging Face](https://huggingface.co/ewhauser/outis-deberta-v3-base) into the
Hugging Face cache, and later runs load it from there. Each release pins the
model revision it was tested with. Only the model is downloaded; your text
never leaves the machine. Once the model is cached, `HF_HUB_OFFLINE=1` forbids
network access. To use a checkpoint directory instead, pass it as
`Deidentifier(model_dir)`, `--model-dir` or `OUTIS_MODEL_DIR`.

Add the `pdf` extra (`pip install 'outis[pdf]'`) for PDF file input.

To work from source, use [uv](https://docs.astral.sh/uv/) and prefix the
commands below with `uv run`:

```sh
git clone https://github.com/ewhauser/outis.git
cd outis
uv sync
```

A checkout uses `models/deid-deberta-v3-base` when its weight file is present,
and otherwise downloads the model as above. To keep the weights in the checkout:

```sh
uv run hf download ewhauser/outis-deberta-v3-base hf/model.safetensors --local-dir models/deid-deberta-v3-base
```

### Python

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
outis serve --port 8000
```

```sh
curl http://127.0.0.1:8000/v1/deidentify \
  -H 'Content-Type: application/json' \
  -d '{"text":"Patient Mara Calder, DOB 1974-06-13. Creatinine 1.2 mg/dL.","mode":"redact"}'
```

Interactive docs are served at `http://127.0.0.1:8000/docs`, and the OpenAPI
schema at `/openapi.json`. Synthetic mode needs a replacement secret in
`OUTIS_SECRET`; in a checkout, `uv run scripts/serve_local.py --port 8000`
creates and reuses one in `.cache/local-api-secret`. For other deployments, see
[deployment](docs/deployment.md).

### Docker

Images with the model built in are published to the GitHub Container Registry
for CPU (`linux/amd64`, `linux/arm64`) and NVIDIA GPUs (`linux/amd64`). They
run without network access.

```sh
docker run --rm -p 8000:8000 -e OUTIS_API_KEY=change-me ghcr.io/ewhauser/outis:cpu
```

```sh
docker run --rm --gpus all -p 8000:8000 -e OUTIS_API_KEY=change-me ghcr.io/ewhauser/outis:gpu
```

Requests then send the key in an `X-API-Key` header. See
[deployment](docs/deployment.md#docker) for tags and settings.

### CLI

```sh
outis text --input note.txt --mode redact
```

`outis batch` de-identifies a whole note table, JSONL or Parquet (with the
`outis[parquet]` extra), in one process. Every row needs a unique note ID;
synthetic mode also needs a patient ID column and `OUTIS_SECRET`. Output is
written as it goes. With `--resume`, the same command starts the run or, after
an interruption, continues it from the last completed batch:

```sh
outis batch --input notes.parquet --output deid.parquet \
  --id-column note_id --text-column note_text \
  --patient-column patient_id --mode synthetic --resume
```

See [batch processing](docs/deployment.md#batch-processing) for the output
format, resume guarantee, out-of-memory handling and measured throughput.

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
| [Synthetic replacement](docs/replacement.md) | Secrets, patient anchors, date shifting, identity linking, project scope |
| [Deployment](docs/deployment.md) | Launcher, default model, corpus batch runs and throughput, security controls |
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
