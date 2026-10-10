---
title: Local clinical PHI de-identification
description: Local PHI detection, typed redaction and patient-consistent synthetic replacement for English clinical text.
tableOfContents: false
editUrl: false
hero:
  title: Find and remove PHI in clinical text, on your own machine
  tagline: A fine-tuned DeBERTa-v3-base detector for 14 PHI types, with typed redaction and patient-consistent synthetic replacement. It ships as a Python library, a CLI and a local HTTP API.
  actions:
    - text: Quickstart
      link: ./docs/quickstart/
      icon: right-arrow
    - text: See the results
      link: ./docs/results/
      variant: secondary
---

:::caution[Research software]
Outis has **not** met all of its [evaluation targets](./docs/evaluation/), and it
has been evaluated only on synthetic and public benchmark text, never on real
clinical records. It does not by itself provide HIPAA Safe Harbor or Expert
Determination de-identification. Read the [limitations](./docs/limitations/)
before using it.
:::

Outis is Greek for "nobody": the name Odysseus gave the Cyclops.

## What it does

| | |
| --- | --- |
| **14 PHI types** | NAME, DATE, AGE, ADDRESS, LOCATION, FACILITY, PHONE, EMAIL, URL, IP, MRN, SSN, ACCOUNT and ID. |
| **Redaction or replacement** | Typed markers or masks, or fictional values that stay consistent per patient. Dates shift by a stable per-patient offset that preserves intervals. See [synthetic replacement](./docs/replacement/). |
| **Model-only detection** | One fine-tuned encoder, decoded to exact original character offsets. There is no rule layer. See the [model card](./docs/model-card/). |
| **Configurable policies** | Per-type actions, custom identifier rules, linked messages, explicit restoration, and JSON, CSV, XML and plain-text file input. See the [API guide](./docs/api/). |
| **Local by design** | No text is sent to a hosted service. The server is loopback-only by default, with an optional API key, no request logging and `Cache-Control: no-store`. See [deployment](./docs/deployment/). |

## How it works

```
text ─▶ DeBERTa-v3-base token classifier (overlapping 512-token windows)
          └─▶ constrained BIOES decoder ─▶ spans at original offsets ─▶ redact / replace
```

The default model was fine-tuned on about 116,000 synthetic notes. No real
clinical data was used for training. The weights are on Hugging Face as
[ewhauser/outis-deberta-v3-base](https://huggingface.co/ewhauser/outis-deberta-v3-base),
and the training corpora as
[ewhauser/outis-synthetic-phi](https://huggingface.co/datasets/ewhauser/outis-synthetic-phi).

## How it is measured

Outis is scored against [MedDeID](https://huggingface.co/stighellemans/meddeid-english-synth)
and [GLiNER-PII](https://huggingface.co/nvidia/gliner-PII) on three public
benchmarks under one fixed protocol, and on synthetic notes from an author whose
writing was never used in training. The [results](./docs/results/) page has the
numbers; [evaluation methodology](./docs/evaluation/) and
[external verification](./docs/external-verification/) explain how they are computed.
