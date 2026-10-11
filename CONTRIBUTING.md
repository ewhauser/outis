# Contributing

Thanks for your interest in improving Outis.

## Ground rules

- **No real PHI, ever.** Issues, pull requests, test fixtures and example text
  must be synthetic. If you find a missed-detection pattern in real data,
  rewrite it with invented values before sharing it.
- **No external data in training.** Public benchmarks are for evaluation only.
  See the rules in [external verification](docs/external-verification.md).
- **Report results honestly.** Name the bank behind every number. Do not
  merge development and qualification numbers.
- **Privacy first.** The [primary measure](docs/evaluation.md) is that
  identifiers stay masked. A wrong type on a masked span is an engineering cost, not a leak.

## Development setup

```sh
uv sync --group dev
```

## Checks

CI runs lint and the full test suite on every pull request. Tests run on Linux
with CPU PyTorch across Python 3.13–3.15. Run the checks locally first:

```sh
uv run ruff check .
uv run pytest -q
```

The unit tests do not need the base encoder or GPU. Tests that verify frozen
data artifacts are skipped when the corpora under `data/` are missing.

## Data in the repository

Synthetic corpora under `data/` are gitignored (`data/**/*.jsonl`) and are not
in this repository; only their small `*.manifest.json` files are. Never commit
corpora, sealed or blind qualification banks, or model weights.

## Code layout

```
src/outis/
  service.py, api.py, cli.py   Deidentifier facade, HTTP API and CLI
  policy.py, entities.py, types.py, analysis.py, operations.py
  detection/                   DeBERTa-v3 token classifier and BIOES decoder
  replacement/                 surrogates, nickname identities, transforms
    names/                     name and record witnesses for identity linking
  formats/                     structured data and native files
  data/                        annotation policy, corpus checks, generators
    generators/  v12/  v13/
  evaluation/                  metrics and HTML reports
```

Runtime code (`detection`, `replacement`, `formats` and the top-level
modules) must not import from `data`, `training` or `evaluation`.

`scripts/takeover/` holds the training script, the evaluation script and the
synthetic bank generators. `benchmarks/` holds
the public-benchmark harness and its aggregate results.

## Pull requests

- Keep each PR focused, and describe the behaviour change and how you
  verified it.
- Add or update tests with synthetic examples for detector, rule or
  replacement changes.
- If a change alters detection output, include before-and-after metrics on a
  development bank from authors the training data never saw, and say which
  bank. Include the `privacy` block figures: any-label recall for direct and
  indirect identifiers, any-label precision and exact typed-span F1. Check the
  change against the [comparison rules](docs/evaluation.md#comparing-a-new-model-with-the-current-one). Never tune
  on a sealed qualification bank.
- Add user-visible changes under `Unreleased` in [CHANGELOG.md](CHANGELOG.md).

## Training and evaluation work

See [training](docs/training.md) and [evaluation methodology](docs/evaluation.md).
Put new experiments in new, versioned output directories.
