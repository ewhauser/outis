# Contributing

Thanks for your interest in improving Clinical Deidentifier.

## Ground rules

- **No real PHI, ever.** Issues, pull requests, test fixtures and example text
  must be synthetic. If you find a missed-detection pattern in real data,
  rewrite it with invented values before sharing it.
- **No external data in training.** Public benchmarks are for evaluation only.
  See the rules in [external verification](docs/external-verification.md).
- **Report results honestly.** Report model-only and hybrid results
  separately. Do not merge development and qualification numbers.

## Development setup

```sh
uv venv --python 3.13
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python scripts/bootstrap.py   # only needed for model inference
```

## Checks

CI runs both checks below on every pull request. Run them locally first:

```sh
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
```

The unit tests do not need the base encoder or GPU. Tests that verify frozen
data artifacts are skipped when the corpora under `data/` are missing.

## Data in the repository

Synthetic corpora under `data/` are gitignored, but `main` currently tracks
them until a planned history rewrite (about 1.8 GB, so clones are large).
Before committing any data, read [CLEANUP.md](CLEANUP.md). Add new corpora by
explicit path, list them in the cleanup manifests, and never commit sealed or
blind qualification banks. `scripts/check_data_tracking.py` enforces this in
CI.

## Code layout

```
src/clinical_deid/
  service.py, api.py, cli.py   Deidentifier facade, HTTP API and CLI
  policy.py, entities.py, types.py, analysis.py, operations.py
  detection/                   adapter model, hf_encoder, structural rules
    context/                   context guards (names, dates, records, places, …)
  replacement/                 surrogates, nickname identities, transforms
  formats/                     structured data and native files
  data/                        annotation policy, corpus checks, generators
    generators/  v12/  v13/
  training/                    adapter training
  evaluation/                  metrics and HTML reports
```

Runtime code (`detection`, `replacement`, `formats` and the top-level
modules) must not import from `data`, `training` or `evaluation`.

Scripts under `reports/` and `experiments/` are frozen records of past runs.
They import the module layout that existed when they ran, so run them from
the commit that produced them.

## Pull requests

- Keep each PR focused, and describe the behaviour change and how you
  verified it.
- Add or update tests with synthetic examples for detector, rule or
  replacement changes.
- If a change alters detection output, include before-and-after metrics on a
  development bank, and say which bank. Never tune on a sealed qualification
  bank.
- Add user-visible changes under `Unreleased` in [CHANGELOG.md](CHANGELOG.md).
## Training and evaluation work

See [training](docs/training.md) and the
[evaluation protocol](reports/evaluation_protocol.md). Put new experiments in
new, versioned output directories. Never overwrite a historical protocol or
report.
