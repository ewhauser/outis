# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Removed

- The tracked Privacy-Filter adapter checkpoints (`models/clinical-adapter*`)
  and the `models/privacy-filter-base` provenance file. They remain in git
  history. The external benchmark harness now runs the default checkpoint as
  system `project-mix5`.
- **Breaking:** the rule layer. Detection is now model-only: the regex
  extractors and context guards (`clinical_deid.detection.rules`,
  `clinical_deid.detection.context`), the `backend` option of the service,
  detector and evaluator, the `DEID_BASE_DIR` setting, and per-label
  `model_labels` source selection are gone. On held-out authors the rules
  masked no more direct identifiers than the model and cost precision, indirect
  recall and about 2.4 points of exact typed F1.
- **Breaking:** the OpenAI Privacy Filter adapter detector
  (`clinical_deid.detection.adapter`), its trainer (`clinical_deid.training`),
  `scripts/bootstrap.py`, and the adapter-era qualification, replay and
  scoring scripts. Check out `cbaf3af` to reproduce historical adapter results.
- `opf` is no longer a runtime dependency; it moved to the optional
  `generators` extra used by two legacy corpus generators.

### Changed

- DeBERTa-v2/v3 encoders use a gather-free relative-attention bias
  (`clinical_deid.detection.deberta_attention`) in training and inference.
  Outputs and gradients match the stock transformers code to float32
  rounding; on Apple MPS a training step is about 1.8 times faster. It is
  applied only when the stock code matches the checked transformers version.
  Other encoders are unchanged.
- `scripts/takeover/train_hf.py` loads base models in fp32.
- **Breaking:** the default model is ModernBERT-base mix5 epoch 1,
  `models/takeover-mbbase-mix5-001/step-003803`, for the service, API, CLI and
  local launcher. Its weights are not stored in git.
- Name and record parsing used to link nickname identities during replacement
  moved to `clinical_deid.replacement.names`; it never adds detections.
- `scripts/takeover/eval_bank.py` reports one model-only result including the
  `privacy` block; `scripts/takeover/train_hf.py` selects epochs on model-only
  scores.

### Added

- [GOALS.md](GOALS.md): the accepted project goal, measured as any-label
  recall by identifier class, with regression rules for comparing candidates.
- `privacy` block in every evaluation report: any-label recall for direct and
  indirect identifiers, a document-clustered lower bound on NAME recall,
  any-label precision, and a per-type breakdown of exact misses.
- [AGENTS.md](AGENTS.md) with standing instructions for coding agents.
- Model card, citation file, contributing guide, security policy and CI.
- Focused guides for [training](docs/training.md),
  [evaluation history](docs/evaluation-history.md),
  [synthetic replacement](docs/replacement.md) and
  [deployment](docs/deployment.md).

### Changed

- **Breaking:** `clinical_deid` is split into subpackages (`detection`,
  `replacement`, `formats`, `data`, `training`, `evaluation`). For example,
  `clinical_deid.rules` is now `clinical_deid.detection.rules` and
  `python -m clinical_deid.train` is now `python -m clinical_deid.training.train`.
  `from clinical_deid import Deidentifier`, the HTTP API and the CLI are
  unchanged. Scripts under `reports/` use the old layout; run them from the
  commit that produced them.
- **Breaking:** PDF support moved to the optional `pdf` extra
  (`pip install clinical-deid[pdf]`).
- The annotation policy now states that it is the only policy, includes the
  conventions settled by the convention census, and lists which corpora
  predate it. The duplicate convention summary in the evaluation protocol was
  replaced by a link.
- Promotion is now gated by the privacy targets in GOALS.md. The 99% exact
  typed-span checks are unchanged and reported as a stretch target.
- Rewrote the README around installation, usage, results and limitations.
  The detailed experiment history moved to `docs/`.
- Formatted the codebase with `ruff format` and enabled stricter lint rules.

## [0.1.0] - 2026-10-04

The initial development release. The default checkpoint is the fourth-stage
(v4) adapter.

### Added

- Hybrid detector: a residual clinical adapter and BIOES head on the frozen
  OpenAI Privacy Filter encoder, plus structural extractors and clinical
  context guards for 14 PHI types.
- Redaction and patient-consistent synthetic replacement, with per-patient
  date shifting and name and nickname identity linking.
- FastAPI server, CLI and Python API with policies, masks, linked messages,
  restoration mappings, structured data and native text file input.
- Synthetic corpus generators, training with cached backbone features,
  qualification and development evaluation, and HTML reports.
- External benchmark harness with an ASQ-PHI comparison against MedDeID and
  GLiNER-PII.

### Fixed

- Abbreviated care-site names ("St. Francis Hospital") and "at <name>"
  facilities are kept whole.
- NAME boundaries around initials, possessives and quotes.

[Unreleased]: https://github.com/ewhauser/deidentifier/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ewhauser/deidentifier/releases/tag/v0.1.0
