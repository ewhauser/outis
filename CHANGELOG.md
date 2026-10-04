# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
