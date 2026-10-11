# Changelog

All notable changes to this project are documented here. The project uses
[Semantic Versioning](https://semver.org/). Release Please writes each entry
from the Conventional Commit messages merged since the previous release; see
[RELEASING.md](RELEASING.md).

## [0.2.1](https://github.com/ewhauser/outis/compare/v0.2.0...v0.2.1) (2026-10-11)


### Bug Fixes

* close identifier leaks in policy overlaps and Office extraction ([#8](https://github.com/ewhauser/outis/issues/8)) ([0b04467](https://github.com/ewhauser/outis/commit/0b0446724ebeee656628f6b49370818f5aeafab9))

## [0.2.0](https://github.com/ewhauser/outis/compare/v0.1.0...v0.2.0) (2026-10-11)


### Features

* download the pinned default model from Hugging Face when none is local ([#4](https://github.com/ewhauser/outis/issues/4)) ([6f60245](https://github.com/ewhauser/outis/commit/6f602454f1a9e232e246a03d97dc9adc7295e44b))

## [0.1.0]

First public release.

- Package, CLI and import name `outis`; configuration through `OUTIS_API_KEY`,
  `OUTIS_SECRET`, `OUTIS_MODEL_DIR`, `OUTIS_DEVICE`, `OUTIS_RESTRICTED_DIR` and
  `OUTIS_PLACE_CACHE`.

- DeBERTa-v3-base token classifier with a constrained BIOES decoder over 14 PHI types; detection is
  model-only.
- Redaction, masking and patient-consistent synthetic replacement with
  interval-preserving date shifting.
- Python library, CLI and local FastAPI server with structured-data and file
  input.
- Training script, synthetic bank generators and a public-benchmark scoring
  harness.
