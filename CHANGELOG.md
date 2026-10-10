# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
