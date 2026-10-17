# Changelog

All notable changes to this project are documented here. The project uses
[Semantic Versioning](https://semver.org/). Release Please writes each entry
from the Conventional Commit messages merged since the previous release; see
[RELEASING.md](RELEASING.md).

## [0.5.0](https://github.com/ewhauser/outis/compare/v0.4.0...v0.5.0) (2026-10-17)


### Features

* score PhysioNet Deidentified Medical Text as a restricted benchmark ([#37](https://github.com/ewhauser/outis/issues/37)) ([548ba60](https://github.com/ewhauser/outis/commit/548ba6058f642ecd03860325a18966a1737ce392))


### Documentation

* report the real-notes PhysioNet results in the README and model card ([#39](https://github.com/ewhauser/outis/issues/39)) ([0ffdf2c](https://github.com/ewhauser/outis/commit/0ffdf2c049953494df5fe22ee2d60ae2fa3356a0))
* state how Outis compares with the open models evaluated ([#40](https://github.com/ewhauser/outis/issues/40)) ([0d32c15](https://github.com/ewhauser/outis/commit/0d32c15c658c9f385bec2c2e7d82f3ec592fc4e4))

## [0.4.0](https://github.com/ewhauser/outis/compare/v0.3.0...v0.4.0) (2026-10-13)


### Features

* add corpus-scale outis batch command ([#34](https://github.com/ewhauser/outis/issues/34)) ([f04f670](https://github.com/ewhauser/outis/commit/f04f670925e6695fd247bedddc6f4863a218956a))
* add opt-in realistic synthetic replacement style ([#32](https://github.com/ewhauser/outis/issues/32)) ([6d5438c](https://github.com/ewhauser/outis/commit/6d5438cfb3f39fbc392d1b322157571d2c97c93e))
* add opt-in server-level site dictionaries ([#29](https://github.com/ewhauser/outis/issues/29)) ([2797e9b](https://github.com/ewhauser/outis/commit/2797e9b5c6c27c14f4e5afde423d2ba1e8c1467f))
* add opt-in study replacement scope ([#30](https://github.com/ewhauser/outis/issues/30)) ([765864d](https://github.com/ewhauser/outis/commit/765864d610f56c54c9b21c73a29e6e962be65719))
* detect once and replace later with outis batch ([#36](https://github.com/ewhauser/outis/issues/36)) ([b5d4459](https://github.com/ewhauser/outis/commit/b5d4459430b1a96df272987195ea40f87f8311b9))
* key opt-in replacement scope by project_id instead of study_id ([#33](https://github.com/ewhauser/outis/issues/33)) ([383f4be](https://github.com/ewhauser/outis/commit/383f4be697704a565dad0e495402412d71941968))
* publish CPU and GPU Docker images to GHCR ([#27](https://github.com/ewhauser/outis/issues/27)) ([38eeabd](https://github.com/ewhauser/outis/commit/38eeabd982cbbe360a506e71d958db076f9a998c))
* replace caller-supplied spans and review them before release ([#35](https://github.com/ewhauser/outis/issues/35)) ([6c6eaee](https://github.com/ewhauser/outis/commit/6c6eaee4148cea72afcc3ac43a13518949fc5044))


### Bug Fixes

* two P1 issues from site dictionaries ([#29](https://github.com/ewhauser/outis/issues/29)) and study scope ([#30](https://github.com/ewhauser/outis/issues/30)) ([#31](https://github.com/ewhauser/outis/issues/31)) ([87a94e3](https://github.com/ewhauser/outis/commit/87a94e30914b0a7dca0209bf8e9643594cbf8ad0))

## [0.3.0](https://github.com/ewhauser/outis/compare/v0.2.1...v0.3.0) (2026-10-12)


### Features

* default to bf16 inference on bf16-capable GPUs and MPS ([#26](https://github.com/ewhauser/outis/issues/26)) ([09247a5](https://github.com/ewhauser/outis/commit/09247a54e3a6f0496a27f48f630fcf9d84bc0639))


### Bug Fixes

* close Office package leaks and harden file and decoder boundaries ([#10](https://github.com/ewhauser/outis/issues/10)) ([18a5d32](https://github.com/ewhauser/outis/commit/18a5d32569cdabe93142b2dd8fe1beecc97dfb97))
* keep health checks independent of inference capacity ([#12](https://github.com/ewhauser/outis/issues/12)) ([863d030](https://github.com/ewhauser/outis/commit/863d03003cbc70c9d5e585cad394f3614d63ada8))
* render every replacement through one implementation ([#14](https://github.com/ewhauser/outis/issues/14)) ([b7bb5d4](https://github.com/ewhauser/outis/commit/b7bb5d41b3023e63e9b6c4a680a571198a83ae41))
* report malformed detector output as a server failure, not 422 ([#17](https://github.com/ewhauser/outis/issues/17)) ([387cf0f](https://github.com/ewhauser/outis/commit/387cf0f268133c111796385c914fe2086919f051))


### Performance Improvements

* analyze request identities once and reuse them through rendering ([#15](https://github.com/ewhauser/outis/issues/15)) ([75213e9](https://github.com/ewhauser/outis/commit/75213e9e4d67bba0212cab5d06e159864441ef48))
* batch encoder windows by length instead of arrival order ([#24](https://github.com/ewhauser/outis/issues/24)) ([c931335](https://github.com/ewhauser/outis/commit/c93133587cfab18b26d84fa4ad48fd70f09e84e7))
* build BIOES transition constraints once per process ([#19](https://github.com/ewhauser/outis/issues/19)) ([f8f18f5](https://github.com/ewhauser/outis/commit/f8f18f527d973f54f85b4753c463381d4b9396e1))
* count XLSX shared strings once per workbook ([#18](https://github.com/ewhauser/outis/issues/18)) ([1dcd4a3](https://github.com/ewhauser/outis/commit/1dcd4a3c75bcb0e482a82f90210933ac39f8b93f))

## [Unreleased]

### Added

- Opt-in `replacement.style: "realistic"` for synthetic replacement (also
  `style=` on `surrogates.replace` and `outis text --style realistic`).
  Identifiers keep their exact format through a keyed permutation; SSNs,
  phones, emails, URLs and IPs stay inside never-issued or reserved ranges; and
  places, addresses and facilities get fictional names. `keep_region` (default
  `true`) keeps a place's state and country. The default `obvious` style is
  unchanged.

### Changed

- Require Python 3.13 or later; drop Python 3.11 and 3.12 support and test
  Python 3.13, 3.14, and 3.15 in CI.
- Run the token encoder with bf16 autocast by default on CUDA GPUs that support
  bf16 and on Apple MPS; CPU and older GPUs stay fp32. Pass `precision="fp32"`
  to `load_detector` (or `--precision fp32` to `eval_bank.py`) for the previous
  behaviour.

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
