# External benchmarks

This harness scores the project's detector on public de-identification
benchmarks that the project did not author, and compares it with other open
models under one fixed protocol. Nothing here is used for training, tuning or
checkpoint selection. The process, rules, file formats and how to add a
benchmark or a system are in
[external verification](../docs/external-verification.md).

**Results: [results/COMPARISON.md](results/COMPARISON.md).** One table per
benchmark, with the policy-aligned view first and each benchmark's
as-published scoring beside it.

| Benchmark | Content | Licence | Status |
|---|---|---|---|
| [ASQ-PHI](https://doi.org/10.17632/csz5dzp7nx.1) | 1,051 synthetic clinical search queries | MIT | Development evidence |
| [MedDeID English](https://huggingface.co/datasets/stighellemans/meddeid-english-synthetic-benchmark) | 300 synthetic notes | CC BY 4.0, final evaluation only | Development evidence |
| [Technetium-I](https://huggingface.co/datasets/temlm-foundation/Technetium-I) | fixed 5,000-note sample of the test split | EUPL-1.2 | Uninspected |
| [PhysioNet Deidentified Medical Text](https://physionet.org/content/deidentifiedmedicaltext/1.0/) v1.0 | 2,434 real nursing notes with surrogate PHI | PhysioNet Credentialed Health Data License 1.5.0, research only | Restricted; uninspected |

"Development evidence" means failures on that benchmark have been inspected,
so it is no longer an independent test.

| System | Model |
|---|---|
| `project` | this project's default model |
| `meddeid` | `stighellemans/meddeid-english-synth` (MedDeID 0.4.2, `en-US`) |
| `gliner-trained` | `nvidia/gliner-PII` with its model-card configuration |
| `gliner-facility` | the same model with one added zero-shot facility label |

External models run locally in their own uv environments
(`uv run --no-project --with ...`), from pinned revisions; their weights are
never vendored.

## Reproducing

```sh
cd benchmarks
uv run asq_phi.py
uv run predict_project.py --benchmark asq_phi
uv run --no-project --python 3.13 --with meddeid==0.4.2 predict_meddeid.py --benchmark asq_phi
uv run --no-project --python 3.13 --with gliner==0.2.29 predict_gliner.py --benchmark asq_phi --arm trained
uv run --no-project --python 3.13 --with gliner==0.2.29 predict_gliner.py --benchmark asq_phi --arm facility
uv run score.py --benchmark asq_phi --systems project meddeid gliner-trained gliner-facility
uv run score_aligned.py --benchmark asq_phi --systems project meddeid gliner-trained gliner-facility
uv run compare.py
uv run pytest -q test_external.py
```

`synthetic_benchmarks.py --benchmark meddeid_en` and
`--benchmark technetium_test5k` convert the other two benchmarks. Downloads,
converted corpora and raw predictions are gitignored. Each prediction file has
a receipt binding it to the corpus hash, model revision and runner hash, and
the scorers refuse predictions whose receipt does not match.

## Caveats

- ASQ-PHI, MedDeID English and Technetium-I are synthetic; ASQ-PHI is short
  search queries, not notes. PhysioNet Deidentified Medical Text is real
  nursing notes with surrogate PHI, under a research-only licence; it is run
  locally and only aggregate scores are kept.
- Benchmarks differ from this project's annotation policy (for example
  titles inside names, unannotated ages and years). The aligned view
  neutralises those differences; the as-published columns do not.
- Scores are single runs.
