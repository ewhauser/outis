# HTML evaluation reports

The report builder converts saved evaluation JSON into a self-contained HTML
page. It uses the existing Python environment, with no frontend build, network
requests, model loading, or new runtime dependencies. The page works offline,
and the same config and input bytes produce identical HTML.

```sh
.venv/bin/python scripts/build_report.py \
  --config reports/pipeline_report_config.json \
  --output reports/pipeline_report.html
```

The module form is also supported:

```sh
.venv/bin/python -m clinical_deid.evaluation.report \
  --config reports/pipeline_report_config.json \
  --output reports/pipeline_report.html
```

## Reading benchmark progress

The page opens on the configured `default_run` and leads with its benchmark
verdict and the count of required checks met. This is a checklist count, not a
percentage of deployment readiness. Selecting a run updates the entire summary,
target table, label breakdown, and supporting metrics together.

The target table shows each observed value, its declared target, and the
shortfall. Accuracy shortfalls use percentage points; support shortfalls use
spans. The all-label gates show the weakest recall and smallest label sample.
Per-label rows are ordered by recall, with unassessed labels last. Tiny nonzero
gaps remain visible as `<0.01 pp`; a failing result that would round up to its
target is displayed as `<99%` rather than appearing to meet it.

Notes with uncovered annotated PHI and negative notes left untouched remain
prominent diagnostics. Neither has an acceptance threshold in the current
protocol, so the page does not invent pass/fail gates for them. Detection
coverage is measured before replacement and does not validate final output.
Development and historical qualification are explicitly distinguished;
comparison and timing appear after benchmark progress. The view also identifies
the remaining clinical-validation scope and the absence of a per-family
breakdown.

For a local HTTP preview, serve only the output directory you intend to expose:

```sh
.venv/bin/python scripts/build_report.py \
  --config reports/pipeline_report_config.json \
  --output /tmp/deid-report-preview/index.html
.venv/bin/python -m http.server 4173 --bind 127.0.0.1 \
  --directory /tmp/deid-report-preview
```

## Selecting evidence

Input paths are relative to the config file, not the working directory. Use an
explicit run list; nothing scans for the newest checkpoint or reruns a sealed
qualification. The supplied config preserves the v7 comparison from October 2,
2026. The blind v5 corpus was independent at that run and was subsequently
retired into development, so it is marked `historical_qualification`.

A minimal config for an ordinary evaluation report:

```json
{
  "schema_version": 1,
  "title": "Clinical pipeline evaluation",
  "description": "Saved development evaluation for the next candidate.",
  "runs": [
    {
      "id": "development",
      "label": "Development · hybrid",
      "purpose": "development",
      "path": "next_development_hybrid.json",
      "backend": "hybrid"
    }
  ]
}
```

Optional config fields:

| Field | Meaning |
|---|---|
| `candidate` | Path to a schema v1 selection JSON with `selected_model_dir` and `adapter_sha256`. Every run must match the selected adapter and, when recorded, its model config hash. Omit for comparisons between different adapters. |
| `default_run` | Run ID initially shown in the detailed metrics view. Defaults to the first run. |
| `notes` | Array of report-level explanatory strings. These strings are included verbatim as escaped text; use only aggregate context. |
| Run `note` | Explanation of that run's scope or timing limitations. |
| Run `sha256` | Optional SHA-256 of the source JSON bytes. Generation fails if the source changes. |

Each run needs a unique `id`, a `label`, and a `path`. Supported purposes are
`development`, `diagnostic`, `qualification`, and `historical_qualification`.
The last two require metadata establishing that the original evaluation corpus
was independent and unused for training or checkpoint selection. This checks
the saved metadata; it does not prove independence or current retirement status.

Both schema v1 evaluator outputs and qualification wrappers are supported. For
a wrapper, set `backend` to `model` or `hybrid` to select its embedded
`reports.model` or `reports.hybrid` aggregate. For an ordinary report, `backend`
is optional; if supplied, it must match. Add separate entries to compare both.
The wrapper's qualification decision belongs to the hybrid pipeline.

To refresh, update the config to the intended saved reports and rerun the same
command. Review labels, purpose, and notes when changing datasets or models.
The page shows saved evidence, not live service status. Generating it never
establishes that the running API uses the selected checkpoint.

## Metric interpretation

- **Exact span precision, recall, and F1:** both boundaries and entity type must
  match. Rates are checked against true-positive, false-positive, and
  false-negative counts.
- **PHI character recall:** fraction of gold PHI characters covered by predicted
  spans, regardless of predicted entity type.
- **Leak-free documents:** every gold PHI character is covered. The denominator
  includes documents with no PHI; those count as leak-free even if falsely
  redacted. This is not an exact-match document score.
- **Negative documents untouched:** no prediction on a document without gold
  PHI. This uses its own denominator, and is unavailable when there are no
  negative documents. Other empty-denominator rates are also displayed as N/A.
- **Goals:** the evaluator's recorded goal checks, separate from production
  readiness. Missing checks are unavailable, not passing.
- **Recorded detection components:** neural inference plus rules where both
  are present. A cache replay may combine timings from different runs. Missing
  timing stays unavailable; `elapsed_seconds` is never used as a substitute
  because it can represent scoring only.
- **Recorded documents/second:** documents divided by the component sum; an
  offline, amortized value. Corpus length and composition affect it. These
  timings exclude model loading, HTTP, queueing, replacement, and report I/O,
  and cannot establish request p50/p95 or a controlled CPU/GPU speedup.

Model-only and hybrid results in a paired evaluation share one neural pass;
their times must not be added together. Separate corpora are never aggregated
into a combined accuracy or throughput figure.

## Evidence and validation

The page contains only allowlisted aggregate metrics, model/corpus/source
hashes, source basenames, and config-authored labels and notes. Failure examples,
document text, predictions, full corpus manifests, and absolute source paths
are excluded. Do not put sensitive text in config labels, notes, or filenames.
The builder does not follow file pointers embedded inside evaluation JSON.

It validates schema versions, finite values, count/rate consistency, per-label
totals, goal consistency, and candidate identity before writing. It also
rechecks the inputs after rendering. Output must end in `.html`; input aliases
and output symlinks are rejected. Replacement is atomic, so invalid evidence
leaves an existing page intact. Existing reports without optional provenance,
timing, or goals display unavailable information explicitly.

Source hashes identify the JSON bytes read at build time. They do not attest
that the underlying evaluation was correct. Detector/rule manifest fingerprints
are SHA-256 of sorted compact JSON of the corresponding source-hash mappings.
Reported completion time comes only from the qualification wrapper; regeneration
does not invent a new evaluation timestamp.

```sh
.venv/bin/python -m pytest -q tests/test_report.py tests/test_report_progress.py
.venv/bin/ruff check src/clinical_deid/evaluation/report.py src/clinical_deid/evaluation/report_html.py \
  src/clinical_deid/evaluation/report_progress.py scripts/build_report.py \
  tests/test_report.py tests/test_report_progress.py
```
