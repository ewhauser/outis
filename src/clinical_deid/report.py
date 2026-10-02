"""Build an aggregate-only, offline HTML report from saved evaluation JSON.

This module never loads a detector, a corpus, or a path found inside an
evaluation report. Only files explicitly named in the report config are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .report_html import render_report


LABELS = frozenset(
    (
        "NAME",
        "DATE",
        "ADDRESS",
        "PHONE",
        "EMAIL",
        "MRN",
        "SSN",
        "ACCOUNT",
        "ID",
        "URL",
        "IP",
        "AGE",
        "LOCATION",
        "FACILITY",
    )
)
CHECK_LABELS = {
    "typed_span_precision_at_least_99_percent": "Exact span precision ≥ 99%",
    "typed_span_recall_at_least_99_percent": "Exact span recall ≥ 99%",
    "typed_span_f1_at_least_99_percent": "Exact span F1 ≥ 99%",
    "phi_character_recall_at_least_99_percent": "PHI character recall ≥ 99%",
    "recall_wilson_lower_95_at_least_99_percent": "Recall 95% Wilson lower bound ≥ 99%",
    "all_labels_have_sufficient_support": "All labels meet minimum support",
    "every_label_recall_at_least_99_percent": "Every label recall ≥ 99%",
}
PURPOSES = {"development", "qualification", "historical_qualification", "diagnostic"}


class ReportError(ValueError):
    """An input cannot be represented faithfully in the report."""


def _object(value: Any, field: str) -> dict:
    if not isinstance(value, dict):
        raise ReportError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportError(f"{field} must be a nonempty string")
    return value


def _count(value: Any, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ReportError(
            f"{field} must be a {'positive' if positive else 'nonnegative'} integer"
        )
    return value


def _number(value: Any, field: str, *, rate: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ReportError(f"{field} must be finite and nonnegative")
    if rate and value > 1:
        raise ReportError(f"{field} must be between 0 and 1")
    return float(value)


def _boolean(value: Any, field: str) -> bool | None:
    if value is not None and type(value) is not bool:
        raise ReportError(f"{field} must be boolean or null")
    return value


def _digest(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ReportError(f"{field} must be a SHA-256 hex digest")
    return value.lower()


def _ratio(numerator: int, denominator: int) -> float:
    # The evaluator uses 1 for an empty denominator; the view displays N/A.
    return numerator / denominator if denominator else 1.0


def _wilson_lower(successes: int, trials: int) -> float:
    if not trials:
        return 0.0
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = proportion + z * z / (2 * trials)
    adjustment = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    )
    return max(0.0, (centre - adjustment) / denominator)


def _check_rate(value: Any, expected: float, field: str) -> float:
    actual = _number(value, field, rate=True)
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-9):
        raise ReportError(f"{field} does not agree with its counts")
    return actual


def _metrics(raw: Any, field: str) -> dict:
    raw = _object(raw, field)
    out = {
        key: _count(raw.get(key), f"{field}.{key}")
        for key in ("true_positives", "false_positives", "false_negatives", "support")
    }
    tp, fp, fn = (
        out[key] for key in ("true_positives", "false_positives", "false_negatives")
    )
    if out["support"] != tp + fn:
        raise ReportError(f"{field}.support does not equal TP + FN")
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    for key, expected in (("precision", precision), ("recall", recall), ("f1", f1)):
        out[key] = _check_rate(raw.get(key), expected, f"{field}.{key}")
    return out


def _hash_manifest(raw: Any, field: str) -> str | None:
    if raw is None:
        return None
    raw = _object(raw, field)
    if not raw:
        return None
    # Hash the entire source manifest, but never embed arbitrary filenames.
    for key, value in raw.items():
        if _digest(value, f"{field} digest") is None:
            raise ReportError(f"{field} contains a null digest")
    return hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read(path: Path) -> tuple[dict, str]:
    payload = path.read_bytes()
    try:
        raw = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        # Do not echo a decoder excerpt; a source may contain note text.
        raise ReportError(f"Invalid JSON in {path.name}") from exc
    raw = _object(raw, path.name)
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ReportError(f"{path.name}: only schema_version 1 is supported")
    return raw, hashlib.sha256(payload).hexdigest()


def _normalize(
    raw: dict, entry: dict, source: Path, digest: str, wrapper: dict | None
) -> dict:
    label = _text(entry.get("label"), "run.label")
    purpose = entry.get("purpose", "diagnostic")
    if purpose not in PURPOSES:
        raise ReportError(f"{label}: unsupported purpose")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ReportError(f"{label}: only evaluation schema_version 1 is supported")
    backend = raw.get("backend")
    if backend not in {"model", "hybrid", "unspecified"}:
        raise ReportError(f"{label}: unsupported backend")
    if entry.get("backend", backend) != backend:
        raise ReportError(f"{label}: requested backend differs from report")
    documents = _count(raw.get("documents"), f"{label}.documents", positive=True)
    typed = _metrics(raw.get("typed_span"), "typed_span")
    character = _metrics(raw.get("character"), "character")
    leak = _object(raw.get("document_leak_free"), "document_leak_free")
    leak_count = _count(leak.get("count"), "document_leak_free.count")
    if leak_count > documents:
        raise ReportError("Leak-free count exceeds documents")
    leak = {
        "count": leak_count,
        "rate": _check_rate(
            leak.get("rate"), leak_count / documents, "document_leak_free.rate"
        ),
    }
    negative = raw.get("negative_documents")
    if negative is not None:
        negative = _object(negative, "negative_documents")
        count = _count(negative.get("count"), "negative_documents.count")
        clean = _count(negative.get("clean_count"), "negative_documents.clean_count")
        if clean > count or count > documents or count > leak_count:
            raise ReportError("Negative-document counts disagree with document counts")
        negative = {
            "count": count,
            "clean_count": clean,
            "clean_rate": _check_rate(
                negative.get("clean_rate"),
                _ratio(clean, count),
                "negative_documents.clean_rate",
            ),
        }
    per_label = []
    for name, stats in _object(raw.get("per_label", {}), "per_label").items():
        if name not in LABELS:
            raise ReportError("Unsupported entity label in per_label")
        stats = _object(stats, "per_label entry")
        metrics = _metrics(stats, f"per_label.{name}")
        per_label.append(
            {
                "label": name,
                **metrics,
                "sufficient_support": _boolean(
                    stats.get("sufficient_support"), "sufficient_support"
                ),
            }
        )
    if per_label:
        for key in ("true_positives", "false_positives", "false_negatives", "support"):
            if sum(item[key] for item in per_label) != typed[key]:
                raise ReportError(f"Per-label {key} does not sum to typed-span count")
    per_label.sort(key=lambda item: (item["recall"], item["label"]))
    goal = raw.get("goal")
    if goal is not None:
        goal = _object(goal, "goal")
        checks = _object(goal.get("checks"), "goal.checks")
        if set(checks) != set(CHECK_LABELS) or any(
            type(value) is not bool for value in checks.values()
        ):
            raise ReportError(
                "Schema v1 requires the complete set of boolean goal checks"
            )
        if {item["label"] for item in per_label} != LABELS:
            raise ReportError("Goal verification requires all entity labels")
        threshold = _number(goal.get("threshold"), "goal.threshold", rate=True)
        if threshold != 0.99:
            raise ReportError("Schema v1 goal threshold must be 0.99")
        minimum = _count(
            goal.get("minimum_label_support"), "minimum_label_support", positive=True
        )
        lower = _check_rate(
            raw.get("recall_wilson_lower_95"),
            _wilson_lower(typed["true_positives"], typed["support"]),
            "recall_wilson_lower_95",
        )
        for stats in per_label:
            if stats["sufficient_support"] != (stats["support"] >= minimum):
                raise ReportError(
                    "Per-label sufficient_support disagrees with minimum support"
                )
        expected = {
            "typed_span_precision_at_least_99_percent": typed["precision"] >= threshold,
            "typed_span_recall_at_least_99_percent": typed["recall"] >= threshold,
            "typed_span_f1_at_least_99_percent": typed["f1"] >= threshold,
            "phi_character_recall_at_least_99_percent": character["recall"]
            >= threshold,
            "recall_wilson_lower_95_at_least_99_percent": lower >= threshold,
            "all_labels_have_sufficient_support": all(
                item["support"] >= minimum for item in per_label
            ),
            "every_label_recall_at_least_99_percent": all(
                item["recall"] >= threshold for item in per_label
            ),
        }
        if checks != expected:
            raise ReportError("Goal checks disagree with evaluated metrics")
        passed = _boolean(goal.get("passed"), "goal.passed")
        if passed is None or passed != all(checks.values()):
            raise ReportError("goal.passed disagrees with goal checks")
        goal = {
            "passed": passed,
            "threshold": threshold,
            "minimum_label_support": minimum,
            "recall_wilson_lower_95": lower,
            "checks": [
                {"name": name, "label": CHECK_LABELS[name], "passed": value}
                for name, value in checks.items()
            ],
        }
    corpus = _object(raw.get("corpus_manifest", {}), "corpus_manifest")
    if "rows" in corpus and _count(corpus["rows"], "corpus_manifest.rows") != documents:
        raise ReportError("Corpus row count does not match evaluated documents")
    flags = {}
    for key in (
        "independent_final",
        "used_for_checkpoint_selection",
        "used_for_training",
    ):
        flags[key] = _boolean(raw.get(key, corpus.get(key)), key)
        if key in raw and key in corpus and raw[key] != corpus[key]:
            raise ReportError(f"Conflicting {key} metadata")
    if purpose in {"qualification", "historical_qualification"} and (
        flags["independent_final"] is not True
        or flags["used_for_training"] is not False
        or flags["used_for_checkpoint_selection"] is not False
    ):
        raise ReportError(
            "Qualification purpose requires independent, unused corpus metadata"
        )
    artifacts = _object(raw.get("model_artifacts", {}), "model_artifacts")
    neural = raw.get("neural_inference_seconds")
    rules = raw.get("rule_application_seconds")
    neural = None if neural is None else _number(neural, "neural_inference_seconds")
    rules = None if rules is None else _number(rules, "rule_application_seconds")
    total = neural + rules if neural is not None and rules is not None else None
    if total is not None and not math.isfinite(total):
        raise ReportError("Timing component sum is not finite")
    qualified = _boolean(wrapper.get("qualified"), "qualified") if wrapper else None
    # Qualification is the hybrid decision, never the model-only gate result.
    if (
        qualified is not None
        and backend == "hybrid"
        and (goal is None or qualified != goal["passed"])
    ):
        raise ReportError("Qualification outcome disagrees with hybrid goal")
    device = raw.get("inference_device", "unrecorded")
    if not isinstance(device, str) or not re.fullmatch(
        r"[A-Za-z0-9_:. -]{1,64}", device
    ):
        raise ReportError("Invalid inference_device")
    completed = wrapper.get("completed_at_utc") if wrapper else None
    if completed is not None:
        from datetime import datetime

        try:
            datetime.fromisoformat(completed)
        except (TypeError, ValueError) as exc:
            raise ReportError("Invalid completed_at_utc") from exc
    return {
        "id": entry["id"],
        "label": label,
        "purpose": purpose,
        "note": entry.get("note", ""),
        "backend": backend,
        "documents": documents,
        "typed_span": typed,
        "character": character,
        "document_leak_free": leak,
        "negative_documents": negative,
        "per_label": per_label,
        "goal": goal,
        "timing": {
            "neural_seconds": neural,
            "rules_seconds": rules,
            "total_seconds": total,
            "documents_per_second": documents / total if total else None,
            "neural_inference_performed": _boolean(
                raw.get("neural_inference_performed"), "neural_inference_performed"
            ),
            "device": device,
        },
        "provenance": {
            "source_file": source.name,
            "source_sha256": digest,
            "report_selector": f"reports.{backend}" if wrapper else "root",
            "adapter_sha256": _digest(
                artifacts.get("adapter.safetensors"), "adapter digest"
            ),
            "config_sha256": _digest(artifacts.get("config.json"), "config digest"),
            "corpus_sha256": _digest(corpus.get("sha256"), "corpus digest"),
            "detector_sha256": _hash_manifest(
                raw.get("detector_sources"), "detector_sources"
            ),
            "neural_sha256": _hash_manifest(
                _object(raw.get("neural_runtime", {}), "neural_runtime").get(
                    "neural_sources"
                ),
                "neural_sources",
            ),
            "rules_sha256": _hash_manifest(raw.get("rule_sources"), "rule_sources"),
            "completed_at": completed,
            **flags,
            "qualified": qualified,
        },
    }


def load_report(config_path: str | Path) -> tuple[dict, set[Path]]:
    """Return a validated, allowlisted snapshot and its explicit input paths."""
    config_path = Path(config_path).resolve()
    config, config_digest = _read(config_path)
    inputs = {config_path}
    title = _text(config.get("title"), "title")
    description = config.get("description", "Saved evaluation results")
    _text(description, "description")
    notes = config.get("notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise ReportError("notes must be a list of strings")
    entries = config.get("runs")
    if not isinstance(entries, list) or not entries:
        raise ReportError("runs must be a nonempty list")
    sources: dict[Path, tuple[dict, str]] = {}

    def read_explicit(value: Any) -> tuple[Path, dict, str]:
        path = (config_path.parent / _text(value, "input path")).resolve()
        if path.suffix.lower() != ".json":
            raise ReportError("Report inputs must be JSON files")
        inputs.add(path)
        if path not in sources:
            sources[path] = _read(path)
        raw, digest = sources[path]
        return path, raw, digest

    candidate = None
    candidate_raw = None
    if config.get("candidate") is not None:
        path, candidate_raw, digest = read_explicit(config["candidate"])
        adapter = _digest(
            candidate_raw.get("adapter_sha256"), "selected adapter digest"
        )
        if adapter is None:
            raise ReportError("Candidate adapter digest is required")
        model_dir = _text(candidate_raw.get("selected_model_dir"), "selected_model_dir")
        epoch = candidate_raw.get("selected_epoch")
        candidate = {
            "name": Path(model_dir).name,
            "adapter_sha256": adapter,
            "epoch": _count(epoch, "selected_epoch", positive=True)
            if epoch is not None
            else None,
            "source": {"file": path.name, "sha256": digest},
        }
    runs, ids = [], set()
    for entry in entries:
        entry = _object(entry, "run")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(
            r"[a-zA-Z][a-zA-Z0-9_-]{0,63}", identifier
        ):
            raise ReportError("Run id must be a short alphanumeric identifier")
        if identifier in ids:
            raise ReportError("Run ids must be unique")
        ids.add(identifier)
        if not isinstance(entry.get("note", ""), str):
            raise ReportError("run.note must be a string")
        path, raw, digest = read_explicit(entry.get("path"))
        if (
            entry.get("sha256") is not None
            and _digest(entry["sha256"], "source digest") != digest
        ):
            raise ReportError(f"{path.name} differs from its configured SHA-256")
        wrapper = raw if "reports" in raw else None
        if wrapper is not None:
            reports = _object(wrapper["reports"], "reports")
            backend = entry.get("backend")
            if backend not in {"model", "hybrid"} or backend not in reports:
                raise ReportError(
                    "A qualification wrapper requires backend model or hybrid"
                )
            raw = _object(reports[backend], "embedded report")
        run = _normalize(raw, entry, path, digest, wrapper)
        if candidate is not None:
            if candidate["adapter_sha256"] != run["provenance"]["adapter_sha256"]:
                raise ReportError(
                    f"{entry['label']}: adapter differs from selected candidate"
                )
            selected_config = _digest(
                candidate_raw.get("model_artifacts", {}).get("config.json"),
                "selected config digest",
            )
            if (
                selected_config is not None
                and selected_config != run["provenance"]["config_sha256"]
            ):
                raise ReportError(
                    f"{entry['label']}: model config differs from selected candidate"
                )
            if candidate_raw.get("neural_runtime") is not None and (
                candidate_raw["neural_runtime"] != raw.get("neural_runtime")
            ):
                raise ReportError(
                    f"{entry['label']}: neural runtime differs from selected candidate"
                )
            if (
                run["backend"] == "hybrid"
                and candidate_raw.get("rule_sources") is not None
                and (candidate_raw["rule_sources"] != raw.get("rule_sources"))
            ):
                raise ReportError(
                    f"{entry['label']}: rule sources differ from selected candidate"
                )
            selected_report = candidate_raw.get("actual_development_report")
            if selected_report and Path(selected_report).name == path.name:
                if candidate_raw.get("actual_development_report_sha256") != digest:
                    raise ReportError(
                        "Development report differs from candidate selection evidence"
                    )
        runs.append(run)
    default_run = config.get("default_run", runs[0]["id"])
    if default_run not in ids:
        raise ReportError("default_run must name a configured run")
    warnings = []
    if len({run["provenance"]["corpus_sha256"] for run in runs}) > 1:
        warnings.append(
            "These runs use different corpora. Quality and speed differences are descriptive; they are not controlled experiments."
        )
    for field, name in (
        ("adapter_sha256", "adapters"),
        ("config_sha256", "model configurations"),
        ("neural_sha256", "neural source versions"),
    ):
        if len({run["provenance"][field] for run in runs}) > 1:
            warnings.append(
                f"The selected runs use different or unrecorded {name}; inspect source identities before comparing."
            )
    if (
        len(
            {
                run["provenance"]["rules_sha256"]
                for run in runs
                if run["backend"] == "hybrid"
            }
        )
        > 1
    ):
        warnings.append(
            "Hybrid runs use different or unrecorded rule versions; inspect source identities before comparing."
        )
    if any(
        run["provenance"]["adapter_sha256"] is None
        or run["provenance"]["corpus_sha256"] is None
        for run in runs
    ):
        warnings.append(
            "Some model or corpus hashes are unavailable. Those results cannot be matched to a frozen identity."
        )
    return {
        "schema_version": 1,
        "title": title,
        "description": description,
        "notes": notes,
        "candidate": candidate,
        "generated_at": None,
        "default_run": default_run,
        "runs": runs,
        "warnings": warnings,
        "config_source": {"file": config_path.name, "sha256": config_digest},
    }, inputs


def build_report(config_path: str | Path, output: str | Path) -> dict:
    """Validate all evidence before atomically replacing the output HTML."""
    snapshot, inputs = load_report(config_path)
    output = Path(output)
    if output.suffix.lower() != ".html" or output.is_symlink():
        raise ReportError("Output must be an .html file, not a symlink")
    output = output.resolve()
    if output in inputs or (
        output.exists() and any(output.samefile(path) for path in inputs)
    ):
        raise ReportError("Output must not overwrite an input file")
    html = render_report(snapshot)
    # Recheck source identities after rendering, before replacing a previous report.
    if load_report(config_path)[0] != snapshot:
        raise ReportError(
            "Inputs changed during report generation; retry with stable reports"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=".pipeline-report-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(html)
        os.replace(temp_name, output)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
    return snapshot


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="JSON config; input paths are relative to it",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Self-contained .html output"
    )
    args = parser.parse_args(argv)
    try:
        snapshot = build_report(args.config, args.output)
    except (OSError, ReportError) as exc:
        parser.exit(2, f"Report generation failed: {exc}\n")
    print(f"Wrote {args.output} ({len(snapshot['runs'])} runs; aggregate metrics only)")


if __name__ == "__main__":
    main()
