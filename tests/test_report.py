"""Aggregate-only report checks using invented metrics and no model or corpus."""

import hashlib
import json
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path

import pytest

from outis.evaluation.report import ReportError, build_report, load_report

ADAPTER = "a" * 64
CONFIG = "b" * 64
CORPUS = "c" * 64
LABELS = (
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


def metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "support": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0,
    }


def aggregate():
    return {
        "schema_version": 1,
        "backend": "hybrid",
        "documents": 4,
        "typed_span": metrics(8, 1, 2),
        "character": metrics(40, 2, 5),
        "document_leak_free": {"count": 3, "rate": 0.75},
        "negative_documents": {"count": 1, "clean_count": 0, "clean_rate": 0.0},
        "per_label": {
            **{
                label: {**metrics(0, 0, 0), "sufficient_support": False}
                for label in LABELS
            },
            "NAME": {**metrics(5, 0, 1), "sufficient_support": True},
            "DATE": {**metrics(3, 1, 1), "sufficient_support": True},
        },
        "recall_wilson_lower_95": 0.49016247153664183,
        "goal": {
            "threshold": 0.99,
            "minimum_label_support": 1,
            "checks": {
                "typed_span_precision_at_least_99_percent": False,
                "typed_span_recall_at_least_99_percent": False,
                "typed_span_f1_at_least_99_percent": False,
                "phi_character_recall_at_least_99_percent": False,
                "recall_wilson_lower_95_at_least_99_percent": False,
                "all_labels_have_sufficient_support": False,
                "every_label_recall_at_least_99_percent": False,
            },
            "passed": False,
        },
        "model_artifacts": {"adapter.safetensors": ADAPTER, "config.json": CONFIG},
        "corpus_manifest": {
            "sha256": CORPUS,
            "rows": 4,
            "independent_final": False,
            "used_for_training": False,
            "used_for_checkpoint_selection": True,
        },
        "inference_device": "mps",
        "neural_inference_seconds": 10.0,
        "rule_application_seconds": 2.0,
        "elapsed_seconds": 999.0,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "evidence" / "aggregate.json"
    config_path = tmp_path / "views" / "dashboard.json"
    report = aggregate()
    config = {
        "schema_version": 1,
        "title": "Invented pipeline results",
        "description": "Aggregate fixture; no clinical text",
        "notes": ["Different runs may use different corpora."],
        "runs": [
            {
                "id": "current",
                "label": "Current hybrid",
                "purpose": "development",
                "path": "../evidence/aggregate.json",
                "backend": "hybrid",
            }
        ],
    }
    write_json(source, report)
    write_json(config_path, config)
    return config_path, source, config, report


def test_aggregate_numbers_and_input_identities_are_preserved(inputs):
    config_path, source, _, original = inputs
    snapshot, paths = load_report(config_path)
    run = snapshot["runs"][0]
    assert paths == {config_path.resolve(), source.resolve()}
    assert run["typed_span"] == metrics(8, 1, 2)
    assert run["character"] == metrics(40, 2, 5)
    assert run["document_leak_free"] == {"count": 3, "rate": 0.75}
    assert run["negative_documents"]["clean_rate"] == 0.0
    assert [row["label"] for row in run["per_label"][:2]] == ["DATE", "NAME"]
    assert run["timing"]["total_seconds"] == 12.0
    assert run["timing"]["documents_per_second"] == pytest.approx(4 / 12)
    assert (
        run["provenance"]["source_sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert run["provenance"]["corpus_sha256"] == original["corpus_manifest"]["sha256"]
    assert run["provenance"]["report_selector"] == "root"


@pytest.mark.parametrize(
    "field", ["neural_inference_seconds", "rule_application_seconds", "both"]
)
def test_missing_timing_never_uses_evaluation_elapsed_seconds(inputs, field):
    config_path, source, _, raw = inputs
    for key in ("neural_inference_seconds", "rule_application_seconds"):
        if field in (key, "both"):
            raw.pop(key)
    write_json(source, raw)
    timing = load_report(config_path)[0]["runs"][0]["timing"]
    assert timing["total_seconds"] is None
    assert timing["documents_per_second"] is None
    if field == "both":
        assert timing["neural_seconds"] is None and timing["rules_seconds"] is None


def test_cached_neural_time_remains_marked_as_reused(inputs):
    config_path, source, _, raw = inputs
    raw["neural_inference_performed"] = False
    write_json(source, raw)
    timing = load_report(config_path)[0]["runs"][0]["timing"]
    assert timing["neural_inference_performed"] is False
    assert timing["neural_seconds"] == 10.0
    assert timing["rules_seconds"] == 2.0


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), 2),
        (("documents",), 0),
        (("documents",), True),
        (("typed_span", "support"), 11),
        (("typed_span", "true_positives"), -1),
        (("typed_span", "precision"), 0.5),
        (("typed_span", "recall"), 1.1),
        (("typed_span", "f1"), float("nan")),
        (("character", "false_positives"), 0.5),
        (("document_leak_free", "count"), 5),
        (("document_leak_free", "rate"), 0.5),
        (("negative_documents", "clean_count"), 2),
        (("corpus_manifest", "rows"), 5),
        (("goal", "passed"), True),
        (("goal", "checks"), {"unknown_check": True}),
        (("goal", "threshold"), 0.95),
        (("goal", "minimum_label_support"), 0),
        (("recall_wilson_lower_95",), 1.0),
        (("per_label", "SSN", "sufficient_support"), True),
        (("neural_inference_seconds",), float("inf")),
        (("rule_application_seconds",), -1),
        (("neural_inference_performed",), "false"),
        (("independent_final",), True),
    ],
)
def test_inconsistent_or_invalid_aggregate_is_rejected(inputs, path, value):
    config_path, source, _, raw = inputs
    target = raw
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    write_json(source, raw)
    with pytest.raises(ReportError):
        load_report(config_path)


def test_per_label_counts_must_sum_to_overall_metrics(inputs):
    config_path, source, _, raw = inputs
    raw["per_label"]["NAME"] = metrics(4, 0, 1)
    write_json(source, raw)
    with pytest.raises(ReportError, match="does not sum"):
        load_report(config_path)


@pytest.mark.parametrize("attack", ["passing_checks", "subset_checks", "missing_label"])
def test_goal_cannot_claim_success_by_forging_checks_or_omitting_evidence(
    inputs, attack
):
    config_path, source, _, raw = inputs
    if attack == "passing_checks":
        raw["goal"]["checks"] = dict.fromkeys(raw["goal"]["checks"], True)
        raw["goal"]["passed"] = True
    elif attack == "subset_checks":
        raw["goal"]["checks"].pop("all_labels_have_sufficient_support")
    else:
        raw["per_label"].pop("SSN")
    write_json(source, raw)
    with pytest.raises(ReportError, match="Goal checks|complete set|all entity labels"):
        load_report(config_path)


def test_empty_support_is_retained_without_inventing_observations(inputs):
    config_path, source, _, raw = inputs
    raw["per_label"]["SSN"] = {**metrics(0, 0, 0), "sufficient_support": False}
    write_json(source, raw)
    rows = load_report(config_path)[0]["runs"][0]["per_label"]
    empty = next(row for row in rows if row["label"] == "SSN")
    assert empty["support"] == 0
    assert empty["sufficient_support"] is False


@pytest.mark.parametrize("backend", ["model", "hybrid"])
def test_wrapper_selects_requested_backend_and_keeps_qualification_identity(
    inputs, backend
):
    config_path, source, config, raw = inputs
    model = deepcopy(raw)
    model["backend"] = "model"
    model["rule_application_seconds"] = 0.0
    wrapper = {
        "schema_version": 1,
        "qualified": False,
        "completed_at_utc": "2026-09-30T12:34:56+00:00",
        "reports": {"model": model, "hybrid": raw},
    }
    config["runs"][0]["backend"] = backend
    write_json(source, wrapper)
    write_json(config_path, config)
    run = load_report(config_path)[0]["runs"][0]
    assert run["backend"] == backend
    assert run["provenance"]["report_selector"] == f"reports.{backend}"
    assert run["provenance"]["qualified"] is False
    assert run["provenance"]["completed_at"] == wrapper["completed_at_utc"]
    assert run["timing"]["rules_seconds"] == (0.0 if backend == "model" else 2.0)


def test_wrapper_requires_explicit_backend_and_consistent_hybrid_decision(inputs):
    config_path, source, config, raw = inputs
    write_json(
        source, {"schema_version": 1, "qualified": True, "reports": {"hybrid": raw}}
    )
    with pytest.raises(ReportError, match="Qualification outcome"):
        load_report(config_path)
    config["runs"][0].pop("backend")
    write_json(config_path, config)
    with pytest.raises(ReportError, match="requires backend"):
        load_report(config_path)


def test_historical_qualification_requires_and_preserves_independence(inputs):
    config_path, source, config, raw = inputs
    config["runs"][0]["purpose"] = "historical_qualification"
    write_json(config_path, config)
    with pytest.raises(ReportError, match="Qualification purpose"):
        load_report(config_path)
    raw["corpus_manifest"].update(
        independent_final=True,
        used_for_training=False,
        used_for_checkpoint_selection=False,
    )
    write_json(source, raw)
    run = load_report(config_path)[0]["runs"][0]
    assert run["purpose"] == "historical_qualification"
    assert run["provenance"]["independent_final"] is True
    assert run["provenance"]["used_for_checkpoint_selection"] is False
    assert run["provenance"]["used_for_training"] is False
    assert run["provenance"]["qualified"] is None


def test_paths_are_relative_to_config_and_embedded_paths_are_never_read(
    inputs, tmp_path, monkeypatch
):
    config_path, source, _, raw = inputs
    raw["data_file"] = str(tmp_path / "unapproved-patient-notes.json")
    raw["corpus_manifest"]["path"] = raw["data_file"]
    write_json(source, raw)
    original_read = Path.read_bytes
    reads = []

    def checked_read(path):
        reads.append(path.resolve())
        assert path.resolve() in {config_path.resolve(), source.resolve()}
        return original_read(path)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(Path, "read_bytes", checked_read)
    snapshot, _ = load_report(config_path)
    assert snapshot["runs"][0]["documents"] == 4
    assert set(reads) == {config_path.resolve(), source.resolve()}


def test_allowlist_excludes_failure_examples_raw_notes_and_absolute_paths(
    inputs, tmp_path
):
    config_path, source, _, raw = inputs
    marker = "PRIVATE_PATIENT_EXAMPLE_DO_NOT_EXPORT"
    raw.update(
        failures=[{"text": marker}],
        raw_notes=[marker],
        scope=marker,
        prediction_cache=f"/private/{marker}/cache.json",
    )
    raw["typed_span"]["source_text"] = marker
    raw["per_label"]["NAME"]["example"] = marker
    raw["corpus_manifest"]["path"] = f"/private/{marker}/notes.jsonl"
    raw["detector_sources"] = {f"/private/{marker}/model.py": "d" * 64}
    write_json(source, raw)
    output = tmp_path / "report.html"
    snapshot = build_report(config_path, output)
    for serialized in (json.dumps(snapshot), output.read_text()):
        assert marker not in serialized
        assert str(tmp_path) not in serialized
        assert "raw_notes" not in serialized
        assert "prediction_cache" not in serialized
    assert snapshot["runs"][0]["provenance"]["detector_sha256"]


class ScriptParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.scripts = []
        self.current_script = None

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag == "script":
            self.current_script = {"attrs": dict(attrs), "text": ""}
            self.scripts.append(self.current_script)

    def handle_data(self, data):
        if self.current_script is not None:
            self.current_script["text"] += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.current_script = None


def test_html_escapes_config_text_and_prevents_script_breakout(inputs, tmp_path):
    config_path, _, config, _ = inputs
    attack = '</script><script id="injected-script">alert("marker")</script><img src=x onerror=alert(1)>'
    config["title"] = attack
    config["runs"][0]["label"] = attack
    config["notes"] = [attack]
    write_json(config_path, config)
    output = tmp_path / "report.html"
    build_report(config_path, output)
    html = output.read_text()
    parser = ScriptParser()
    parser.feed(html)
    assert not any(attrs.get("id") == "injected-script" for _, attrs in parser.tags)
    assert not any("onerror" in attrs for _, attrs in parser.tags)
    assert attack not in html
    payloads = [
        json.loads(script["text"])
        for script in parser.scripts
        if script["attrs"].get("type") == "application/json"
    ]
    assert len(payloads) == 1
    assert payloads[0]["title"] == attack
    assert payloads[0]["runs"][0]["label"] == attack


def test_output_is_deterministic_for_unchanged_sources(inputs, tmp_path):
    config_path, _, _, _ = inputs
    first, second = tmp_path / "first.html", tmp_path / "second.html"
    assert build_report(config_path, first) == build_report(config_path, second)
    assert first.read_bytes() == second.read_bytes()


def test_invalid_input_never_overwrites_existing_html(inputs, tmp_path):
    config_path, source, _, raw = inputs
    output = tmp_path / "report.html"
    output.write_text("previous validated report")
    raw["typed_span"]["recall"] = 0.123
    write_json(source, raw)
    with pytest.raises(ReportError):
        build_report(config_path, output)
    assert output.read_text() == "previous validated report"


@pytest.mark.parametrize("alias", ["hardlink", "symlink"])
def test_output_cannot_alias_an_input(inputs, tmp_path, alias):
    config_path, source, _, _ = inputs
    before = source.read_bytes()
    output = tmp_path / "aliased.html"
    if alias == "hardlink":
        output.hardlink_to(source)
    else:
        output.symlink_to(source)
    with pytest.raises(ReportError, match="input file|symlink"):
        build_report(config_path, output)
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    "kind", ["adapter", "config", "development_evidence", "neural_runtime", "rules"]
)
def test_selected_candidate_mismatch_is_rejected(inputs, kind):
    config_path, source, config, _ = inputs
    candidate = {
        "schema_version": 1,
        "selected_model_dir": "/private/models/candidate-seven",
        "selected_epoch": 7,
        "adapter_sha256": ADAPTER,
        "model_artifacts": {"config.json": CONFIG},
        "actual_development_report": str(source),
        "actual_development_report_sha256": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
    }
    if kind == "adapter":
        candidate["adapter_sha256"] = "e" * 64
    elif kind == "config":
        candidate["model_artifacts"]["config.json"] = "e" * 64
    elif kind == "development_evidence":
        candidate["actual_development_report_sha256"] = "e" * 64
    elif kind == "neural_runtime":
        candidate["neural_runtime"] = {"neural_sources": {"model.py": "e" * 64}}
    else:
        candidate["rule_sources"] = {"rules.py": "e" * 64}
    config["candidate"] = "candidate.json"
    write_json(config_path.parent / "candidate.json", candidate)
    write_json(config_path, config)
    with pytest.raises(ReportError, match="differ"):
        load_report(config_path)


def test_selected_candidate_exports_basename_and_approved_identity_only(inputs):
    config_path, _, config, _ = inputs
    candidate = {
        "schema_version": 1,
        "selected_model_dir": "/private/models/candidate-seven",
        "selected_epoch": 7,
        "adapter_sha256": ADAPTER,
        "model_artifacts": {"config.json": CONFIG},
        "private_notes": "PRIVATE_CANDIDATE_NOTE",
    }
    config["candidate"] = "candidate.json"
    write_json(config_path.parent / "candidate.json", candidate)
    write_json(config_path, config)
    snapshot, _ = load_report(config_path)
    assert snapshot["candidate"]["name"] == "candidate-seven"
    assert snapshot["candidate"]["epoch"] == 7
    assert snapshot["candidate"]["adapter_sha256"] == ADAPTER
    assert "/private/models" not in json.dumps(snapshot)
    assert "PRIVATE_CANDIDATE_NOTE" not in json.dumps(snapshot)
