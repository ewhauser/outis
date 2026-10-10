"""Benchmark distance calculations from invented, normalized aggregate runs."""

from copy import deepcopy

import pytest

from outis.evaluation.report_html import render_report
from outis.evaluation.report_progress import benchmark_progress

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
GATES = (
    "typed_span_precision_at_least_99_percent",
    "typed_span_recall_at_least_99_percent",
    "typed_span_f1_at_least_99_percent",
    "phi_character_recall_at_least_99_percent",
    "recall_wilson_lower_95_at_least_99_percent",
    "all_labels_have_sufficient_support",
    "every_label_recall_at_least_99_percent",
)


def page(run):
    run = {"id": "reviewed", "label": "Reviewed evaluation", "backend": "hybrid", **run}
    return render_report(
        {"title": "Benchmark evidence", "runs": [run], "default_run": run["id"]}
    )


def test_html_leads_with_benchmark_and_preserves_historical_scope(run):
    run["purpose"] = "historical_qualification"
    output = page(run)
    assert "Historical benchmark not met" in output
    assert "5<span> / 7</span>" in output
    assert output.index("Historical benchmark not met") < output.index(
        "Development and qualification side by side"
    )
    assert "Notes with uncovered annotated PHI" in output
    assert "acceptance target not set" in output
    assert "before replacement" in output
    assert "Lowest label recall (NAME)" in output
    assert "9.00 pp" in output
    assert "Smallest label sample" in output


def test_html_keeps_tiny_nonzero_shortfalls_visible(run):
    run["character"] = metrics(989999, 0, 10001)
    run["goal"]["checks"][3]["passed"] = False
    output = page(run)
    assert "&lt;99%" in output
    assert "&lt;0.01 pp" in output
    assert "0.00 pp" not in output


def test_html_without_goal_does_not_claim_benchmark_met(run):
    run.pop("goal")
    output = page(run)
    assert "Benchmark status unavailable" in output
    assert "Benchmark checks unavailable" in output
    assert "required checks met" not in output
    assert "Recall target not recorded" in output


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


@pytest.fixture
def run():
    rows = [
        {"label": label, **metrics(100, 0, 0), "sufficient_support": True}
        for label in LABELS
    ]
    rows[0].update(metrics(90, 2, 10))
    rows[1].update(metrics(98, 3, 2))
    # 1,388 matched / 1,400 gold spans; the Wilson gate and NAME/DATE
    # recall gates fail while the five remaining aggregate gates pass.
    return {
        "documents": 100,
        "typed_span": metrics(1388, 5, 12),
        "character": metrics(9900, 0, 100),
        "document_leak_free": {"count": 90, "rate": 0.9},
        "per_label": rows,
        "goal": {
            "threshold": 0.99,
            "minimum_label_support": 20,
            "recall_wilson_lower_95": 0.9850776282206727,
            "passed": False,
            "checks": [
                {
                    "name": name,
                    "label": name,
                    "passed": name not in (GATES[4], GATES[6]),
                }
                for name in GATES
            ],
        },
    }


def by_name(result):
    return {gate["name"]: gate for gate in result["gates"]}


def test_seven_gate_distances_use_declared_targets_and_correct_units(run):
    result = benchmark_progress(run)
    assert result["status"] == "not_met"
    gates = by_name(result)
    assert set(gates) == set(GATES)
    assert len(result["gates"]) == 7
    expected = [
        (GATES[0], 1388 / 1393, 0.99, "rate", True),
        (GATES[1], 1388 / 1400, 0.99, "rate", True),
        (GATES[2], run["typed_span"]["f1"], 0.99, "rate", True),
        (GATES[3], 0.99, 0.99, "rate", True),
        (GATES[4], 0.9850776282206727, 0.99, "rate", False),
        (GATES[5], 100, 20, "count", True),
        (GATES[6], 0.90, 0.99, "rate", False),
    ]
    for name, actual, target, unit, passed in expected:
        assert gates[name]["actual"] == pytest.approx(actual)
        assert gates[name]["target"] == target
        assert gates[name]["unit"] == unit
        assert gates[name]["gap"] == pytest.approx(max(0, target - actual))
        assert gates[name]["passed"] is passed
    assert result["passed_checks"] == 5
    assert result["total_checks"] == 7


def test_all_seven_supported_passing_gates_mark_benchmark_met(run):
    run["typed_span"] = metrics(1400, 0, 0)
    run["character"] = metrics(14000, 0, 0)
    run["document_leak_free"] = {"count": 100, "rate": 1.0}
    for row in run["per_label"]:
        row.update(metrics(100, 0, 0))
    run["goal"]["recall_wilson_lower_95"] = 0.9972636092227101
    run["goal"]["passed"] = True
    for check in run["goal"]["checks"]:
        check["passed"] = True
    result = benchmark_progress(run)
    assert result["status"] == "met"
    assert result["passed_checks"] == result["total_checks"] == 7
    assert result["labels_below_target"] == 0
    assert result["leaking_documents"] == 0
    assert result["leaking_rate"] == 0.0
    assert all(gate["passed"] is True and gate["gap"] == 0 for gate in result["gates"])


def test_worst_labels_sort_by_recall_and_shortfall_is_a_rate_distance(run):
    run["per_label"].reverse()
    result = benchmark_progress(run)
    rows = result["label_rows"]
    assert [row["label"] for row in rows[:2]] == ["NAME", "DATE"]
    assert result["labels_assessed"] == 14
    assert result["labels_below_target"] == 2
    assert result["weakest_label"] == "NAME"
    assert result["weakest_recall"] == 0.9
    assert rows[0]["gap"] == pytest.approx(0.09)
    assert rows[1]["gap"] == pytest.approx(0.01)
    assert rows[0]["missed"] == 10
    assert rows[0]["support"] == 100
    assert rows[0]["passed"] is False
    assert all(row["gap"] == 0 and row["passed"] is True for row in rows[2:])


def test_leaking_document_diagnostic_has_no_invented_gate_or_target(run):
    before = benchmark_progress(run)
    assert before["leaking_documents"] == 10
    assert before["leaking_rate"] == pytest.approx(0.1)
    run["document_leak_free"] = {"count": 89, "rate": 0.89}
    after = benchmark_progress(run)
    assert after["leaking_documents"] == 11
    assert after["leaking_rate"] == pytest.approx(0.11)
    assert after["gates"] == before["gates"]
    assert after["passed_checks"] == before["passed_checks"]
    assert after["total_checks"] == 7
    assert not any(
        "leak" in gate["name"] or "document" in gate["name"] for gate in after["gates"]
    )


def test_missing_goal_keeps_actuals_but_does_not_invent_targets_or_a_verdict(run):
    run.pop("goal")
    result = benchmark_progress(run)
    assert result["status"] == "unavailable"
    assert result["total_checks"] == 0
    assert result["passed_checks"] == 0
    assert result["labels_below_target"] is None
    assert result["weakest_recall"] == 0.9
    assert by_name(result)[GATES[0]]["actual"] == pytest.approx(1388 / 1393)
    assert len(result["gates"]) == 7
    assert all(
        gate["target"] is None and gate["gap"] is None and gate["passed"] is None
        for gate in result["gates"]
    )
    assert all(
        row["gap"] is None and row["passed"] is None for row in result["label_rows"]
    )


@pytest.mark.parametrize("value", [None, []])
def test_missing_per_label_evidence_makes_label_gates_unavailable(run, value):
    if value is None:
        run.pop("per_label")
    else:
        run["per_label"] = value
    result = benchmark_progress(run)
    assert result["status"] == "unavailable"
    assert result["labels_assessed"] == 0
    assert result["labels_below_target"] is None
    assert result["weakest_label"] is None
    assert result["weakest_recall"] is None
    assert result["label_rows"] == []
    for name in (GATES[5], GATES[6]):
        gate = by_name(result)[name]
        assert gate["actual"] is None
        assert gate["gap"] is None
        assert gate["passed"] is None


def test_zero_support_is_not_a_perfect_recall_and_support_gap_is_a_count(run):
    row = next(row for row in run["per_label"] if row["label"] == "AGE")
    row.update(metrics(0, 0, 0), sufficient_support=False)
    run["goal"]["checks"][5]["passed"] = False
    result = benchmark_progress(run)
    assert result["status"] == "unavailable"
    assert result["labels_assessed"] == 13
    assert result["labels_below_target"] == 2
    assert result["weakest_label"] == "NAME"
    assert result["label_rows"][-1]["label"] == "AGE"
    assert result["label_rows"][-1]["recall"] is None
    assert result["label_rows"][-1]["gap"] is None
    assert result["label_rows"][-1]["passed"] is None
    support_gate = by_name(result)[GATES[5]]
    assert support_gate["actual"] == 0
    assert support_gate["target"] == 20
    assert support_gate["gap"] == 20
    assert support_gate["unit"] == "count"
    assert support_gate["passed"] is False
    recall_gate = by_name(result)[GATES[6]]
    assert recall_gate["actual"] is None
    assert recall_gate["gap"] is None
    assert recall_gate["passed"] is None


def test_empty_span_denominators_do_not_show_recorded_one_as_perfect_accuracy(run):
    run["typed_span"] = metrics(0, 0, 0)
    run["character"] = metrics(0, 0, 0)
    result = benchmark_progress(run)
    assert result["status"] == "unavailable"
    for name in GATES[:5]:
        gate = by_name(result)[name]
        assert gate["actual"] is None
        assert gate["gap"] is None
        assert gate["passed"] is None


def test_no_predicted_spans_is_missing_precision_evidence_but_measured_zero_recall(run):
    run["typed_span"] = metrics(0, 0, 10)
    result = benchmark_progress(run)
    gates = by_name(result)
    assert result["status"] == "unavailable"
    assert gates[GATES[0]]["actual"] is None
    assert gates[GATES[0]]["gap"] is None
    assert gates[GATES[0]]["passed"] is None
    assert gates[GATES[1]]["actual"] == 0.0
    assert gates[GATES[1]]["gap"] == 0.99
    assert gates[GATES[2]]["actual"] == 0.0
    assert gates[GATES[2]]["gap"] == 0.99


def test_incomplete_recorded_checks_cannot_produce_a_complete_readiness_verdict(run):
    run["goal"]["checks"].pop()
    result = benchmark_progress(run)
    assert result["status"] == "unavailable"
    assert result["total_checks"] == 6
    assert by_name(result)[GATES[6]]["passed"] is None


def test_diagnostic_counts_are_unavailable_without_document_denominator(run):
    run.pop("documents")
    result = benchmark_progress(run)
    assert result["leaking_documents"] is None
    assert result["leaking_rate"] is None


def test_progress_is_deterministic_and_does_not_mutate_normalized_evidence(run):
    original = deepcopy(run)
    first = benchmark_progress(run)
    second = benchmark_progress(run)
    assert first == second
    assert run == original
