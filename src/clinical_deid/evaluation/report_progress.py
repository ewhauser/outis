"""Summarize distance to recorded benchmark gates without reading any files."""

from __future__ import annotations

import math


_GATES = (
    ("typed_span_precision_at_least_99_percent", "Exact span precision", "rate"),
    ("typed_span_recall_at_least_99_percent", "Exact span recall", "rate"),
    ("typed_span_f1_at_least_99_percent", "Exact span F1", "rate"),
    ("phi_character_recall_at_least_99_percent", "PHI character recall", "rate"),
    (
        "recall_wilson_lower_95_at_least_99_percent",
        "Recall 95% Wilson lower bound",
        "rate",
    ),
    ("all_labels_have_sufficient_support", "Minimum label support", "count"),
    ("every_label_recall_at_least_99_percent", "Minimum label recall", "rate"),
)


def _rate(value):
    if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1:
        return float(value)
    return None


def _count(value):
    return value if type(value) is int and value >= 0 else None


def benchmark_progress(run: dict) -> dict:
    """Describe the seven recorded gates of an already normalized report run.

    This is a pure presentation helper: it does not infer targets, establish
    deployment readiness, or add a leak-free-document gate. Rate gaps are
    fractions for the renderer to convert to percentage points. A missing goal,
    incomplete check set, or undefined metric leaves status unavailable. Empty
    denominators and unsupported labels never inherit the evaluator's synthetic
    perfect rate. Recorded checks supply pass/fail only for evaluable gates.
    """
    goal = run.get("goal") or {}
    threshold = _rate(goal.get("threshold"))
    minimum_support = _count(goal.get("minimum_label_support"))
    recorded_checks = goal.get("checks") or []
    known_names = {name for name, _, _ in _GATES}
    checks = {
        check["name"]: check
        for check in recorded_checks
        if isinstance(check, dict)
        and check.get("name") in known_names
        and type(check.get("passed")) is bool
    }
    complete_checks = (
        len(recorded_checks) == len(_GATES)
        and set(checks) == known_names
        and type(goal.get("passed")) is bool
        and goal["passed"] == all(check["passed"] for check in checks.values())
    )

    label_rows = []
    for stats in run.get("per_label") or []:
        support = stats["support"]
        recall = _rate(stats.get("recall")) if support > 0 else None
        comparable = recall is not None and threshold is not None
        label_rows.append(
            {
                "label": stats["label"],
                "recall": recall,
                "support": support,
                "missed": stats["false_negatives"],
                "gap": max(0.0, threshold - recall) if comparable else None,
                "passed": recall >= threshold if comparable else None,
            }
        )
    label_rows.sort(
        key=lambda item: (
            item["recall"] is None,
            item["recall"] if item["recall"] is not None else 0.0,
            item["label"],
        )
    )
    assessed = [item for item in label_rows if item["recall"] is not None]
    weakest = assessed[0] if assessed else None

    typed = run.get("typed_span") or {}
    character = run.get("character") or {}
    tp = _count(typed.get("true_positives"))
    fp = _count(typed.get("false_positives"))
    fn = _count(typed.get("false_negatives"))
    support = _count(typed.get("support"))
    character_support = _count(character.get("support"))
    actuals = (
        _rate(typed.get("precision"))
        if tp is not None and fp is not None and tp + fp > 0
        else None,
        _rate(typed.get("recall")) if support else None,
        _rate(typed.get("f1"))
        if tp is not None and fp is not None and fn is not None and tp + fp + fn > 0
        else None,
        _rate(character.get("recall")) if character_support else None,
        _rate(goal.get("recall_wilson_lower_95")) if support else None,
        min(item["support"] for item in label_rows) if label_rows else None,
        weakest["recall"]
        if weakest is not None and len(assessed) == len(label_rows)
        else None,
    )
    gates = []
    for (name, label, unit), actual in zip(_GATES, actuals):
        target = minimum_support if unit == "count" else threshold
        comparable = actual is not None and target is not None
        check = checks.get(name)
        gates.append(
            {
                "name": name,
                "label": check.get("label", label) if check else label,
                "actual": actual,
                "target": target,
                "gap": max(0, target - actual) if comparable else None,
                "unit": unit,
                "passed": check["passed"] if check and comparable else None,
            }
        )
    evaluable = complete_checks and all(gate["passed"] is not None for gate in gates)
    status = ("met" if goal["passed"] else "not_met") if evaluable else "unavailable"

    documents = _count(run.get("documents"))
    leak_free = _count((run.get("document_leak_free") or {}).get("count"))
    leaking_documents = (
        documents - leak_free
        if documents and leak_free is not None and leak_free <= documents
        else None
    )
    return {
        "status": status,
        "passed_checks": sum(gate["passed"] is True for gate in gates),
        "total_checks": len(checks),
        "leaking_documents": leaking_documents,
        "leaking_rate": leaking_documents / documents
        if leaking_documents is not None
        else None,
        "labels_below_target": sum(item["passed"] is False for item in label_rows)
        if threshold is not None and assessed
        else None,
        "labels_assessed": len(assessed),
        "weakest_label": weakest["label"] if weakest else None,
        "weakest_recall": weakest["recall"] if weakest else None,
        "gates": gates,
        "label_rows": label_rows,
    }
