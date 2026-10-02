"""Render a normalized evaluation snapshot as a standalone, offline HTML report.

This module accepts aggregate metrics only. It does not load reports, model
artifacts, corpora, or patient text, and makes no qualification decisions.
"""

from __future__ import annotations

import html
import json
import math
from typing import Any

from .report_progress import benchmark_progress


def _text(value: Any, fallback: str = "Unavailable") -> str:
    return html.escape(fallback if value is None else str(value), quote=True)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _count(value: Any) -> str:
    number = _number(value)
    return (
        f"{int(number):,}"
        if number is not None and number.is_integer()
        else "Unavailable"
    )


def _percent(value: Any) -> str:
    if value == "N/A":
        return "N/A"
    number = _number(value)
    if number is None or not 0 <= number <= 1:
        return "Unavailable"
    # Preserve small deviations from 100%, even for near-perfect evaluations.
    formatted = f"{number * 100:.2f}".rstrip("0").rstrip(".")
    if number < 1 and formatted == "100":
        formatted = "<100"
    return html.escape(formatted) + "%"


def _metric_rate(metrics: dict, key: str) -> Any:
    """Do not turn a source's empty-denominator convention into a quality claim."""
    true_positives = _number(metrics.get("true_positives"))
    false_positives = _number(metrics.get("false_positives"))
    predicted = (
        true_positives + false_positives
        if true_positives is not None and false_positives is not None
        else None
    )
    support = _number(metrics.get("support"))
    if (key == "precision" and predicted == 0) or (key == "recall" and support == 0):
        return "N/A"
    if key == "f1" and support == 0 and predicted == 0:
        return "N/A"
    return metrics.get(key)


def _seconds(value: Any) -> str:
    number = _number(value)
    return f"{number:,.3f} s" if number is not None and number >= 0 else "Unavailable"


def _boolean(value: Any) -> str:
    return "Yes" if value is True else "No" if value is False else "Unavailable"


def _bar(value: Any, label: str, detail: str = "") -> str:
    number = _number(value)
    extra = f'<span class="metric-detail">{detail}</span>' if detail else ""
    if number is None or not 0 <= number <= 1:
        unavailable = "N/A" if value == "N/A" else "Unavailable"
        return (
            f'<div class="metric"><span class="muted">{unavailable}</span>{extra}</div>'
        )
    percentage = number * 100
    return (
        '<div class="metric"><span class="metric-value">'
        + _percent(number)
        + f'</span><span class="bar" role="meter" aria-label="{_text(label)}" '
        + f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percentage:.12g}" '
        + f'aria-valuetext="{_percent(number)}"><span style="width:{percentage:.9f}%"></span></span>'
        + extra
        + "</div>"
    )


def _purpose(run: dict) -> str:
    return {
        "development": "Development",
        "qualification": "Qualification",
        "historical_qualification": "Historical qualification",
        "diagnostic": "Diagnostic",
    }.get(run.get("purpose"), "Unspecified purpose")


def _backend(run: dict) -> str:
    return {
        "model": "Model only",
        "hybrid": "Hybrid",
        "unspecified": "Backend unspecified",
    }.get(run.get("backend"), "Backend unspecified")


def _badge(label: str, style: str = "neutral") -> str:
    return f'<span class="badge {style}">{_text(label)}</span>'


def _counts(metrics: dict) -> str:
    return (
        f"<span>{_count(metrics.get('true_positives'))} matched</span>"
        f"<span>{_count(metrics.get('false_positives'))} extra</span>"
        f"<span>{_count(metrics.get('false_negatives'))} missed</span>"
    )


def _comparison(runs: list[dict], selected: int) -> str:
    rows = []
    for index, run in enumerate(runs):
        typed, chars = run.get("typed_span") or {}, run.get("character") or {}
        leak = run.get("document_leak_free") or {}
        label = str(run.get("label") or run.get("id") or f"Run {index + 1}")
        rows.append(
            f'<tr data-comparison-index="{index}" class="{"selected-row" if index == selected else ""}">'
            f'<th scope="row"><span class="run-name">{_text(label)}</span>'
            f'<span class="row-subtitle">{_text(_purpose(run))} · {_text(_backend(run))}</span>'
            f'<span class="row-subtitle">{_count(run.get("documents"))} notes</span></th>'
            f"<td>{_bar(_metric_rate(typed, 'precision'), label + ' typed span precision')}</td>"
            f"<td>{_bar(_metric_rate(typed, 'recall'), label + ' typed span recall')}</td>"
            f"<td>{_bar(_metric_rate(typed, 'f1'), label + ' typed span F1')}</td>"
            f"<td>{_bar(_metric_rate(chars, 'recall'), label + ' character recall')}</td>"
            f"<td>{_bar(leak.get('rate'), label + ' leak-free document rate', _count(leak.get('count')) + ' / ' + _count(run.get('documents')) + ' notes')}</td>"
            "</tr>"
        )
    if not rows:
        return '<p class="empty">No evaluation runs are included in this snapshot.</p>'
    return (
        '<div class="table-scroll" tabindex="0" role="region" aria-label="Run quality comparison">'
        '<table class="comparison"><caption class="sr-only">Quality across recorded runs. All bars use a zero to one hundred percent scale.</caption>'
        '<thead><tr><th scope="col">Recorded run</th><th scope="col">Span precision</th>'
        '<th scope="col">Span recall</th><th scope="col">Span F1</th>'
        '<th scope="col">Character recall</th><th scope="col">Leak-free notes</th></tr></thead>'
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _metric_card(title: str, metrics: dict) -> str:
    return (
        f'<div class="metric-card"><h4>{_text(title)}</h4><div class="triple">'
        + "".join(
            f'<div><span class="small-label">{name}</span><strong>{_percent(_metric_rate(metrics, key))}</strong></div>'
            for name, key in (
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1", "f1"),
            )
        )
        + '</div><div class="count-line">'
        + _counts(metrics)
        + "</div>"
        + f'<p class="support">Gold support: {_count(metrics.get("support"))}</p></div>'
    )


def _target_rate(value: Any, target: Any) -> str:
    """Keep a tiny shortfall visible when the displayed rates round equally."""
    actual, threshold = _number(value), _number(target)
    if actual is not None and threshold is not None and actual < threshold:
        if _percent(actual) == _percent(threshold):
            return "&lt;" + _percent(threshold)
    return _percent(value)


def _gap(value: Any, unit: str = "rate") -> str:
    value = _number(value)
    if value is None:
        return "Unavailable"
    if value == 0:
        return "None"
    if unit == "count":
        return _count(value) + " spans"
    points = value * 100
    return "&lt;0.01 pp" if points < 0.01 else f"{points:.2f} pp"


def _target_bar(value: Any, target: Any, label: str) -> str:
    actual, threshold = _number(value), _number(target)
    if actual is None:
        return '<span class="muted">N/A</span>'
    marker = (
        f'<i class="target-marker" style="left:{threshold * 100:.9f}%" title="Target: {_percent(threshold)}"></i>'
        if threshold is not None
        else ""
    )
    color = "gap-fill" if threshold is not None and actual < threshold else "met-fill"
    return (
        f'<div class="target-metric"><strong>{_target_rate(actual, threshold)}</strong>'
        f'<span class="target-track" role="meter" aria-label="{_text(label)}" '
        f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{actual * 100:.12g}">'
        f'<span class="{color}" style="width:{actual * 100:.9f}%"></span>{marker}</span></div>'
    )


def _progress_overview(run: dict, progress: dict, index: int) -> str:
    purpose = run.get("purpose")
    prefix = (
        "Historical benchmark"
        if purpose == "historical_qualification"
        else ("Development benchmark" if purpose == "development" else "Benchmark")
    )
    status = progress["status"]
    headline = (
        prefix
        + {"met": " met", "not_met": " not met", "unavailable": " status unavailable"}[
            status
        ]
    )
    provenance = run.get("provenance") or {}
    date = provenance.get("completed_at")
    date_line = f" · Evaluated {_text(str(date).split('T')[0])}" if date else ""
    count_line = (
        f"<strong>{progress['passed_checks']}<span> / {progress['total_checks']}</span></strong>"
        "<span>required checks met</span>"
        if progress["total_checks"]
        else "<strong>—</strong><span>Benchmark checks unavailable</span>"
    )
    goal = run.get("goal") or {}
    target = goal.get("threshold")
    chars = run.get("character") or {}
    character_rate = _metric_rate(chars, "recall")
    char_gap = next(
        (
            gate["gap"]
            for gate in progress["gates"]
            if gate["name"] == "phi_character_recall_at_least_99_percent"
        ),
        None,
    )
    below, assessed = progress["labels_below_target"], progress["labels_assessed"]
    labels_value = _count(below) if below is not None else "Unavailable"
    labels_note = (
        f"of {assessed} assessed labels · target {_percent(target)} each"
        if target is not None
        else "Recall target not recorded"
    )
    negative = run.get("negative_documents") or {}
    negative_count = _number(negative.get("count"))
    negative_value = (
        _percent(negative.get("clean_rate"))
        if negative_count
        else ("N/A" if negative_count == 0 else "Unavailable")
    )
    negative_note = (
        f"{_count(negative.get('clean_count'))} / {_count(negative.get('count'))} notes left untouched"
        if negative_count
        else "No negative documents in this run"
        if negative_count == 0
        else "Negative-document metrics unavailable"
    )
    label_evidence = (
        f"{_count(len(progress['label_rows']) - assessed)} labels without recall evidence"
        if progress["label_rows"]
        else "Per-label evidence unavailable"
    )
    return (
        f'<div class="benchmark-hero {status}"><div><p class="eyebrow">Recorded benchmark result</p>'
        f'<h3 id="run-title-{index}">{headline}</h3>'
        f'<p class="evidence-line">{_text(run.get("label"))} · {_count(run.get("documents"))} notes{date_line}</p>'
        f'<p class="run-note">{_text(run.get("note") or "Saved evaluation; not a live service measurement.")}</p>'
        '</div><div class="gate-tally">' + count_line + "</div></div>"
        '<div class="readiness-cards">'
        '<div class="readiness-card privacy-card"><span class="small-label">Notes with uncovered annotated PHI</span>'
        f"<strong>{_count(progress['leaking_documents'])}</strong>"
        f"<span>of {_count(run.get('documents'))} notes · {_percent(progress['leaking_rate'])}</span>"
        "<small>Coverage diagnostic · acceptance target not set</small></div>"
        '<div class="readiness-card"><span class="small-label">PHI character recall</span>'
        f"<strong>{_target_rate(character_rate, target)}</strong><span>Target {_percent(target)}</span>"
        f"<small>Shortfall: {_gap(char_gap)}</small></div>"
        '<div class="readiness-card"><span class="small-label">Labels below recall target</span>'
        f"<strong>{labels_value}</strong><span>{labels_note}</span>"
        f"<small>{label_evidence}</small></div>"
        '<div class="readiness-card"><span class="small-label">Negative notes untouched</span>'
        f"<strong>{negative_value}</strong><span>{negative_note}</span>"
        "<small>False-positive diagnostic · acceptance target not set</small></div></div>"
    )


def _benchmark_gates(progress: dict) -> str:
    order = {
        name: i
        for i, name in enumerate(
            (
                "typed_span_recall_at_least_99_percent",
                "phi_character_recall_at_least_99_percent",
                "every_label_recall_at_least_99_percent",
                "recall_wilson_lower_95_at_least_99_percent",
                "typed_span_precision_at_least_99_percent",
                "typed_span_f1_at_least_99_percent",
                "all_labels_have_sufficient_support",
            )
        )
    }
    rows = []
    for gate in sorted(progress["gates"], key=lambda item: order.get(item["name"], 99)):
        gate_label = gate["label"].removesuffix(" ≥ 99%")
        if gate["name"] == "every_label_recall_at_least_99_percent":
            gate_label = "Lowest label recall"
            if progress["weakest_label"]:
                gate_label += f" ({progress['weakest_label']})"
        elif gate["name"] == "all_labels_have_sufficient_support":
            gate_label = "Smallest label sample"
        is_rate = gate["unit"] == "rate"
        actual = (
            _target_bar(gate["actual"], gate["target"], gate["label"])
            if is_rate
            else _count(gate["actual"])
        )
        target = (
            _percent(gate["target"])
            if is_rate
            else _count(gate["target"]) + " spans / label"
        )
        status = (
            _badge("Met", "teal")
            if gate["passed"] is True
            else (
                _badge("Not met", "amber")
                if gate["passed"] is False
                else _badge("Unavailable")
            )
        )
        rows.append(
            f'<tr><th scope="row">{_text(gate_label)}</th><td>{actual}</td>'
            f'<td>{target}</td><td class="numeric">{_gap(gate["gap"], gate["unit"])}</td><td>{status}</td></tr>'
        )
    return (
        '<section class="subsection benchmark-gates"><div class="subsection-heading"><div>'
        '<p class="eyebrow">01 / Distance to the benchmark</p><h4>Every required check must pass</h4></div></div>'
        '<p class="muted">Shortfall is measured in percentage points (pp), or spans for support. '
        "Bars run from 0 to 100%; the dark marker is the target. A check count is not a percentage of deployment readiness.</p>"
        '<div class="table-scroll" tabindex="0" role="region" aria-label="Benchmark targets and shortfalls">'
        '<table class="gates-table"><caption class="sr-only">Recorded benchmark results against every required target</caption>'
        '<thead><tr><th scope="col">Required check</th><th scope="col">Observed</th><th scope="col">Target</th>'
        '<th scope="col">Shortfall</th><th scope="col">Status</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def _label_table(run: dict, progress: dict) -> str:
    rows = []
    source_rows = {item["label"]: item for item in run.get("per_label") or []}
    target = (run.get("goal") or {}).get("threshold")
    for item in progress["label_rows"]:
        support = source_rows[item["label"]].get("sufficient_support")
        badge = (
            _badge("Below required support", "amber")
            if support is False
            else _badge("Support sufficient", "neutral")
            if support is True
            else '<span class="muted">Not assessed</span>'
        )
        rows.append(
            f'<tr><th scope="row"><code>{_text(item.get("label"))}</code></th>'
            f"<td>{_target_bar(item['recall'], target, item['label'] + ' recall')}</td>"
            f'<td class="numeric">{_percent(target)}</td>'
            f'<td class="numeric">{_gap(item["gap"])}</td>'
            f'<td class="numeric">{_count(item.get("support"))}</td>'
            f'<td class="numeric">{_count(item.get("missed"))}</td>'
            f"<td>{badge}</td></tr>"
        )
    if not rows:
        return '<p class="empty">Per-label metrics are unavailable for this run.</p>'
    return (
        '<div class="table-scroll" tabindex="0" role="region" aria-label="Per-label quality metrics">'
        '<table class="labels-table"><caption class="sr-only">Per-label recall targets and shortfalls, weakest observed recall first.</caption>'
        '<thead><tr><th scope="col">PHI label</th><th scope="col">Recall · 0–100%</th>'
        '<th scope="col">Target</th><th scope="col">Shortfall</th><th scope="col">Gold spans</th>'
        '<th scope="col">Total missed</th><th scope="col">Support check</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _timing(run: dict) -> str:
    timing = run.get("timing") or {}
    performed = timing.get("neural_inference_performed")
    if performed is False:
        note = "Cached timing composition: neural inference was not performed in this evaluation. The recorded total and throughput combine cached neural timing with this rules pass; they are not measurements of one end-to-end run."
    elif performed is True:
        note = "Neural inference was performed for this evaluation. Throughput is the recorded offline evaluation rate; it does not measure live service latency or concurrent capacity."
    else:
        note = "Whether neural inference was performed is unavailable. Treat these values as recorded offline timings, not a measured live-service performance claim."
    speed = _number(timing.get("documents_per_second"))
    speed_text = f"{speed:,.3f}" if speed is not None and speed >= 0 else "Unavailable"
    neural, rules, total = (
        _number(timing.get(key))
        for key in ("neural_seconds", "rules_seconds", "total_seconds")
    )
    composition = ""
    if (
        neural is not None
        and rules is not None
        and total is not None
        and neural >= 0
        and rules >= 0
        and total > 0
        and neural + rules > 0
    ):
        neural_share = 100 * neural / (neural + rules)
        rules_share = 100 - neural_share
        aria_label = f"Recorded stage composition: neural {_seconds(neural)}, rules {_seconds(rules)}. Shares of the neural and rules component sum."
        composition = (
            '<div class="stage-composition"><span class="small-label">Stage shares of neural + rules time</span>'
            f'<div class="stage-strip" role="img" aria-label="{_text(aria_label)}">'
            f'<span class="neural-stage" style="width:{neural_share:.9f}%"></span>'
            f'<span class="rules-stage" style="width:{rules_share:.9f}%"></span></div>'
            '<div class="stage-legend">'
            f'<span><i class="neural-stage" aria-hidden="true"></i>Neural · {_seconds(neural)}</span>'
            f'<span><i class="rules-stage" aria-hidden="true"></i>Rules · {_seconds(rules)}</span>'
            "</div></div>"
        )
    return (
        '<div class="subsection"><h4>Recorded performance</h4><div class="timing-grid">'
        f'<div class="throughput"><span class="small-label">Recorded documents / second</span><strong>{speed_text}</strong><span>offline evaluation rate</span></div>'
        '<dl class="timing-list">'
        f"<div><dt>Neural stage</dt><dd>{_seconds(timing.get('neural_seconds'))}</dd></div>"
        f"<div><dt>Rules stage</dt><dd>{_seconds(timing.get('rules_seconds'))}</dd></div>"
        f"<div><dt>Recorded component sum</dt><dd>{_seconds(timing.get('total_seconds'))}</dd></div>"
        f"<div><dt>Device</dt><dd>{_text(timing.get('device'))}</dd></div>"
        "</dl></div>"
        + composition
        + '<p class="timing-note">'
        + _text(note)
        + "</p></div>"
    )


def _provenance(run: dict) -> str:
    provenance = run.get("provenance") or {}
    fields = (
        ("Source report", "source_file"),
        ("Report selector", "report_selector"),
        ("Report SHA-256", "source_sha256"),
        ("Adapter SHA-256", "adapter_sha256"),
        ("Configuration SHA-256", "config_sha256"),
        ("Corpus SHA-256", "corpus_sha256"),
        ("Neural SHA-256", "neural_sha256"),
        ("Detector SHA-256", "detector_sha256"),
        ("Rules SHA-256", "rules_sha256"),
        ("Completed at", "completed_at"),
    )
    rows = [
        f"<div><dt>{name}</dt><dd><code>{_text(provenance.get(key))}</code></dd></div>"
        for name, key in fields
    ]
    for name, key in (
        ("Independent final", "independent_final"),
        ("Used for checkpoint selection", "used_for_checkpoint_selection"),
        ("Used for training", "used_for_training"),
        ("Qualified in source", "qualified"),
    ):
        if key == "qualified" and run.get("backend") == "model":
            name = "Hybrid qualified in source"
        rows.append(
            f"<div><dt>{name}</dt><dd>{_boolean(provenance.get(key))}</dd></div>"
        )
    return (
        '<details class="source-details"><summary>Source details &amp; artifact hashes</summary><dl class="source-list">'
        + "".join(rows)
        + "</dl></details>"
    )


def _run_panel(run: dict, index: int, selected: int) -> str:
    progress = benchmark_progress(run)
    provenance = run.get("provenance") or {}
    adapter = provenance.get("adapter_sha256")
    corpus = provenance.get("corpus_sha256")
    return (
        f'<article class="run-panel" id="run-panel-{index}" data-run-index="{index}"'
        + (" hidden" if index != selected else "")
        + f' aria-labelledby="run-title-{index}">'
        + _progress_overview(run, progress, index)
        + '<p class="artifact-line">Recorded adapter <code>'
        + _text(adapter[:12] if adapter else None)
        + "</code> · Corpus <code>"
        + _text(corpus[:12] if corpus else None)
        + "</code> · Coverage is measured against annotated PHI, before replacement.</p>"
        + _benchmark_gates(progress)
        + '<section class="subsection"><div class="subsection-heading"><div>'
        '<p class="eyebrow">02 / Recall blockers</p><h4>Which identifiers still miss the target?</h4></div></div>'
        '<p class="muted">Weakest observed recall first. Total missed spans are exact boundary/type mismatches; '
        "they are not a count of exposed identifiers. Zero-support labels are unassessed.</p>"
        + _label_table(run, progress)
        + '</section><section class="use-boundary"><h4>What remains beyond this benchmark?</h4>'
        "<p>Passing these checks establishes performance on the recorded evaluation. "
        "It does not establish readiness on real clinical records or verify the final redacted or replaced output.</p>"
        "<p>The current protocol has no acceptance threshold for document leakage or negative-note preservation. "
        "Those decisions, representative clinical validation, and output-level checks remain separate. "
        "Per-family results are not included in this view.</p></section>"
        '<details class="supporting-details"><summary>Additional quality metrics &amp; error counts</summary><div class="metric-cards">'
        + _metric_card("Exact typed spans", run.get("typed_span") or {})
        + _metric_card("PHI character coverage", run.get("character") or {})
        + '</div></details><details class="supporting-details"><summary>Recorded performance &amp; stage timing</summary>'
        + _timing(run)
        + "</details>"
        + _provenance(run)
        + "</article>"
    )


_STYLE = """
:root{color-scheme:light;--ink:#172b40;--muted:#54667b;--line:#d9e4ed;--blue:#2167ae;--teal:#087f83;--amber:#845408;--paper:#fff;--canvas:#f2f6fa}
*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,select{font:inherit}button,select,summary{cursor:pointer}button:focus-visible,select:focus-visible,summary:focus-visible,[tabindex]:focus-visible{outline:3px solid #277db9;outline-offset:4px}button{border:1px solid #b8cddd;background:white;color:#1c527e;border-radius:8px;padding:9px 14px;font-weight:650}button:hover{background:#edf5fb}main{max-width:1240px;margin:0 auto;padding:44px 32px 36px}h1,h2,h3,h4,p{margin:0}h1{font-size:clamp(30px,4vw,46px);line-height:1.12;letter-spacing:-1.5px;max-width:900px;font-weight:720}h2{font-size:23px;letter-spacing:-.5px}h3{font-size:25px;letter-spacing:-.55px;line-height:1.3}h4{font-size:16px;font-weight:700}p+p{margin-top:9px}.eyebrow{text-transform:uppercase;letter-spacing:1.7px;font-size:10px;font-weight:800;color:var(--teal);margin-bottom:10px}.intro{position:relative;padding-bottom:28px}.intro-top{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:20px}.intro-top .eyebrow{margin:0}.description{max-width:860px;color:var(--muted);font-size:16px;margin:17px 0}.meta{color:var(--muted);font-size:12px}.candidate{display:flex;flex-wrap:wrap;gap:10px 26px;align-items:center;margin-top:19px;border-top:1px solid var(--line);padding-top:17px}.candidate .small-label{display:inline;margin-right:9px}.candidate-name{font-weight:650}.candidate details{flex-basis:100%;color:var(--muted);font-size:12px}.candidate details code{overflow-wrap:anywhere}.notes{padding:16px 20px;background:#e9f2f8;border-left:3px solid var(--blue);border-radius:0 8px 8px 0;margin-bottom:22px;font-size:13px}.notes.warning{background:#fff4df;border-color:#b47712}.notes p+p{margin-top:7px}.panel{background:var(--paper);border:1px solid var(--line);border-radius:14px;margin-bottom:24px;box-shadow:0 4px 20px #172b4004;overflow:hidden}.section-heading{padding:25px 27px 21px;display:flex;gap:16px;align-items:flex-end;justify-content:space-between}.section-heading p{color:var(--muted);font-size:13px;margin-top:5px}.scale-key{white-space:nowrap;font-size:11px;color:var(--muted);display:flex;align-items:center;gap:8px}.scale-key i{display:inline-block;width:48px;height:4px;background:linear-gradient(90deg,#d8e7ef,var(--teal));border-radius:4px}.table-scroll{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left;font-variant-numeric:tabular-nums}th,td{padding:17px 17px;vertical-align:middle;border-top:1px solid #e4ebf1}thead th{background:#f7f9fc;font-size:11px;color:var(--muted);font-weight:650;text-transform:uppercase;letter-spacing:.55px}tbody th{font-weight:600}tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}.comparison{min-width:880px;table-layout:fixed}.comparison th:first-child{width:26%;padding-left:27px}.comparison td{width:14.8%}.selected-row{background:#f1f8fb}.selected-row>th{box-shadow:inset 3px 0 var(--teal)}.run-name{display:block;font-size:13px;overflow-wrap:anywhere}.row-subtitle{display:block;font-size:11px;font-weight:400;color:var(--muted);margin-top:3px}.metric{min-width:84px;max-width:190px}.metric-value{display:block;font-size:14px;font-weight:680;color:#205172;font-variant-numeric:tabular-nums;white-space:nowrap}.bar{display:block;height:5px;background:#e0eaf0;border-radius:5px;overflow:hidden;margin-top:7px}.bar>span{display:block;height:100%;background:var(--teal);border-radius:5px}.metric-detail{display:block;font-size:10px;color:var(--muted);margin-top:6px}.comparison-foot{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding:14px 27px;background:#fafcfe}.selector{display:flex;align-items:center;gap:12px;margin-left:auto}.selector label{color:var(--muted);font-size:12px;white-space:nowrap}.selector select{max-width:430px;min-width:190px;border:1px solid #b5c9d9;border-radius:8px;background:#fff;color:var(--ink);padding:9px 34px 9px 12px;font-size:13px}.run-panel{padding:25px 27px;border-top:1px solid var(--line)}.run-panel[hidden]{display:none}.run-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}.badges{display:flex;flex-wrap:wrap;gap:6px;justify-content:flex-end}.badge{display:inline-block;font-size:10px;line-height:1.45;font-weight:700;letter-spacing:.1px;border-radius:5px;padding:5px 8px;background:#edf1f5;color:#46576b;white-space:normal}.badge.teal{background:#e1f3ee;color:#176659}.badge.amber{background:#fff0d3;color:var(--amber)}.run-note{color:var(--muted);font-size:13px;max-width:980px;margin-top:14px}.metric-cards{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:24px}.metric-card{border:1px solid var(--line);border-radius:10px;padding:20px;background:#fcfdff}.metric-card h4{font-size:14px;margin-bottom:15px}.triple{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}.small-label{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.65px;font-weight:650}.triple strong{display:block;font-size:clamp(16px,2vw,24px);letter-spacing:-.5px;line-height:1.4;font-variant-numeric:tabular-nums}.count-line{display:flex;flex-wrap:wrap;gap:8px 18px;border-top:1px solid var(--line);padding-top:12px;margin-top:16px;font-size:12px;font-variant-numeric:tabular-nums}.support{font-size:11px;color:var(--muted);margin-top:6px}.document-summary{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}.document-summary>div{padding:3px 2px}.document-summary strong{display:block;font-size:21px;margin-top:3px;font-variant-numeric:tabular-nums}.document-summary>div>span:last-child{font-size:12px;color:var(--muted);display:block;margin-top:2px}.subsection{border-top:1px solid var(--line);margin-top:26px;padding-top:23px}.subsection-heading{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:15px}.labels-table{min-width:770px;font-size:12px}.labels-table th,.labels-table td{padding:12px 12px}.labels-table th:first-child{padding-left:0;width:14%}.labels-table td:nth-child(2){width:22%}.labels-table thead th{background:#fff}.labels-table code{font-size:12px;font-weight:650}.labels-table .metric-value{font-size:12px}.numeric{text-align:right;white-space:nowrap}.muted{color:var(--muted)}.small{font-size:11px}.subsection>p{font-size:12px;margin:9px 0}.checks{list-style:none;display:grid;grid-template-columns:1fr 1fr;padding:0;gap:0 24px;margin:14px 0 0}.checks li{display:flex;align-items:center;justify-content:space-between;gap:14px;border-bottom:1px solid #e8eef3;padding:11px 0;font-size:12px}.checks small{display:block;color:var(--muted);font-size:11px;overflow-wrap:anywhere}.checks .badge{flex-shrink:0}.timing-grid{display:grid;grid-template-columns:1fr 1fr;gap:30px;margin-top:15px}.throughput{background:#ecf6f7;border-radius:9px;padding:19px 20px;display:flex;flex-direction:column;justify-content:center}.throughput strong{font-size:32px;letter-spacing:-1px;line-height:1.3;color:#126b73;font-variant-numeric:tabular-nums}.throughput>span:last-child{font-size:12px;color:var(--muted)}dl{margin:0}.timing-list>div{display:flex;justify-content:space-between;gap:20px;padding:8px 0;border-bottom:1px solid #e7edf2;font-size:12px}.timing-list dt{color:var(--muted)}dd{margin:0}.timing-list dd{font-variant-numeric:tabular-nums;font-weight:600;text-align:right;overflow-wrap:anywhere}.timing-note{color:var(--muted);max-width:900px;line-height:1.6}.source-details{margin-top:24px;background:#f7f9fc;border:1px solid var(--line);border-radius:8px;padding:0 16px}summary{font-size:12px;font-weight:650;padding:13px 0}.source-list{padding-bottom:14px}.source-list>div{display:grid;grid-template-columns:190px 1fr;gap:16px;padding:8px 0;border-top:1px solid #e0e8ef;font-size:11px}.source-list dt{color:var(--muted)}.source-list code{font-size:11px;overflow-wrap:anywhere;white-space:normal}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.definitions{padding:0 27px 8px}.definitions summary{font-size:15px;padding:21px 0}.definition-list{display:grid;grid-template-columns:1fr 1fr;gap:18px 28px;padding:0 0 18px}.definition-list dt{font-size:12px;font-weight:700;margin-bottom:3px}.definition-list dd{font-size:12px;color:var(--muted)}.empty{padding:22px 27px;color:var(--muted);font-size:13px}.subsection .empty{padding:10px 0}footer{color:var(--muted);font-size:11px;display:flex;gap:12px;justify-content:space-between;padding:0 2px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}noscript{display:block;background:#fff4df;padding:14px 18px;margin:0 27px 20px;font-size:12px}
.stage-composition{margin-top:18px}.stage-strip{display:flex;height:12px;overflow:hidden;border-radius:5px;margin:8px 0;background:#e0eaf0}.stage-strip>span{height:100%}.neural-stage{background:var(--blue)}.rules-stage{background:var(--teal)}.stage-legend{display:flex;flex-wrap:wrap;gap:10px 22px;color:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}.stage-legend>span{display:flex;align-items:center;gap:6px}.stage-legend i{width:8px;height:8px;border-radius:2px}.stage-strip,.stage-legend i{-webkit-print-color-adjust:exact;print-color-adjust:exact}
@media(max-width:760px){main{padding:25px 16px}h1{letter-spacing:-.8px}.intro-top{align-items:flex-start}.intro-top button{font-size:11px;padding:7px 9px}.section-heading{padding:20px;align-items:flex-start;flex-direction:column}.run-panel{padding:20px}.run-heading{flex-direction:column;gap:12px}.badges{justify-content:flex-start}.selector{margin:0;flex-wrap:wrap;width:100%}.selector select{width:100%;max-width:100%;min-width:0}.metric-cards{grid-template-columns:1fr}.triple strong{font-size:23px}.checks{grid-template-columns:1fr}.timing-grid{grid-template-columns:1fr;gap:14px}.source-list>div{grid-template-columns:1fr;gap:4px}.definitions{padding:0 20px 8px}.definition-list{grid-template-columns:1fr}.comparison-foot{padding:14px 20px}.document-summary{gap:20px}.document-summary strong{font-size:18px}.subsection-heading{align-items:flex-start}.comparison th:first-child{padding-left:20px}.candidate{gap:8px 18px}.scale-key{display:none}footer{flex-direction:column}.panel{border-radius:11px}}
@media print{body{background:white;font-size:10pt}main{max-width:none;padding:0}h1{font-size:25pt}.panel{box-shadow:none;border-radius:0;overflow:visible}.intro-top button,.selector,.scale-key{display:none}.run-panel[hidden]{display:block!important}.run-panel{break-before:page;padding:16px 0;border:0}.section-heading{padding:14px 0}.comparison{min-width:0;table-layout:auto}.table-scroll{overflow:visible}.comparison th:first-child{width:24%;padding-left:6px}th,td{padding:9px 6px}.metric{min-width:0}.metric-value{font-size:9pt}.row-subtitle{font-size:8pt}.bar{-webkit-print-color-adjust:exact;print-color-adjust:exact}.labels-table{min-width:0}.labels-table th,.labels-table td{padding:8px}.metric-card,.document-summary,.timing-grid,.checks li{break-inside:avoid}.source-details{background:white}.source-details>*,.definitions>*{display:block!important}details>summary{list-style:none}.definitions{padding:0}.definition-list{display:grid!important}.source-list>div{break-inside:avoid}.run-heading{align-items:flex-start}.metric-cards{gap:12px}.notes{break-inside:avoid}footer{margin-top:15px}a{color:inherit}.badges .badge{border:1px solid #ccd5dc}.intro{padding-bottom:15px}}
"""


_PROGRESS_STYLE = """
main{padding-top:28px}h1{font-size:34px;letter-spacing:-1px}.intro{padding-bottom:20px}.intro-top{margin-bottom:12px}.description{margin:10px 0;font-size:14px}.intro .meta{font-size:11px}.section-heading{padding:20px 27px}.section-heading h2{font-size:19px}.run-panel{padding-top:22px}
.benchmark-hero{display:flex;align-items:center;justify-content:space-between;gap:28px;border-left:4px solid #b47712;padding:4px 0 4px 20px}.benchmark-hero.met{border-color:var(--teal)}.benchmark-hero.unavailable{border-color:#778699}.benchmark-hero h3{font-size:29px;line-height:1.2}.benchmark-hero .eyebrow{color:var(--amber);margin-bottom:7px}.benchmark-hero.met .eyebrow{color:var(--teal)}.benchmark-hero .run-note{margin-top:7px;font-size:12px;max-width:750px}.evidence-line{font-size:13px;margin-top:10px;font-weight:600}.gate-tally{min-width:165px;flex-shrink:0;background:#fff5df;padding:16px 20px;border-radius:10px;text-align:center;display:flex;flex-direction:column}.met .gate-tally{background:#e1f3ee}.unavailable .gate-tally{background:#edf1f5}.gate-tally strong{font-size:42px;line-height:1.15;font-variant-numeric:tabular-nums}.gate-tally strong span{font-size:26px;color:var(--muted);font-weight:500}.gate-tally>span{font-size:11px;color:var(--muted);margin-top:6px}
.readiness-cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:24px}.readiness-card{border:1px solid var(--line);border-radius:9px;background:#fbfdff;padding:15px 16px;display:flex;flex-direction:column;min-width:0}.readiness-card.privacy-card{background:#fff9ef;border-color:#e8d5b4}.readiness-card>.small-label{min-height:32px;font-size:10px;line-height:1.5}.readiness-card strong{font-size:30px;letter-spacing:-.6px;font-variant-numeric:tabular-nums;line-height:1.35;margin:4px 0 6px;overflow-wrap:anywhere}.readiness-card>span:not(.small-label){font-size:11px}.readiness-card small{color:var(--muted);font-size:10px;line-height:1.5;margin-top:8px}.artifact-line{color:var(--muted);font-size:10px;margin:12px 0 0;overflow-wrap:anywhere}.artifact-line code{font-size:10px}
.benchmark-gates{margin-top:22px}.subsection-heading .eyebrow{font-size:10px;margin-bottom:5px}.benchmark-gates>p{margin:10px 0 16px}.gates-table{min-width:730px;font-size:12px}.gates-table th,.gates-table td{padding:13px 12px}.gates-table tbody th{width:34%;font-weight:550}.gates-table td:nth-child(2){width:24%}.gates-table th:first-child{padding-left:0}.gates-table thead th{background:#fff}.gates-table td:last-child{white-space:nowrap}.target-metric{min-width:110px;max-width:220px}.target-metric strong{font-weight:650;font-size:13px;font-variant-numeric:tabular-nums}.target-track{position:relative;display:block;height:7px;border-radius:3px;background:#e7eef3;margin:7px 0 3px}.target-track>span{display:block;height:100%;border-radius:3px}.gap-fill{background:#c18b35}.met-fill{background:var(--teal)}.target-track .target-marker{position:absolute;display:block;top:-3px;height:13px;width:2px;background:#243b52;transform:translateX(-1px)}.target-track{-webkit-print-color-adjust:exact;print-color-adjust:exact}.labels-table td:nth-child(2){width:25%}.labels-table td:last-child{min-width:140px}
.use-boundary{margin-top:25px;padding:18px 20px;border:1px solid var(--line);border-radius:9px;background:#f4f7fb}.use-boundary h4{font-size:14px}.use-boundary p{font-size:12px;color:var(--muted);margin-top:7px}.supporting-details{border-top:1px solid var(--line);margin-top:22px}.supporting-details>summary{font-size:13px;padding:16px 0}.supporting-details>.subsection{border-top:0;padding-top:0;margin-top:0}.supporting-details .metric-cards{margin-top:0}.evidence-context{padding:0 27px}.evidence-context>details>summary{font-size:14px;padding:18px 0}.evidence-context .candidate{margin-top:0;border-top:0;padding:0 0 16px}.evidence-context .notes{margin-bottom:16px}
@media(max-width:1000px){.readiness-cards{grid-template-columns:repeat(2,minmax(0,1fr))}.benchmark-hero h3{font-size:25px}}
@media(max-width:760px){h1{font-size:29px}.benchmark-hero{align-items:flex-start;flex-direction:column;gap:16px;padding-left:14px}.gate-tally{width:100%;flex-direction:row;align-items:center;justify-content:space-between;padding:12px 16px}.gate-tally strong{font-size:32px}.gate-tally>span{margin:0;font-size:12px}.readiness-card{padding:12px}.readiness-card strong{font-size:25px}.readiness-card>.small-label{font-size:9px;min-height:40px}.section-heading{padding:18px 20px}.benchmark-hero h3{font-size:25px}.evidence-context{padding:0 20px}.evidence-line{font-size:12px}.use-boundary{padding:15px}.gates-table th:first-child{padding-left:8px}}
@media print{.benchmark-hero,.readiness-cards,.use-boundary{break-inside:avoid}.readiness-cards{grid-template-columns:repeat(4,1fr)}.gates-table{min-width:0}.gates-table th,.gates-table td{padding:9px 6px}.target-metric{min-width:0}.supporting-details>*{display:block!important}.evidence-context>details>*{display:block!important}.gate-tally{border:1px solid #ddd}.run-panel:first-of-type{break-before:auto}}
"""


_SCRIPT = """
(() => {
  'use strict';
  const select = document.getElementById('run-select');
  const panels = Array.from(document.querySelectorAll('[data-run-index]'));
  const rows = Array.from(document.querySelectorAll('[data-comparison-index]'));
  const status = document.getElementById('selection-status');
  if (select) {
    select.addEventListener('change', () => {
      const index = Number(select.value);
      if (!Number.isInteger(index) || index < 0 || index >= panels.length) return;
      panels.forEach((panel, position) => { panel.hidden = position !== index; });
      rows.forEach((row, position) => { row.classList.toggle('selected-row', position === index); });
      status.textContent = 'Showing details for ' + select.options[select.selectedIndex].textContent;
    });
  }
  document.getElementById('download-snapshot').addEventListener('click', () => {
    const snapshot = JSON.parse(document.getElementById('report-snapshot').textContent);
    const blob = new Blob([JSON.stringify(snapshot, null, 2) + '\\n'], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'evaluation-snapshot.json';
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  window.addEventListener('beforeprint', () => {
    document.querySelectorAll('details').forEach(detail => {
      detail.dataset.preprintOpen = String(detail.open);
      detail.open = true;
    });
  });
  window.addEventListener('afterprint', () => {
    document.querySelectorAll('details').forEach(detail => {
      detail.open = detail.dataset.preprintOpen === 'true';
    });
  });
})();
"""


def render_report(snapshot: dict) -> str:
    """Return a self-contained HTML document from a normalized metric snapshot.

    Missing metrics remain unavailable. Values are shown as recorded; the
    renderer does not infer timing, recompute goals, or establish qualification.
    Every source string is escaped, including the optional downloadable JSON.
    """
    runs = snapshot.get("runs") or []
    selected = next(
        (
            index
            for index, run in enumerate(runs)
            if run.get("id") == snapshot.get("default_run")
        ),
        0,
    )
    title = snapshot.get("title") or "Clinical de-identification evaluation"
    description = (
        snapshot.get("description")
        or "Aggregate quality and performance from recorded offline evaluations."
    )
    generated = snapshot.get("generated_at")
    generated_line = f"Snapshot generated {_text(generated)} · " if generated else ""
    candidate = snapshot.get("candidate")
    candidate_html = ""
    if candidate:
        source = candidate.get("source") or {}
        candidate_html = (
            '<div class="candidate"><span><span class="small-label">Candidate</span>'
            f'<span class="candidate-name">{_text(candidate.get("name"))}</span></span>'
            f'<span><span class="small-label">Epoch</span>{_count(candidate.get("epoch"))}</span>'
            "<details><summary>Candidate provenance</summary>"
            f"<p>Adapter SHA-256: <code>{_text(candidate.get('adapter_sha256'))}</code></p>"
            f"<p>Source: <code>{_text(source.get('file'))}</code></p>"
            f"<p>Source SHA-256: <code>{_text(source.get('sha256'))}</code></p></details></div>"
        )
    notes = snapshot.get("notes") or []
    warnings = snapshot.get("warnings") or []
    notices = (
        (
            '<aside class="notes">'
            + "".join(f"<p>{_text(note)}</p>" for note in notes)
            + "</aside>"
        )
        if notes
        else ""
    )
    if warnings:
        notices += (
            '<aside class="notes warning" aria-label="Snapshot limitations">'
            + "".join(f"<p>{_text(warning)}</p>" for warning in warnings)
            + "</aside>"
        )
    options = "".join(
        f'<option value="{index}"'
        + (" selected" if index == selected else "")
        + f">{_text(run.get('label') or run.get('id') or f'Run {index + 1}')}</option>"
        for index, run in enumerate(runs)
    )
    selector = (
        '<div class="selector"><label for="run-select">Inspect a run</label><select id="run-select">'
        + options
        + "</select></div>"
        if runs
        else ""
    )
    # Escaping the HTML-significant characters also neutralizes </script>,
    # comments, and HTML entities inside an application/json script element.
    embedded = json.dumps(
        snapshot, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )
    embedded = (
        embedded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_text(title)}</title><style>{_STYLE}{_PROGRESS_STYLE}</style>"
        "<noscript><style>.run-panel[hidden]{display:block!important}.selector,#download-snapshot{display:none}</style></noscript>"
        '</head><body><main><header class="intro"><div class="intro-top"><p class="eyebrow">Benchmark progress / saved evidence</p>'
        '<button id="download-snapshot" type="button">Download metric snapshot</button></div>'
        f'<h1>{_text(title)}</h1><p class="description">{_text(description)}</p>'
        f'<p class="meta">{generated_line}{len(runs)} recorded {"run" if len(runs) == 1 else "runs"} · Offline, self-contained report</p>'
        + "</header>"
        + '<section class="panel" aria-labelledby="detail-title"><div class="section-heading"><div>'
        '<h2 id="detail-title">Benchmark progress</h2><p>Select the evidence to assess.</p></div>'
        + selector
        + "</div>"
        + '<p id="selection-status" class="sr-only" role="status" aria-live="polite"></p>'
        + (
            "<noscript>JavaScript is disabled. Details for every run are shown below.</noscript>"
            if runs
            else ""
        )
        + "".join(_run_panel(run, index, selected) for index, run in enumerate(runs))
        + (
            '<p class="empty">There are no run details to display.</p>'
            if not runs
            else ""
        )
        + "</section>"
        + '<section class="panel" aria-labelledby="comparison-title"><div class="section-heading"><div>'
        '<p class="eyebrow">03 / Context</p><h2 id="comparison-title">Development and qualification side by side</h2>'
        "<p>These are distinct evaluations, not a timeline of improvement. Development scores cannot substitute for qualification.</p></div>"
        '<div class="scale-key"><span>0%</span><i aria-hidden="true"></i><span>100%</span></div></div>'
        + _comparison(runs, selected)
        + '<p class="comparison-foot">Every bar uses the full 0–100% scale. Percentages are rounded to two decimal places; exact counts appear in the selected run. N/A indicates a zero denominator.</p></section>'
        + '<section class="panel evidence-context"><details><summary>Evidence scope &amp; selected candidate</summary>'
        + candidate_html
        + notices
        + "</details></section>"
        + '<section class="panel definitions"><details><summary>Metric definitions &amp; interpretation</summary><dl class="definition-list">'
        "<div><dt>Exact typed spans</dt><dd>A match requires the expected entity label and exact start and end boundaries. Matched, extra, and missed counts are the recorded true positives, false positives, and false negatives.</dd></div>"
        "<div><dt>Precision, recall, and F1</dt><dd>Precision measures how many predicted spans are correct. Recall measures how many gold spans are found. F1 balances the two. All values retain the source evaluation’s definition.</dd></div>"
        "<div><dt>PHI character coverage</dt><dd>Character metrics measure overlap with annotated PHI characters, rather than exact typed boundaries. They help expose partial coverage that span-level scores alone can hide.</dd></div>"
        "<div><dt>Leak-free notes &amp; negative documents</dt><dd>A leak-free note has no uncovered annotated PHI characters. This includes PHI-free notes even if they were falsely redacted. A clean negative note has no predicted PHI. Negative-document performance is N/A when no negative notes were evaluated.</dd></div>"
        "<div><dt>Goals &amp; qualification</dt><dd>Goal checks and qualification flags are copied from the source; no status is inferred from high scores. The qualification wrapper’s decision applies to its hybrid run, including when displayed beside model-only metrics. Development and diagnostic results do not establish independent qualification. Historical qualification applies to its recorded artifacts.</dd></div>"
        "<div><dt>Recorded performance &amp; provenance</dt><dd>Timing describes offline evaluation and excludes model loading, HTTP, queueing, replacement, and report I/O. Cached timing composition is not one end-to-end measurement. Artifact hashes identify the recorded inputs and implementation; this report does not verify those artifacts or reproduce their evaluation.</dd></div>"
        "</dl></details></section><footer><span>Aggregate metrics only · No corpus text included</span>"
        "<span>Generated from a normalized, inspectable snapshot</span></footer></main>"
        f'<script id="report-snapshot" type="application/json">{embedded}</script><script>{_SCRIPT}</script>'
        "</body></html>"
    )
