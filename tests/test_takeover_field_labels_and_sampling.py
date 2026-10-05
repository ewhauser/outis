"""Synthetic-only tests for MRN field-label derivatives and the mixture sampler."""

import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "scripts/takeover"


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"takeover_{name}", ROOT / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


labels = load("normalize_field_labels")
sampler = load("sample_mixture")


def mrn_row(text, value):
    start = text.index(value)
    return {
        "id": "r1",
        "split": "train",
        "text": text,
        "spans": [{"start": start, "end": start + len(value), "label": "MRN"}],
    }


@pytest.mark.parametrize("caption", ["MRN: ", "MRN ", "MRN# ", "MRN#"])
def test_caption_moves_outside_and_prefix_stays(caption):
    text = f"Chart {caption}QX-0041122 reviewed."
    row, changed = labels.normalize_row(mrn_row(text, f"{caption}QX-0041122"))
    span = row["spans"][0]
    assert changed == 1
    assert text[span["start"] : span["end"]] == "QX-0041122"


def test_other_labels_and_bare_values_are_unchanged():
    text = "MRN QX-0041122 and ID MRN-77 on file."
    row = mrn_row(text, "QX-0041122")
    row["spans"].append(
        {"start": text.index("MRN-77"), "end": text.index("MRN-77") + 6, "label": "ID"}
    )
    out, changed = labels.normalize_row(row)
    assert changed == 0 and out["spans"] == row["spans"]


def test_refuses_non_training_rows():
    row = mrn_row("MRN: QX-1", "MRN: QX-1")
    row["split"] = "validation"
    with pytest.raises(ValueError):
        labels.normalize_row(row)


def test_quotas_favour_small_banks_and_respect_caps():
    plan = sampler.quotas({"big": 10000, "small": 100, "tiny": 5}, 1000)
    assert sum(plan.values()) == 1000
    assert plan["tiny"] == 5
    assert plan["small"] > 100 * 1000 / 10105


def test_select_keeps_pairs_together_and_covers_rare_features():
    rows = []
    for i in range(40):
        rows.append(
            {
                "id": f"a{i}",
                "pair_id": f"p{i}",
                "template_family": "common",
                "spans": [],
            }
        )
        rows.append(
            {
                "id": f"b{i}",
                "pair_id": f"p{i}",
                "template_family": "common",
                "spans": [{"start": 0, "end": 1, "label": "NAME"}],
            }
        )
    rows.append(
        {
            "id": "rare",
            "template_family": "rare",
            "spans": [{"start": 0, "end": 1, "label": "IP"}],
        }
    )
    picked = sampler.select(rows, 7, random.Random(1))
    ids = {r["id"] for r in picked}
    assert "rare" in ids
    assert all((f"a{i}" in ids) == (f"b{i}" in ids) for i in range(40))
    assert len(picked) <= 7
