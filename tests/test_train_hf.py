"""CPU-only checks for corpus guards and interrupted training persistence."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

SPEC = importlib.util.spec_from_file_location(
    "takeover_train_hf", Path(__file__).parents[1] / "scripts/takeover/train_hf.py"
)
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


def write_bank(path, *, split="train", count=1):
    rows = [
        {
            "id": str(i),
            "text": f"Invented note {i}",
            "spans": [],
            "split": split,
            "suite": "development",
            "annotation_policy": "clinical-phi-v2",
        }
        for i in range(count)
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(payload)
    Path(str(path) + ".manifest.json").write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "rows": count,
                "label_support": dict.fromkeys(train.ENTITY_TYPES, 0),
            }
        )
    )
    return rows


@pytest.mark.parametrize(
    "name",
    [
        "blind_qualification_v9_revision3.jsonl",
        "blind_qualification_v6.jsonl",
        "blind_qualification_v7.jsonl",
        "proxyq_a1.jsonl",
        "x.sealed",
        "retired_v8_development_001.jsonl",
        "reports/v15_name_next_step_analysis_001/recovery_data_001/derived.jsonl",
    ],
)
def test_forbidden_training_paths_are_rejected_before_read(tmp_path, monkeypatch, name):
    monkeypatch.setattr(
        train, "load_verified_corpus", lambda _: pytest.fail("bank must not be opened")
    )
    with pytest.raises(SystemExit, match="refusing"):
        train.read_corpus(tmp_path / name, "train")


def test_guard_checks_symlink_target_and_manifest_target(tmp_path, monkeypatch):
    forbidden = tmp_path / "blind_qualification_v9_revision3.jsonl"
    alias = tmp_path / "ordinary.jsonl"
    alias.symlink_to(forbidden)
    monkeypatch.setattr(
        train, "load_verified_corpus", lambda _: pytest.fail("bank must not be opened")
    )
    with pytest.raises(SystemExit, match="refusing"):
        train.read_corpus(alias, "train")
    alias.unlink()
    Path(str(alias) + ".manifest.json").symlink_to(
        Path(str(forbidden) + ".manifest.json")
    )
    with pytest.raises(SystemExit, match="refusing"):
        train.read_corpus(alias, "train")


@pytest.mark.parametrize("split", ["test", "validation", None])
def test_only_train_split_can_train(tmp_path, split):
    path = tmp_path / "invented.jsonl"
    write_bank(path, split=split)
    with pytest.raises(SystemExit, match="only train-split|final/test"):
        train.read_corpus(path, "train")


def test_corpus_requires_frozen_manifest_and_pins_used_bytes(tmp_path):
    path = tmp_path / "invented.jsonl"
    rows = write_bank(path)
    actual, provenance = train.read_corpus(path, "train")
    assert actual == rows
    assert provenance["sha256"] == train.sha256_file(path)
    assert provenance["manifest_sha256"] == train.sha256_file(
        Path(str(path) + ".manifest.json")
    )
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        train.read_corpus(path, "train")


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, **kwargs):
        return SimpleNamespace(loss=self.weight.square())

    def save_pretrained(self, path, **kwargs):
        path.mkdir(parents=True)
        (path / "model.json").write_text(json.dumps({"weight": self.weight.item()}))


class FakeTokenizer:
    pad_token_id = 0

    def save_pretrained(self, path):
        (path / "tokenizer.json").write_text("{}")


def summary(step, best):
    return {
        "last_evaluated_step": step,
        "best": {"step": best, "checkpoint": f"step-{best:06d}"},
        "history": [{"step": i} for i in range(1, step + 1)],
        "corpus_audit": {"test_used_for_training": False},
    }


def test_each_evaluation_persists_summary_and_best_artifacts(tmp_path):
    model = FakeModel()
    train.save_evaluation(
        tmp_path, model, FakeTokenizer(), {"encoder": "fixture"}, summary(1, 1), True
    )
    first = (tmp_path / "hf/model.json").read_bytes()
    model.weight.data.fill_(2.0)
    train.save_evaluation(
        tmp_path, model, FakeTokenizer(), {"encoder": "fixture"}, summary(2, 1), False
    )
    assert (tmp_path / "hf/model.json").read_bytes() == first
    root = json.loads((tmp_path / "training_summary.json").read_text())
    assert root["last_evaluated_step"] == 2
    assert root["best"]["step"] == 1
    assert (
        json.loads((tmp_path / "step-000002/training_summary.json").read_text())[
            "checkpoint"
        ]["step"]
        == 2
    )
    model.weight.data.fill_(3.0)
    train.save_evaluation(
        tmp_path, model, FakeTokenizer(), {"encoder": "fixture"}, summary(3, 3), True
    )
    assert json.loads((tmp_path / "hf/model.json").read_text())["weight"] == 3.0
    assert not (tmp_path / "publication_in_progress.json").exists()


def test_checkpoint_save_interruption_preserves_previous_published_state(
    tmp_path, monkeypatch
):
    model = FakeModel()
    train.save_evaluation(tmp_path, model, FakeTokenizer(), {}, summary(1, 1), True)
    old = (tmp_path / "training_summary.json").read_bytes()
    monkeypatch.setattr(
        model,
        "save_pretrained",
        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        train.save_evaluation(tmp_path, model, FakeTokenizer(), {}, summary(2, 2), True)
    assert (tmp_path / "training_summary.json").read_bytes() == old
    assert (tmp_path / "hf/model.json").is_file()
    assert not (tmp_path / "step-000002").exists()


def test_root_publication_interruption_leaves_recovery_marker_and_checkpoint(
    tmp_path, monkeypatch
):
    model = FakeModel()
    train.save_evaluation(tmp_path, model, FakeTokenizer(), {}, summary(1, 1), True)
    real_replace = train.os.replace

    def interrupted(source, destination):
        if Path(source).name == ".hf.pending":
            raise KeyboardInterrupt
        return real_replace(source, destination)

    monkeypatch.setattr(train.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        train.save_evaluation(tmp_path, model, FakeTokenizer(), {}, summary(2, 2), True)
    marker = json.loads((tmp_path / "publication_in_progress.json").read_text())
    assert marker["step"] == 2
    assert (tmp_path / ".hf.previous/model.json").is_file()
    assert (
        json.loads((tmp_path / "step-000002/training_summary.json").read_text())[
            "best"
        ]["step"]
        == 2
    )


def test_resume_provenance_does_not_invent_clean_ancestry(tmp_path):
    source = tmp_path / "hf"
    source.mkdir()
    (source / "model.json").write_text("{}")
    unknown = train.resume_provenance(str(source))
    assert unknown["test_used_for_training"] is None
    assert unknown["optimizer_restored"] is False
    (tmp_path / "training_summary.json").write_text(
        json.dumps({"corpus_audit": {"test_used_for_training": False}})
    )
    audited = train.resume_provenance(str(source))
    assert audited["test_used_for_training"] is False
    assert audited["training_summary_sha256"] == train.sha256_file(
        tmp_path / "training_summary.json"
    )
    assert audited["artifacts"]["model.json"] == train.sha256_file(
        source / "model.json"
    )


def invoke_training(tmp_path, monkeypatch, *, window_count=1, extra=(), loads=None):
    path = tmp_path / "train.jsonl"
    write_bank(path, count=2)
    model = FakeModel()
    loads = [] if loads is None else loads

    def load_model(*args, **kwargs):
        loads.append(kwargs)
        return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForTokenClassification=SimpleNamespace(from_pretrained=load_model),
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *a, **k: FakeTokenizer()
            ),
        ),
    )
    monkeypatch.setattr(
        train, "build_windows", lambda *a: [([1, 2], [0, 0])] * window_count
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_hf.py",
            "--model-name",
            "public/fixture",
            "--train",
            str(path),
            "--output",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--epochs",
            "1",
            "--eval-every",
            "0.1",
            *extra,
        ],
    )
    train.main()
    return json.loads((tmp_path / "out/training_summary.json").read_text())


def test_main_flushes_partial_accumulation_and_persists_final_summary(
    tmp_path, monkeypatch
):
    result = invoke_training(
        tmp_path, monkeypatch, window_count=3, extra=("--grad-accum", "2")
    )
    assert result["completed_steps"] == result["planned_steps"] == 2
    assert result["status"] == "completed"
    assert (
        result["best"]["step"] == 2
    )  # No selection banks means retain latest weights.
    assert [entry["step"] for entry in result["history"]] == [1, 2]
    assert result["corpus_audit"]["test_used_for_training"] is False
    assert result["corpus_audit"]["train"][0]["rows_used"] == 2


def test_main_loads_base_model_in_fp32(tmp_path, monkeypatch):
    loads = []
    invoke_training(tmp_path, monkeypatch, loads=loads)
    assert [kwargs["dtype"] for kwargs in loads] == [train.torch.float32]


def test_main_records_effective_subset(tmp_path, monkeypatch):
    result = invoke_training(tmp_path, monkeypatch, extra=("--limit-train", "1"))
    audit = result["corpus_audit"]
    assert audit["train"][0]["rows"] == 2
    assert audit["train"][0]["rows_used"] == audit["effective_train_rows"] == 1
    assert len(audit["effective_train_rows_sha256"]) == 64


def test_completed_evaluation_survives_later_training_interrupt(tmp_path, monkeypatch):
    original = FakeModel.forward
    calls = 0

    def interrupt_second(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(self, **kwargs)

    monkeypatch.setattr(FakeModel, "forward", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        invoke_training(tmp_path, monkeypatch, window_count=2)
    result = json.loads((tmp_path / "out/training_summary.json").read_text())
    assert result["status"] == "training"
    assert result["last_evaluated_step"] == result["best"]["step"] == 1
    assert (tmp_path / "out/step-000001/training_summary.json").is_file()


def test_interrupted_root_cannot_be_used_as_resume_source(tmp_path):
    (tmp_path / "hf").mkdir()
    (tmp_path / "publication_in_progress.json").write_text("{}")
    with pytest.raises(SystemExit, match="interrupted publication"):
        train.resume_provenance(str(tmp_path / "hf"))


def test_legacy_annotation_policy_cannot_train(tmp_path):
    path = tmp_path / "legacy.jsonl"
    rows = write_bank(path)
    rows[0]["annotation_policy"] = "clinical-phi-v1"
    payload = json.dumps(rows[0]) + "\n"
    path.write_text(payload)
    sidecar = Path(str(path) + ".manifest.json")
    manifest = json.loads(sidecar.read_text())
    manifest["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="annotation policy"):
        train.read_corpus(path, "train")


@pytest.mark.parametrize(
    "metadata",
    [
        {"suite": "final"},
        {"split": "test"},
        {"independent_final": True},
        {"independent": True},
        {"proxy_role": "aggregate_only"},
        {"is_proxy": True},
        {"role": "blind-proxy"},
        {"bank_role": "sealed_qualification"},
        {"dataset_role": "independent"},
        {"independent_final": "false"},
        {"suite": "FINAL"},
    ],
)
@pytest.mark.parametrize("role", ["train", "selection"])
def test_manifest_roles_rejected_before_corpus_read(
    tmp_path, monkeypatch, metadata, role
):
    path = tmp_path / "ordinary.jsonl"
    Path(str(path) + ".manifest.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(
        train,
        "load_verified_corpus",
        lambda _: pytest.fail("corpus must not be opened"),
    )
    with pytest.raises(SystemExit, match="refusing"):
        train.read_corpus(path, role)


def test_manifest_change_during_load_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "ordinary.jsonl"
    rows = write_bank(path)
    sidecar = Path(str(path) + ".manifest.json")
    declaration = json.loads(sidecar.read_text())
    monkeypatch.setattr(
        train,
        "load_verified_corpus",
        lambda _: (rows, dict(declaration, suite="final")),
    )
    with pytest.raises(SystemExit, match="changed after metadata preflight"):
        train.read_corpus(path, "train")


@pytest.mark.parametrize("separator", [" ", "  ", "\n", "\n  ", "\t", "\u00a0"])
def test_alignment_owns_whitespace_tokens_inside_gold_span(separator):
    text = "Ada" + separator + "Lovelace"
    right = 3 + len(separator)
    tags, stats = train.align(
        text,
        [{"start": 0, "end": len(text), "label": "NAME"}],
        [(0, 3), (3, right), (right, len(text))],
    )
    assert [train.TAGS[tag] for tag in tags] == ["B-NAME", "I-NAME", "E-NAME"]
    assert stats == {
        "conflict_tokens": 0,
        "misaligned_spans": 0,
        "unrepresented_spans": 0,
    }


def test_alignment_does_not_own_outside_or_between_entity_whitespace():
    text = "  Ada  Lovelace  "
    tags, _ = train.align(
        text,
        [
            {"start": 2, "end": 5, "label": "NAME"},
            {"start": 7, "end": 15, "label": "NAME"},
        ],
        [(0, 2), (2, 5), (5, 7), (7, 15), (15, 17)],
    )
    assert [train.TAGS[tag] for tag in tags] == ["O", "S-NAME", "O", "S-NAME", "O"]


def test_whitespace_fallback_does_not_capture_nonphi_text_or_straddle_gold_boundary():
    # Deliberately include a gold trailing whitespace character to exercise the
    # guard independently of corpus-policy validation.
    text = "Ada  note"
    tags, _ = train.align(
        text, [{"start": 0, "end": 4, "label": "NAME"}], [(0, 3), (3, 5), (5, 9)]
    )
    assert [train.TAGS[tag] for tag in tags] == ["S-NAME", "O", "O"]
    tags, _ = train.align(
        text, [{"start": 0, "end": 5, "label": "NAME"}], [(0, 3), (3, 9)]
    )
    assert [train.TAGS[tag] for tag in tags] == ["S-NAME", "O"]


def selection_report(direct, indirect, precision, typed_p=0.9, typed_r=0.9):
    return {
        "bank": {
            "model": {
                "typed_span": {"precision": typed_p, "recall": typed_r},
                "privacy": {
                    "direct_recall": direct,
                    "indirect_recall": indirect,
                    "name_recall": direct,
                    "any_label_precision": precision,
                },
            }
        }
    }


def test_privacy_selection_prefers_direct_recall_over_typed_spans():
    better_typed = selection_report(0.9980, 0.9999, 0.99, typed_p=0.99, typed_r=0.99)
    better_direct = selection_report(0.9990, 0.9990, 0.98, typed_p=0.90, typed_r=0.90)
    assert train.selection_key(better_direct) > train.selection_key(better_typed)
    assert train.selection_key(better_typed, "typed_min_pr") > train.selection_key(
        better_direct, "typed_min_pr"
    )


def test_privacy_selection_ignores_sub_hundredth_point_recall_noise():
    noisy = selection_report(0.99851, 0.9990, 0.97)
    precise = selection_report(0.99849, 0.9990, 0.98)
    assert train.selection_key(precise) > train.selection_key(noisy)


def test_privacy_selection_skips_banks_without_a_measure():
    report = selection_report(0.999, None, 0.99)
    assert train.selection_key(report)[1] == 0.0
