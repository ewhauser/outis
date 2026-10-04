from collections import defaultdict
from datetime import date
from pathlib import Path

from clinical_deid.data.annotation_policy import (
    annotation_errors,
    canonical_identity_group,
    identity_split,
)

from clinical_deid.data.v13.contexts import (
    FAMILIES,
    NEGATIVE_FAMILIES,
    validate_families,
)
from clinical_deid.data.v13.synthetic import generate


def test_original_family_contract_and_all_families_render():
    validate_families()
    rows = generate(140)
    assert {r["family"] for r in rows} == {
        f.family_id for f in FAMILIES + NEGATIVE_FAMILIES
    }
    assert len(rows) == 175
    assert sum(not r["spans"] for r in rows) == 35


def test_every_occurrence_owns_exact_gold_and_public_stays_outside():
    for row in generate(140):
        assert not annotation_errors(row["text"], row["spans"])
        gold = {(s["start"], s["end"], s["label"]) for s in row["spans"]}
        for o in row["occurrence_ownership"]:
            assert row["text"][o["start"] : o["end"]] == o["rendered_value"]
            if o["gold_label"]:
                assert (o["start"], o["end"], o["gold_label"]) in gold
            else:
                assert not any(
                    s["start"] < o["end"] and o["start"] < s["end"]
                    for s in row["spans"]
                )
        for evidence in row["phenomenon_evidence"].values():
            for e in evidence:
                assert 0 <= e["start"] < e["end"] <= len(row["text"])


def test_all_primary_secondary_and_paired_owners_remain_train():
    pairs = defaultdict(list)
    for row in generate(140):
        groups = {canonical_identity_group(name) for name in row["full_identities"]}
        assert groups == set(row["identity_groups"])
        assert all(identity_split(group) == "train" for group in groups)
        pairs[row["pair_id"]].append(row)
    for unit in pairs.values():
        assert len(unit) in {1, 2}
        assert len({tuple(r["identity_groups"]) for r in unit}) == 1
        if len(unit) == 2:
            assert [r["negative"] for r in unit] == [False, True]


def test_latent_timelines_match_every_rendered_date_and_duration():
    for row in generate(140):
        latent = {
            key: date.fromisoformat(value) for key, value in row["latent_dates"].items()
        }
        for order in row["date_order"]:
            assert (latent[order["after"]] - latent[order["before"]]).days >= order[
                "minimum_days"
            ]
        for o in row["occurrence_ownership"]:
            if o["entity_kind"] == "date":
                dt = latent[o["placeholder"]]
                assert o["rendered_value"] in (str(dt.year), dt.strftime("%B %Y"))
            if o["entity_kind"] == "duration":
                assert (
                    int(o["rendered_value"])
                    == (latent["date_b"] - latent["date_a"]).days
                )


def test_alias_label_follows_metadata_not_public_placeholder_name():
    rows = [r for r in generate(140) if r["family"] == "facility_correspondence_alias"]
    assert rows
    for row in rows:
        sites = [
            o for o in row["occurrence_ownership"] if o["entity_kind"] == "facility"
        ]
        assert {o["placeholder"] for o in sites} == {"care_site", "public_site"}
        assert all(
            o["gold_label"] == "FACILITY" and o["ownership"] == "personal"
            for o in sites
        )
        assert len({o["rendered_value"] for o in sites}) == 2


def test_generation_determinism_and_no_unsupported_priority_expansion():
    first = generate(140)
    assert first == generate(140)
    assert not any(s["label"] == "AGE" for r in first for s in r["spans"])
    assert all(
        o["rendered_value"].isdecimal() or len(o["rendered_value"].split()) == 2
        for r in first
        for o in r["occurrence_ownership"]
        if o["entity_kind"] == "date"
    )


def test_identifier_retention_uses_context_with_distinct_same_shape_values():
    rows = generate(140)
    retained = [r for r in rows if r["retention_family"]]
    assert retained and len(retained) <= len(rows) / 8
    assert len({r["retention_family"] for r in retained}) == 6
    for row in retained:
        identifiers = [
            o for o in row["occurrence_ownership"] if o["entity_kind"] == "identifier"
        ]
        assert sum(o["gold_label"] == "MRN" for o in identifiers) == 1
        assert len({o["rendered_value"] for o in identifiers}) == len(identifiers)
        assert len({len(o["rendered_value"]) for o in identifiers}) == 1
        assert any(o["gold_label"] != "MRN" for o in identifiers)


def test_public_dates_are_not_in_the_future_of_fictional_record_and_scheduled_events_are():
    seen_future = False
    seen_scheduled = False
    for row in generate(140):
        record_date = date.fromisoformat(row["latent_dates"]["fictional_document_date"])
        if record_date.year > 2026:
            seen_future = True
            assert row["temporal_cohort"] == "future_record_year_stress"
        for o in row["occurrence_ownership"]:
            if o["entity_kind"] != "date":
                continue
            moment = date.fromisoformat(row["latent_dates"][o["placeholder"]])
            assert 1970 <= moment.year <= 2060
            if o["ownership"] == "public":
                assert moment <= record_date
            if o["gold_label"] == "DATE" and "booked" in o["semantic_role"]:
                seen_scheduled = True
                assert moment > record_date
    assert seen_future and seen_scheduled


def test_recipient_id_prose_explicitly_says_routing_not_account():
    from clinical_deid.data.v13.contexts import RETENTION_CONTEXTS

    item = next(
        x
        for x in RETENTION_CONTEXTS
        if x["id"] == "retention_recipient_routing_and_patient_chart"
    )
    assert "routing identifier for the secure referral inbox" in item["template"]
    assert "account identifier" not in item["template"]
    assert (
        next(m for m in item["mentions"] if m["owner"] == "recipient")["label"] == "ID"
    )


def test_every_context_pair_contains_an_actual_date_or_site_ownership_flip():
    units = defaultdict(list)
    for row in generate(140):
        units[row["pair_id"]].append(row)
    for rows in units.values():
        if len(rows) != 2:
            continue
        private = {
            (o["entity_kind"], o["rendered_value"])
            for o in rows[0]["occurrence_ownership"]
            if o["ownership"] == "personal" and o["entity_kind"] in {"date", "facility"}
        }
        public = {
            (o["entity_kind"], o["rendered_value"])
            for o in rows[1]["occurrence_ownership"]
            if o["ownership"] == "public" and o["entity_kind"] in {"date", "facility"}
        }
        assert private & public


def test_date_forms_vary_within_every_family_independently_of_month():
    from clinical_deid.data.v13.synthetic import choose_values
    import random

    for f_index, family in enumerate(FAMILIES):
        if not any(m.entity_kind == "date" for m in family.mentions):
            continue
        observed = defaultdict(set)
        month_forms = defaultdict(set)
        for cycle in range(72):
            index = cycle * len(FAMILIES) + f_index
            values, latent, owners, kinds = choose_values(
                index, family, random.Random(index)
            )
            for mention in family.mentions:
                if mention.entity_kind == "date":
                    key = mention.placeholder
                    form = "year" if values[key].isdecimal() else "month_year"
                    observed[key].add(form)
                    month_forms[(key, latent[key].month)].add(form)
        assert all(forms == {"year", "month_year"} for forms in observed.values())
        # The primary date has all12months and both forms in each month;
        # secondary shifted dates are checked at aggregate family level above.
        if "date_a" in observed:
            assert len({month for key, month in month_forms if key == "date_a"}) == 12
            assert all(
                len(forms) == 2
                for (key, month), forms in month_forms.items()
                if key == "date_a"
            )


def test_packaged_payloads_reproduce_and_lineage_changes_only_bookkeeping():
    import hashlib
    import json
    from clinical_deid.data.v13.synthetic import build_payloads
    from clinical_deid.data.v13.replay import build_replay
    import pytest

    root = Path(__file__).resolve().parents[1]
    from clinical_deid.data.v13.replay import SOURCE

    required = [
        SOURCE,
        root / "data" / "v13_focused_train.jsonl",
        root / "data" / "v13_replay_train.jsonl",
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        pytest.skip(f"Gitignored frozen corpora are absent: {', '.join(missing)}")
    for filename, raw in build_payloads().items():
        assert raw == (root / "data" / filename).read_bytes()
        manifest = json.loads(
            (root / "data" / (filename + ".manifest.json")).read_bytes()
        )
        assert manifest["split"] == "train" and manifest["suite"] == "training"
        assert manifest["independent_final"] is False
        assert manifest["used_for_training"] is True
        assert manifest["used_for_checkpoint_selection"] is False
        assert manifest["sha256"] == hashlib.sha256(raw).hexdigest()
        # Source hashes in the manifest record the pre-restructure module
        # layout. Byte-identical payload reproduction above is the live check.
        assert (
            hashlib.sha256(
                (root / manifest["author_declaration"]).read_bytes()
            ).hexdigest()
            == manifest["author_declaration_sha256"]
        )
        assert (
            hashlib.sha256((root / manifest["lineage"]).read_bytes()).hexdigest()
            == manifest["lineage_sha256"]
        )
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == 3000
        assert all(r["split"] == "train" and r["suite"] == "training" for r in rows)
        if filename == "v13_focused_train.jsonl":
            assert all(
                r["dataset_status"] == "TRAIN_ONLY" and "draft_status" not in r
                for r in rows
            )
            lineage = json.loads((root / manifest["lineage"]).read_bytes())
            assert lineage["clinical_text_or_annotations_modified"] is False
            assert len(lineage["rows"]) == 3000
    replay, lines = build_replay()
    assert len(lines) == 3000 and len({r["source_line_1based"] for r in lines}) == 3000
    assert all(
        hashlib.sha256(line).hexdigest() == proof["exact_source_line_sha256"]
        for line, proof in zip(replay.splitlines(keepends=True), lines)
    )


def test_replay_rejects_source_hash_mismatch_before_parsing(tmp_path, monkeypatch):
    import pytest
    from clinical_deid.data.v13 import replay as v13_replay

    source = tmp_path / "untrusted_train.jsonl"
    source.write_bytes(b"this is not JSON and must not be parsed")
    monkeypatch.setattr(v13_replay, "SOURCE", source)
    with pytest.raises(ValueError, match="authorized pinned SHA"):
        v13_replay.load_source()
