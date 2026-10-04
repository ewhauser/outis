"""Original focused partial-DATE and care-FACILITY TRAIN curriculum.

The V13 clean-context prose is CC0-1.0; implementation is Apache-2.0. Canonical
TRAIN identity sampling is reused from V12. No development corpus is read.
Builds 3,000 focused rows and an exact 3,000-row subset of pinned V12 TRAIN.
Use `python -m clinical_deid.data.v13.synthetic --check` to verify installed payloads.
The report packaging script supplies hashed provenance and policy sidecars.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, timedelta
import re
from itertools import product
import hashlib
import json
from pathlib import Path
import random
from string import Formatter

from clinical_deid.data.annotation_policy import POLICY_VERSION, annotation_errors, canonical_identity_group, identity_split
from clinical_deid.data.v12.synthetic import identity
from clinical_deid.data.generators.remediation import get_encoding
from clinical_deid.data.v13.contexts import FAMILIES, NEGATIVE_FAMILIES, CLINICAL_CONTEXT_PARAGRAPHS, RETENTION_CONTEXTS, validate_families
from clinical_deid.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
VERSION = '13.0'
SEED = 13091
YEAR_MIN, YEAR_MAX = 1970, 2060
SITE_FIRST = ('Amaranth', 'Celandine', 'Hyssop', 'Tansy', 'Valerian', 'Saffron', 'Madder', 'Verbena', 'Borage', 'Anise', 'Teasel', 'Caraway')
SITE_SECOND = ('Spindle', 'Lattice', 'Compass', 'Harbor', 'Terrace', 'Orchard', 'Bridge', 'Court', 'Path', 'Crescent', 'Gate', 'Steps')
SITE_KIND = ('Care Centre', 'Rehabilitation Hospital', 'Medical Pavilion', 'Health Campus', 'Recovery House', 'Outpatient Centre')
PLACES = ('Sedge Hollow, Maine', 'Mullein Crossing, Oregon', 'Balsam Reach, Vermont', 'Sorrel Quay, Iowa')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def choose_values(index, family, rng):
    patient, clinician = identity(rng), identity(rng)
    while clinician['group'] == patient['group']:
        clinician = identity(rng)
    # Independently sweep months within each prose family; do not use year as
    # a positive/public label. Both date widths are intentionally partial.
    month = (index // len(FAMILIES)) % 12 + 1
    year = YEAR_MIN + (index * 13 + index // len(FAMILIES)) % (YEAR_MAX - YEAR_MIN + 1)
    a = date(year, month, 1 + index % 10)
    gap = min(14 + index % 62, (date(year, 12, 28) - a).days)
    b = a + timedelta(days=max(1, gap))
    # Factorial form assignments are independently shuffled per family/month.
    # Cycling the eight combinations gives both forms within each month and
    # family, instead of accidentally locking date form to the35-family stride.
    form_rng = random.Random(f'v13-focused-form:{SEED}:{family.family_id}:{month}')
    form_cube = list(product(('year', 'month_year'), repeat=3))
    form_rng.shuffle(form_cube)
    form_values = form_cube[(index // (len(FAMILIES) * 12)) % len(form_cube)]
    kinds = dict(zip(('date_a', 'date_b', 'public_date'), form_values))
    scheduled = any(m.entity_kind == 'date' and m.ownership == 'personal' and re.search(r'\b(?:booked|scheduled|future)\b', m.semantic_role) for m in family.mentions)
    latent = {'date_a': a, 'date_b': b, 'public_date': a if scheduled else rng.choice((a, b))}
    latent['fictional_document_date'] = a if scheduled else b
    for group in family.same_value_groups:
        if group[0] in latent:
            for key in group[1:]:
                latent[key] = latent[group[0]]
                kinds[key] = kinds[group[0]]
    values = {key: value.strftime('%B %Y') if kinds[key] == 'month_year' else str(value.year) for key, value in latent.items() if key in kinds}
    brand = f'{rng.choice(SITE_FIRST)} {rng.choice(SITE_SECOND)}'
    public_brand = f'{rng.choice(SITE_FIRST)} {rng.choice(SITE_SECOND)}'
    while public_brand == brand:
        public_brand = f'{rng.choice(SITE_FIRST)} {rng.choice(SITE_SECOND)}'
    values.update(patient=patient['full'], clinician=clinician['full'], care_site=f'{brand} {rng.choice(SITE_KIND)}',
                  public_site=f'{public_brand} {rng.choice(SITE_KIND)}', place=rng.choice(PLACES), interval_days=str((b - a).days))
    for group in family.same_value_groups:
        for key in group[1:]:
            values[key] = values[group[0]]
    for group in family.same_entity_groups:
        assert group == ('care_site', 'public_site')
        values['public_site'] = brand
        assert values['public_site'] != values['care_site']
    owners = (patient, clinician) if any(m.placeholder == 'clinician' for m in family.mentions) else (patient,)
    return values, latent, owners, kinds


def render(family, values):
    mentions = {(m.placeholder, m.occurrence): m for m in family.mentions}
    count, pieces, spans, occurrences = Counter(), [], [], []
    length = 0
    for literal, key, spec, conversion in Formatter().parse(family.template):
        pieces.append(literal); length += len(literal)
        if key is None:
            continue
        assert not spec and not conversion
        mention = mentions[key, count[key]]; count[key] += 1
        value = values[key]; start = length; pieces.append(value); length += len(value)
        label = mention.target_label
        if mention.ownership == 'personal' and label is None:
            label = {'person': 'NAME', 'place': 'LOCATION'}.get(mention.entity_kind)
        if label is not None:
            assert mention.ownership == 'personal'
            spans.append({'start': start, 'end': length, 'label': label, 'text': value})
        occurrences.append({**asdict(mention), 'start': start, 'end': length, 'rendered_value': value, 'gold_label': label})
    return ''.join(pieces), spans, occurrences



def append_retention(text, spans, occurrences, retention, rng):
    mentions = {m['placeholder']: m for m in retention['mentions']}
    serials = rng.sample(range(10000000, 99999999), len(mentions))
    shape = rng.randrange(3)
    values = {key: (str(serial), f'K-{serial}', f'R/{serial}/7')[shape] for key, serial in zip(mentions, serials)}
    pieces = [text, '\n\n']; length = len(text) + 2
    for literal, key, spec, conversion in Formatter().parse(retention['template']):
        pieces.append(literal); length += len(literal)
        if key is None:
            continue
        mention = mentions[key]; value = values[key]; start = length
        pieces.append(value); length += len(value)
        label = mention['label']
        if label:
            spans.append({'start': start, 'end': length, 'label': label, 'text': value})
        occurrences.append({'placeholder': key, 'occurrence': 0, 'ownership': 'public' if label is None else 'personal',
                            'entity_kind': 'identifier', 'semantic_role': retention['id'], 'target_label': label,
                            'start': start, 'end': length, 'rendered_value': value, 'gold_label': label})
    return ''.join(pieces), spans, occurrences


def position_note(text, spans, occurrences, rng, running, used_texts):
    paragraphs = list(CLINICAL_CONTEXT_PARAGRAPHS)
    rng.shuffle(paragraphs)
    options = []
    for before in range(5):
        for after in range(5):
            if before + after > 6:
                continue
            prefix = '\n\n'.join(paragraphs[:before]) + ('\n\n' if before else '')
            suffix = ('\n\n' if after else '') + '\n\n'.join(paragraphs[before:before + after])
            length = len(prefix) + len(text) + len(suffix)
            if prefix + text + suffix in used_texts:
                continue
            additions = defaultdict(Counter)
            for occurrence in occurrences:
                if occurrence['entity_kind'] in {'date', 'facility'}:
                    key = occurrence['ownership'] + ':' + occurrence['entity_kind']
                    additions[key][min(3, 4 * (len(prefix) + occurrence['start']) // length)] += 1
            imbalance = 0.
            for key, add in additions.items():
                counts = [running[key][q] + add[q] for q in range(4)]
                mean = sum(counts) / 4
                imbalance += sum((n - mean)**2 for n in counts) / max(1, sum(counts))
            options.append((imbalance + .00001 * (before + after), before, after, prefix, suffix, additions))
    _, before, after, prefix, suffix, additions = min(options, key=lambda x: x[:3])
    for key, add in additions.items():
        running[key].update(add)
    shift = len(prefix)
    spans = [{**s, 'start': s['start'] + shift, 'end': s['end'] + shift} for s in spans]
    occurrences = [{**o, 'start': o['start'] + shift, 'end': o['end'] + shift} for o in occurrences]
    return prefix + text + suffix, spans, occurrences, {'leading_paragraphs': before, 'trailing_paragraphs': after}


def make_row(number, family, values, latent, owners, rng, running, pair_id, used_texts, retention=None):
    text, spans, occurrences = render(family, values)
    if retention is not None:
        text, spans, occurrences = append_retention(text, spans, occurrences, retention, rng)
    text, spans, occurrences, placement = position_note(text, spans, occurrences, rng, running, used_texts)
    # Paired negatives retain the positive unit's canonical TRAIN owner groups.
    groups = sorted({owner['group'] for owner in owners})
    evidence = defaultdict(list)
    for o in occurrences:
        key = {'date': 'partial_date' if o['ownership'] == 'personal' else 'public_date',
               'facility': 'care_site' if o['ownership'] == 'personal' else 'public_institution',
               'person': 'personal_name', 'place': 'personal_locality', 'duration': 'clinical_duration',
               'identifier': 'contextual_identifier' if o['ownership'] == 'personal' else 'public_reference_identifier'}[o['entity_kind']]
        evidence[key].append({'start': o['start'], 'end': o['end']})
    for phrase in family.non_entity_phrases:
        start = text.index(phrase)
        assert not any(s['start'] < start + len(phrase) and start < s['end'] for s in spans)
        evidence['generic_service'].append({'start': start, 'end': start + len(phrase)})
    for kind in ('date', 'facility'):
        personal = {o['rendered_value'] for o in occurrences if o['entity_kind'] == kind and o['ownership'] == 'personal'}
        public = {o['rendered_value'] for o in occurrences if o['entity_kind'] == kind and o['ownership'] == 'public'}
        shared = personal & public
        for o in occurrences:
            if o['entity_kind'] == kind and o['rendered_value'] in shared:
                evidence['same_value_different_' + kind + '_ownership'].append({'start': o['start'], 'end': o['end']})
    if family.same_entity_groups:
        evidence['care_site_alias'] = [{'start': o['start'], 'end': o['end']} for o in occurrences if o['entity_kind'] == 'facility' and o['ownership'] == 'personal']
    for order in family.date_order:
        assert (latent[order.after] - latent[order.before]).days >= order.minimum_days
    for gap in family.date_gaps:
        assert (latent[gap.end] - latent[gap.start]).days == int(values[gap.days_placeholder])
    row = {'id': f'v13-focused-train-{SEED}-{number:05d}', 'text': text, 'spans': spans, 'split': 'train',
           'suite': 'training', 'annotation_policy': POLICY_VERSION, 'identity_group': owners[0]['group'],
           'identity_groups': groups, 'full_identity': owners[0]['full'], 'full_identities': sorted({p['full'] for p in owners}),
           'pair_id': pair_id, 'template_family': 'v13_focused.' + family.family_id, 'family': family.family_id,
           'cohort': 'realistic', 'negative': not spans, 'phenomena': sorted(evidence), 'phenomenon_evidence': dict(evidence),
           'focus': family.focus, 'note_form': family.note_form, 'placement': placement,
           'occurrence_ownership': occurrences, 'latent_dates': {key: value.isoformat() for key, value in latent.items()},
           'same_value_groups': family.same_value_groups, 'same_entity_groups': family.same_entity_groups,
           'date_order': [asdict(o) for o in family.date_order], 'date_gaps': [asdict(g) for g in family.date_gaps],
           'retention_family': retention['id'] if retention is not None else None,
           'temporal_cohort': 'future_record_year_stress' if latent['fictional_document_date'].year > 2026 else 'historical_or_current_record_year',
           'scheduled_date_support': any(o['entity_kind'] == 'date' and o['ownership'] == 'personal' and re.search(r'\b(?:booked|scheduled|future)\b', o['semantic_role']) for o in occurrences),
           'dataset_status': 'TRAIN_ONLY', 'draft_source_id': f'v13-focused-draft-{SEED}-{number:05d}'}
    assert not annotation_errors(text, spans)
    assert all(identity_split(group) == 'train' for group in groups)
    assert {canonical_identity_group(n) for n in row['full_identities']} == set(groups)
    used_texts.add(text)
    return row


def generate(positive_count=2400, initial_positions=None):
    assert 0 < positive_count <= 2800 and positive_count % 4 == 0
    validate_families()
    rows, running, used_texts = [], defaultdict(Counter), set()
    for key, counts in (initial_positions or {}).items():
        running[key].update(counts)
    for index in range(positive_count):
        rng = random.Random(f'v13-focused:{SEED}:{index}')
        family = FAMILIES[index % len(FAMILIES)]
        values, latent, owners, kinds = choose_values(index, family, rng)
        pair_id = f'v13-focused-{SEED}-unit-{index:05d}'
        retention = RETENTION_CONTEXTS[(index // 8) % len(RETENTION_CONTEXTS)] if index % 8 == 0 else None
        row = make_row(len(rows), family, values, latent, owners, rng, running, pair_id, used_texts, retention)
        rows.append(row)
        if index % 4 == 0:
            positive_kinds = {o['entity_kind'] for o in row['occurrence_ownership'] if o['ownership'] == 'personal'} & {'date', 'facility'}
            compatible = [f for f in NEGATIVE_FAMILIES if positive_kinds & {m.entity_kind for m in f.mentions if m.ownership == 'public'}]
            negative = compatible[(index // 4) % len(compatible)]
            negative_values = dict(values)
            negative_latent = dict(latent)
            private_dates = [o for o in row['occurrence_ownership'] if o['entity_kind'] == 'date' and o['ownership'] == 'personal']
            private_sites = [o for o in row['occurrence_ownership'] if o['entity_kind'] == 'facility' and o['ownership'] == 'personal']
            if private_dates:
                negative_values['public_date'] = private_dates[0]['rendered_value']
                negative_latent['public_date'] = latent[private_dates[0]['placeholder']]
            if private_sites:
                negative_values['public_site'] = private_sites[0]['rendered_value']
            rows.append(make_row(len(rows), negative, negative_values, negative_latent, owners, rng, running, pair_id, used_texts))
    assert len({r['id'] for r in rows}) == len(rows)
    assert len({r['text'] for r in rows}) == len(rows)
    return rows



def replay_position_counts(payload):
    """Seed clinical mention placement from the approved TRAIN-only replay."""
    counts = defaultdict(Counter)
    for line in payload.splitlines():
        row = json.loads(line)
        if row['split'] != 'train':
            raise ValueError('Replay must contain TRAIN rows only')
        for span in row['spans']:
            if span['label'] in {'DATE', 'FACILITY'}:
                counts['personal:' + span['label'].lower()][min(3, 4 * span['start'] // len(row['text']))] += 1
    return counts


def build_payloads():
    """Return reviewed corpus bytes without writing or running a model."""
    from clinical_deid.data.v13.replay import build_replay
    replay, _ = build_replay()
    rows = generate(initial_positions=replay_position_counts(replay))
    focused = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in rows).encode('utf-8')
    return {'v13_focused_train.jsonl': focused, 'v13_replay_train.jsonl': replay}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', required=True, help='Reproduce and compare installed payload bytes; read-only')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    args = parser.parse_args()
    payloads = build_payloads()
    for name, payload in payloads.items():
        if (args.data_dir/name).read_bytes() != payload:
            raise ValueError(f'Reproduction mismatch: {name}')
    print(json.dumps({name: {'sha256': digest(raw), 'rows': len(raw.splitlines())} for name, raw in payloads.items()}, indent=2))


if __name__ == '__main__':
    main()
