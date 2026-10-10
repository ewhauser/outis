# Evaluation methodology

This page describes how the detector is measured and how a new model is
compared with the current one. All results so far come from synthetic or
public benchmark text; none comes from real clinical records.

## The measure that matters: identifiers masked

Privacy risk comes from identifiers left in the output. A record number
labelled as an ID instead of an MRN is still removed, so type and boundary
errors cost consistency of replacement, not privacy.

- **Masked under any label**: every character of a gold span is covered by
  some predicted span, whatever type it was given.
- **Any-label precision**: the share of predicted spans that overlap at least
  one gold PHI character.

Identifiers are grouped by how directly they identify a person:

- **Direct**: NAME, SSN, MRN, ACCOUNT, ID, PHONE, EMAIL, ADDRESS, URL, IP.
- **Indirect**: DATE, AGE (over 89), LOCATION, FACILITY.

## Targets

| Measure | Target |
| --- | --- |
| Direct identifiers masked under any label | at least 99% recall |
| Indirect identifiers masked under any label | at least 97% recall |
| NAME recall, 95% lower confidence bound | at least 98% |
| Precision, any label | at least 90% |
| Exact typed-span F1 | at least 95% |

The targets are floors. Exact typed-span F1 stays a target because consistent
fictional replacement needs the right type and boundaries. Precision is
relaxed because over-masking costs usefulness, not privacy.

Every evaluation report also gives:

- the miss rate for each entity type, and for each miss whether it was a wrong
  type, a different boundary, partial coverage or not detected at all;
- PHI-bearing notes fully covered, and PHI-free notes left untouched;
- results by note type and by author or template family;
- confidence bounds with documents as the sampling unit, because errors
  cluster by template.

The evaluator (`src/outis/evaluation/evaluate.py`) writes all of this
in the `privacy` block of its report.

## Comparing a new model with the current one

A candidate is compared with the current default on the same bank, written by
an author the training data never saw, with the same detector code.

A candidate is rejected if:

- any-label recall on direct identifiers, or on NAME alone, decreases;
- exact typed-span F1 or any-label precision falls more than 0.5 percentage
  points below the current model, or more than 1.0 point below the best value
  recorded for any earlier candidate on that bank;
- it crosses a target floor that the current model met.

A type change on a span that stays masked, or a boundary change on a span
that stays fully covered, is not a regression on its own.

Scores on banks written by the same generators as the training data are not
evidence of generalisation and are reported separately.

## Evaluation banks

- **Held-out synthetic author** (`heldout_dev_v2`): 2,400 notes written by an
  author isolated from the training generators. The main development bank.
- **Blind places bank** (`blind_places_v3`): 2,352 notes testing real city and
  care-site names, and public references (instruments, diseases, institutions,
  states) that must not be tagged. Scored once per model, aggregates only.
- **Public benchmarks**: ASQ-PHI, MedDeID English and Technetium-I, scored
  under a policy-aligned view. See [external verification](external-verification.md).

The evaluation banks are not published, so that they stay unseen by any
training data, including future contributions.

## What this does not establish

Synthetic banks are development evidence only. Nothing here is a measurement
on real clinical notes, and no output of this project should be described as
HIPAA compliant or de-identified under HIPAA on the strength of these numbers.
A deployment needs its own evaluation on a hand-annotated sample of its real
notes and an appropriate privacy determination.
