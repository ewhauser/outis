# Documentation

Start with the [project README](../README.md) for installation, usage and
current results.

The [project goal](../GOALS.md) defines what the detector must achieve and how
candidates are compared. Agents should also read [AGENTS.md](../AGENTS.md).

## Using the detector

- [Clinical text API](api.md): requests, policies, masks, structured inputs,
  file processing and WebSockets.
- [Synthetic replacement](replacement.md): secrets, patient anchors, date
  shifting and identity linking.
- [Deployment](deployment.md): local launcher, default model, batch limits
  and security controls.

## Model

- [Model card](../MODEL_CARD.md): architecture, training data, evaluation and
  limitations.
- [Training](training.md): training detector candidates and generating
  synthetic corpora.
- [Annotation policy](annotation-policy-v2.md): the only annotation policy.
  Span conventions for every corpus and bank, including the conventions
  settled by the convention census.
- [Places and institutions addendum](annotation-policy-v2-places.md): part of
  the policy. Real localities, care-site abbreviations and public references.

## Evaluation

- [Evaluation history](evaluation-history.md): every qualification and
  development result, and the HTML report.
- [Evaluation protocol](../reports/evaluation_protocol.md): privacy and
  labelling gates, and blind-bank rules.
- [External verification](external-verification.md): public benchmarks,
  comparisons with other open models, and the rules that keep that data out of
  training.
- [Blind places bank brief](blind-places-bank-brief.md): the isolated
  diagnostic bank for place-recognition work.
- [Care-site and locality bank brief](care-site-locality-bank-brief.md):
  brief for the tk13 training bank, ready to author.
- [Targeted gaps bank brief](targeted-gaps-bank-brief.md): brief for the
  tk14 training bank (history years, pharmacy ZIPs, phone extensions, funeral
  homes).
- [Name mentions bank brief](name-mentions-bank-brief.md): tk15, first-name
  and short name mentions.
- [Transcript bank brief](transcript-bank-brief.md): tk16, phone and
  telehealth transcripts.
- [Held-out development bank brief](heldout-dev-bank-brief.md): a fresh,
  uninspected development bank for the next promotion decision (draft).
- [HTML evaluation reports](html-reports.md): building and reading the report
  page.

## Research records

These pages record active or completed research. They describe plans and
decisions, not supported behaviour.

- [Synthetic data remediation](synthetic-data-remediation.md): the v11 data
  plan.
- [NAME and LOCATION improvement plan](name-location-improvement-plan.md).
- [Mix4 run closure](mix4-run-closure.md).
