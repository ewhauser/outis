# Documentation

Start with the [project README](../README.md) for installation, usage and
current results.

## Using the detector

- [Clinical text API](api.md): requests, policies, masks, structured inputs,
  file processing and WebSockets.
- [Synthetic replacement](replacement.md): secrets, patient anchors, date
  shifting and identity linking.
- [Deployment](deployment.md): local launcher, detector backends, batch limits
  and security controls.

## Model

- [Model card](../MODEL_CARD.md): architecture, training data, evaluation and
  limitations.
- [Training](training.md): reproducing the default adapter and generating
  synthetic corpora.
- [Annotation policy v2](annotation-policy-v2.md): the span conventions for
  development and training corpora.
- [Places and institutions addendum](annotation-policy-v2-places.md): real
  localities, care-site abbreviations and public references.

## Evaluation

- [Evaluation history](evaluation-history.md): every qualification and
  development result, and the HTML report.
- [Evaluation protocol](../reports/evaluation_protocol.md): internal
  qualification gates and blind-bank rules.
- [External verification](external-verification.md): public benchmarks,
  comparisons with other open models, and the rules that keep that data out of
  training.
- [Blind places bank brief](blind-places-bank-brief.md): the isolated
  diagnostic bank for place-recognition work.
- [HTML evaluation reports](html-reports.md): building and reading the report
  page.

## Research records

These pages record active or completed research. They describe plans and
decisions, not supported behaviour.

- [Synthetic data remediation](synthetic-data-remediation.md): the v11 data
  plan.
- [NAME and LOCATION improvement plan](name-location-improvement-plan.md).
- [Mix4 run closure](mix4-run-closure.md).
