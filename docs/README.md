# Documentation

Start with the [project README](../README.md) for installation, usage and
current results.

## Using the detector

- [Clinical text API](api.md): requests, policies, masks, structured inputs,
  file processing and WebSockets.
- [Synthetic replacement](replacement.md): secrets, patient anchors, date
  shifting, identity linking, and patient or study scope.
- [Deployment](deployment.md): local launcher, default model, batch limits
  and security controls.

## Model

- [Model card](../MODEL_CARD.md): architecture, training data, evaluation and
  limitations.
- [Training](training.md): training detector candidates and generating
  synthetic corpora.
- [Annotation policy](annotation-policy-v2.md): span conventions for every
  entity type.
- [Places and institutions addendum](annotation-policy-v2-places.md): real
  localities, care-site abbreviations and public references.

## Evaluation

- [Evaluation methodology](evaluation.md): targets, any-label masking and how
  a new model is compared with the current one.
- [External verification](external-verification.md): public benchmarks,
  comparisons with other open models, and the rules that keep that data out of
  training.
- [HTML evaluation reports](html-reports.md): building and reading the report
  page.
