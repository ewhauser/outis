## physionet_deid: policy-aligned masking (aligned3)

Neutral characters (convention and requirement differences) are excluded for every system; see `score_aligned.py`.

| System | Value-char recall | Non-PHI redaction | Direct masked | NAME masked (doc 95% lb) | Indirect masked | Leak-free PHI docs | Clean hard negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| project | 77.30% | 1.06% | 75.06% | 75.44% (72.17%) | 65.20% | 54.94% | 58.85% |
| meddeid | 66.54% | 3.91% | 71.95% | 72.06% (68.55%) | 26.94% | 39.81% | 43.08% |
| gliner-trained | 71.69% | 1.04% | 69.50% | 70.37% (66.27%) | 59.31% | 49.46% | 40.15% |
| gliner-facility | 75.33% | 1.35% | 69.50% | 70.46% (66.51%) | 65.43% | 51.50% | 32.09% |

| System | NAME | ID | PHONE | EMAIL | ADDRESS | LOCATION | FACILITY | DATE | AGE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| project | 75.44% | 100.00% | 83.67% | 0.00% | 8.33% | 72.34% | 59.73% | 65.96% | 60.00% |
| meddeid | 72.06% | 0.00% | 85.71% | 0.00% | 16.67% | 82.98% | 66.44% | 13.86% | 60.00% |
| gliner-trained | 70.37% | 100.00% | 63.27% | 0.00% | 16.67% | 72.34% | 29.53% | 65.06% | 60.00% |
| gliner-facility | 70.46% | 100.00% | 61.22% | 0.00% | 16.67% | 82.98% | 61.74% | 65.06% | 60.00% |

Span support after neutralisation: NAME 1124, ID 1, PHONE 49, EMAIL 1, ADDRESS 12, LOCATION 47, FACILITY 149, DATE 664, AGE 5. Gold spans with no policy value left: 112.

### Misses: check for convention gaps before blaming a model

Partly covered spans are described by where and what is left uncovered. A shape that recurs for every system (for example `trailing:caps_abbreviation`) is usually a convention the neutral rules miss, not a model error.

| System | Label | Not detected | Partly covered | Most common uncovered shapes |
|---|---|---:|---:|---|
| project | ADDRESS | 9 | 2 | `trailing:other` 1, `leading:lowercase_words` 1 |
| project | AGE | 2 | 0 | – |
| project | DATE | 224 | 2 | `trailing:digits` 2 |
| project | EMAIL | 1 | 0 | – |
| project | FACILITY | 60 | 0 | – |
| project | LOCATION | 13 | 0 | – |
| project | NAME | 272 | 4 | `trailing:lowercase_words` 4 |
| project | PHONE | 8 | 0 | – |
| meddeid | ADDRESS | 9 | 1 | `leading:other` 1 |
| meddeid | AGE | 2 | 0 | – |
| meddeid | DATE | 570 | 2 | `trailing:digits` 2 |
| meddeid | EMAIL | 1 | 0 | – |
| meddeid | FACILITY | 49 | 1 | `leading:title_like_word` 1 |
| meddeid | ID | 1 | 0 | – |
| meddeid | LOCATION | 6 | 2 | `leading:title_like_word` 2 |
| meddeid | NAME | 290 | 24 | `leading:title_like_word` 14, `trailing:other` 8, `trailing:lowercase_words` 5 |
| meddeid | PHONE | 6 | 1 | `trailing:lowercase_words` 1 |
| gliner-trained | ADDRESS | 8 | 2 | `trailing:other` 2 |
| gliner-trained | AGE | 2 | 0 | – |
| gliner-trained | DATE | 228 | 4 | `leading:digits` 2, `trailing:digits` 2, `leading:lowercase_words` 1 |
| gliner-trained | EMAIL | 1 | 0 | – |
| gliner-trained | FACILITY | 88 | 17 | `trailing:other` 9, `leading:title_like_word` 8 |
| gliner-trained | LOCATION | 12 | 1 | `leading:other` 1 |
| gliner-trained | NAME | 317 | 16 | `leading:title_like_word` 9, `trailing:lowercase_words` 4, `leading:lowercase_words` 3 |
| gliner-trained | PHONE | 18 | 0 | – |
| gliner-facility | ADDRESS | 8 | 2 | `trailing:other` 2 |
| gliner-facility | AGE | 2 | 0 | – |
| gliner-facility | DATE | 229 | 3 | `leading:digits` 2, `trailing:digits` 2 |
| gliner-facility | EMAIL | 1 | 0 | – |
| gliner-facility | FACILITY | 50 | 7 | `leading:title_like_word` 6, `trailing:other` 1 |
| gliner-facility | LOCATION | 8 | 0 | – |
| gliner-facility | NAME | 316 | 16 | `leading:title_like_word` 9, `trailing:lowercase_words` 4, `leading:lowercase_words` 3 |
| gliner-facility | PHONE | 18 | 1 | `leading:digits` 1 |

**Possible convention gaps** (same shape for most systems, at least 5 spans each): NAME `leading:title_like_word`.
