## physionet_deid: label-agnostic

| System | Char recall | Non-PII redaction | Age-adjusted | Exact-span P | Exact-span R | Exact-span F1 | Leak-free PHI docs | Clean hard negatives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| project | 72.96% | 0.88% | 0.87% | 26.25% | 50.37% | 34.52% | 51.45% | 58.79% |
| meddeid | 63.93% | 3.33% | 3.31% | 3.37% | 14.28% | 5.45% | 39.13% | 43.66% |
| gliner-trained | 65.81% | 0.88% | 0.86% | 18.22% | 47.23% | 26.29% | 45.55% | 39.85% |
| gliner-facility | 71.73% | 1.13% | 1.12% | 16.19% | 48.20% | 24.23% | 48.86% | 31.74% |

## physionet_deid: coarse typed exact spans

| System | Typed P | Typed R | Typed F1 | NAME R | DATE R | LOCATION R | ID R | PHONE R | EMAIL R | AGE R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| project | 25.55% | 49.03% | 33.60% | 46.53% | 64.16% | 28.84% | 0.00% | 34.69% | 0.00% | 60.00% |
| meddeid | 3.07% | 12.99% | 4.96% | 10.59% | 7.23% | 29.47% | 0.00% | 40.82% | 0.00% | 0.00% |
| gliner-trained | 17.75% | 46.03% | 25.62% | 42.79% | 61.14% | 27.27% | 0.00% | 38.78% | 0.00% | 60.00% |
| gliner-facility | 15.84% | 47.18% | 23.71% | 42.53% | 60.99% | 36.36% | 0.00% | 38.78% | 0.00% | 60.00% |

Gold support: NAME 1124, DATE 664, LOCATION 319, ID 2, PHONE 49, EMAIL 1, AGE 5.
