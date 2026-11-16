# Final numeric checks (existing outputs only)

## F1 Following the new better advisor after A became worse (share, ties half)
A's lead at onset: {'SV t=41': [24], 'SV t=61': [24], 'VS t=11': [6], 'VS t=31': [6]}

### trial10
| Condition | all | volatile-first (lead 6) | stable-first (lead 24) | t=11 | t=31 | t=41 | t=61 | stable-volatile [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Raw | 35% (31-43) | 42% (35-48) | 29% (18-43) | 47% (35-56) | 38% (25-48) | 31% (15-50) | 27% (21-35) | -13.3 [-22.1, -4.6] pp |
| RawCorrectness | 35% (31-43) | 47% (44-52) | 24% (19-38) | 52% (40-58) | 42% (29-48) | 25% (15-44) | 22% (19-31) | -22.9 [-32.7, -12.9] pp |
| CorrectnessOnly | 28% (16-40) | 46% (26-62) | 10% (0-24) | 47% (25-73) | 45% (27-65) | 10% (0-25) | 10% (0-31) | -36.3 [-43.5, -29.2] pp |
| Summary | 15% (7-24) | 27% (15-48) | 2% (0-6) | 32% (21-52) | 22% (8-44) | 2% (0-6) | 2% (0-6) | -25.0 [-33.3, -17.3] pp |
| Bayesian volatility learner | 98% | 96% | 100% | 96% | 96% | 100% | 100% | 4.2 [0.0, 12.5] pp |
| All-trial count rule | 7% | 15% | 0% | 21% | 8% | 0% | 0% | -14.6 [-27.1, -4.2] pp |
| Last-five count rule | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 0.0 [0.0, 0.0] pp |

### trials8_10
| Condition | all | volatile-first (lead 6) | stable-first (lead 24) | t=11 | t=31 | t=41 | t=61 | stable-volatile [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Raw | 36% (31-43) | 41% (36-51) | 31% (20-41) | 42% (35-51) | 39% (34-50) | 29% (15-44) | 32% (24-38) | -10.2 [-16.5, -4.0] pp |
| RawCorrectness | 34% (30-38) | 44% (39-53) | 25% (19-34) | 46% (35-57) | 42% (35-49) | 24% (16-36) | 25% (21-33) | -19.7 [-25.5, -14.0] pp |
| CorrectnessOnly | 25% (13-36) | 41% (25-60) | 8% (0-18) | 44% (22-74) | 39% (26-50) | 7% (0-15) | 9% (0-27) | -33.5 [-38.7, -28.4] pp |
| Summary | 13% (7-17) | 23% (14-29) | 2% (0-6) | 26% (17-33) | 20% (11-30) | 2% (0-5) | 3% (0-8) | -20.4 [-26.6, -14.7] pp |
| Bayesian volatility learner | 94% | 95% | 92% | 96% | 94% | 88% | 97% | -2.8 [-7.6, 2.1] pp |
| All-trial count rule | 6% | 12% | 0% | 15% | 10% | 0% | 0% | -12.5 [-20.1, -5.9] pp |
| Last-five count rule | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 0.0 [0.0, 0.0] pp |

## F2 Agreement with count rules (held-out, copies combined, non-tied trials)
- CorrectnessOnly vs count_all: kappa 0.10-0.20; agreement 88.9-95.6%
- CorrectnessOnly vs last5: kappa 0.11-0.34; agreement 72.2-79.1%
- Summary vs count_all: kappa 0.21-0.98; agreement 93.5-99.9%
- Summary vs last5: kappa 0.12-0.15; agreement 73.7-76.4%
- Raw vs count_all: kappa 0.01-0.04; agreement 56.5-77.2%
- Raw vs last5: kappa -0.03-0.11; agreement 53.3-66.6%

## F3 Masking (held-out), pp [95% CI]
- Qwen3-VL-8B: P1 main +6.16 [+4.77, +7.52]; holds {'P1': True, 'P2': True, 'P3': True, 'P4': True}
- InternVL3.5-8B: P1 main +7.20 [+5.85, +8.52]; holds {'P1': True, 'P2': True, 'P3': True, 'P4': True}
- Qwen3-VL-32B: P1 main +7.65 [+6.76, +8.61]; holds {'P1': True, 'P2': True, 'P3': True, 'P4': True}
- InternVL3.5-38B: P1 main +5.08 [+4.35, +5.81]; holds {'P1': True, 'P2': True, 'P3': True, 'P4': True}
- Gemma-3-27B: P1 main +7.92 [+6.65, +9.18]; holds {'P1': True, 'P2': True, 'P3': True, 'P4': True}

## F4 Recovery ratio
- Qwen3-VL-8B: combined 0.984 [0.90, 1.07]; per copy 0.925 [0.84, 1.01]; per-copy CorrectnessOnly-Summary -0.76 [-1.71, +0.15] pp
- InternVL3.5-8B: combined 1.007 [0.96, 1.05]; per copy 0.931 [0.86, 1.00]; per-copy CorrectnessOnly-Summary -0.88 [-1.81, -0.05] pp
- Gemma-3-27B: combined 0.893 [0.83, 0.96]; per copy 1.035 [0.94, 1.15]; per-copy CorrectnessOnly-Summary +0.31 [-0.54, +1.18] pp
- Qwen3-VL-32B: combined 0.979 [0.93, 1.03]; per copy 0.728 [0.67, 0.78]; per-copy CorrectnessOnly-Summary -3.76 [-4.54, -3.00] pp
- InternVL3.5-38B: combined 0.904 [0.81, 1.01]; per copy 0.842 [0.77, 0.92]; per-copy CorrectnessOnly-Summary -1.46 [-2.20, -0.69] pp

## F5 API outputs
- Astra raw: n=960 median 5, p90 55, p95 74, p99 152, max 315; nonzero reasoning 45.6%
- Astra correct_only: n=960 median 5, p90 45, p95 60, p99 110, max 228; nonzero reasoning 29.3%
- Astra summary: n=960 median 5, p90 5, p95 5, p99 18, max 24; nonzero reasoning 1.6%
- Astra all: n=2880 median 5, p90 42, p95 57, p99 105, max 315; nonzero reasoning 25.5%
- Fable raw: visible reasoning 76.8% (n=960)
- Fable correct_only: visible reasoning 63.4% (n=960)
- Fable summary: visible reasoning 0.1% (n=960)
- Fable all: visible reasoning 46.8% (n=2880)

## F6 Psych-101 forgetting-rate slope
- Humans: median slope 0.47, positive 73.3%, Wilcoxon p=0.0076; position-adjusted b_h 0.64 [0.23, 1.05]
- hidden InternVL3_5-38B-HF: -0.477 [-0.865, +0.040] p=0.0248 Holm(5)=0.083
- hidden InternVL3_5-8B-HF: -0.549 [-0.827, -0.144] p=0.0699 Holm(5)=0.083
- hidden Qwen3-VL-32B-Instruct: -0.560 [-1.445, -0.126] p=0.0310 Holm(5)=0.083
- hidden Qwen3-VL-8B-Instruct: -0.596 [-1.219, -0.051] p=0.0208 Holm(5)=0.083
- hidden gemma-3-27b-it: -0.552 [-1.380, -0.119] p=0.0106 Holm(5)=0.053
- stated InternVL3_5-38B-HF: -0.440 [-0.835, +0.164] p=0.0310 Holm(5)=0.068
- stated InternVL3_5-8B-HF: -0.608 [-0.924, -0.160] p=0.0185 Holm(5)=0.068
- stated Qwen3-VL-32B-Instruct: -0.565 [-1.569, -0.145] p=0.0234 Holm(5)=0.068
- stated Qwen3-VL-8B-Instruct: -0.593 [-0.999, -0.159] p=0.0164 Holm(5)=0.068
- stated gemma-3-27b-it: -0.445 [-1.045, -0.063] p=0.0137 Holm(5)=0.068
