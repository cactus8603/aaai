# Raw vs. summarized history gap (analyze_main.py)

mode: formal; runs: ho_full_chart, ho_full_coded, ho_full_image, ho_full_rawcoded, ho_full_text, ho_sum_all, ho_sum_cp, ho_sum_window10, ho_sum_window20, ho_sum_window5

## Episode-level metrics (twin-averaged in logit space, t >= 2)

"all" = equal-weight mean of the pet and scenario slots. approx = share of trials whose logit was reconstructed from clipped y (legacy rows; saturates at +-13.8).

| model | data | split | arm | slot | n ep | acc [95% CI] | FB stable | FB volatile | log score | AUC10 | approx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A+|full|text | pet | 48 | 0.520 [0.506, 0.534] | 0.655 | 0.503 | -1.428 | 0.498 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A+|full|text | scenario | 48 | 0.524 [0.508, 0.540] | 0.689 | 0.497 | -1.295 | 0.491 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A+|full|text | all | 48 | 0.522 [0.509, 0.536] | 0.672 | 0.500 | -1.362 | 0.495 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A|full|image | pet | 48 | 0.501 [0.484, 0.518] | 0.500 | 0.507 | -3.565 | 0.515 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A|full|image | scenario | 48 | 0.485 [0.468, 0.502] | 0.521 | 0.496 | -1.525 | 0.504 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A|full|image | all | 48 | 0.493 [0.477, 0.509] | 0.511 | 0.502 | -2.545 | 0.510 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A|full|text | pet | 48 | 0.469 [0.456, 0.481] | 0.545 | 0.481 | -1.320 | 0.488 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A|full|text | scenario | 48 | 0.467 [0.449, 0.486] | 0.582 | 0.476 | -1.279 | 0.483 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A|full|text | all | 48 | 0.468 [0.454, 0.482] | 0.564 | 0.479 | -1.300 | 0.485 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | B'|full|chart | pet | 48 | 0.616 [0.604, 0.628] | 0.821 | 0.492 | -0.936 | 0.481 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | B'|full|chart | scenario | 48 | 0.549 [0.532, 0.567] | 0.684 | 0.495 | -1.041 | 0.489 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | B'|full|chart | all | 48 | 0.583 [0.570, 0.595] | 0.753 | 0.493 | -0.989 | 0.485 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | B|full|image | pet | 48 | 0.502 [0.484, 0.520] | 0.521 | 0.506 | -2.260 | 0.514 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | B|full|image | scenario | 48 | 0.552 [0.534, 0.569] | 0.649 | 0.512 | -1.458 | 0.510 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | B|full|image | all | 48 | 0.527 [0.511, 0.543] | 0.585 | 0.509 | -1.859 | 0.512 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | B|full|text | pet | 48 | 0.628 [0.617, 0.638] | 0.894 | 0.519 | -1.737 | 0.494 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | B|full|text | scenario | 48 | 0.628 [0.620, 0.636] | 0.847 | 0.505 | -1.225 | 0.487 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | B|full|text | all | 48 | 0.628 [0.621, 0.635] | 0.871 | 0.512 | -1.481 | 0.491 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_all|text | pet | 48 | 0.633 [0.629, 0.637] | 0.985 | 0.482 | -5.167 | 0.483 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_all|text | scenario | 48 | 0.630 [0.625, 0.635] | 0.966 | 0.481 | -3.463 | 0.482 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_all|text | all | 48 | 0.631 [0.627, 0.636] | 0.975 | 0.482 | -4.315 | 0.482 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_cp|image | pet | 48 | 0.504 [0.486, 0.521] | 0.522 | 0.513 | -3.936 | 0.520 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_cp|image | scenario | 48 | 0.599 [0.580, 0.617] | 0.746 | 0.610 | -3.058 | 0.603 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_cp|image | all | 48 | 0.551 [0.536, 0.566] | 0.634 | 0.561 | -3.497 | 0.562 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_cp|text | pet | 48 | 0.678 [0.667, 0.688] | 0.924 | 0.719 | -5.077 | 0.696 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_cp|text | scenario | 48 | 0.672 [0.660, 0.683] | 0.907 | 0.702 | -3.393 | 0.680 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_cp|text | all | 48 | 0.675 [0.664, 0.686] | 0.915 | 0.710 | -4.235 | 0.688 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_window:10|text | pet | 48 | 0.617 [0.611, 0.622] | 0.926 | 0.502 | -5.803 | 0.443 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_window:10|text | scenario | 48 | 0.614 [0.608, 0.620] | 0.920 | 0.497 | -3.903 | 0.442 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_window:10|text | all | 48 | 0.615 [0.610, 0.621] | 0.923 | 0.499 | -4.853 | 0.443 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_window:20|text | pet | 48 | 0.582 [0.571, 0.593] | 0.932 | 0.449 | -4.549 | 0.409 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_window:20|text | scenario | 48 | 0.583 [0.571, 0.593] | 0.918 | 0.450 | -2.995 | 0.414 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_window:20|text | all | 48 | 0.582 [0.571, 0.593] | 0.925 | 0.449 | -3.772 | 0.411 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_window:5|text | pet | 48 | 0.692 [0.686, 0.698] | 0.945 | 0.716 | -4.647 | 0.685 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_window:5|text | scenario | 48 | 0.689 [0.682, 0.696] | 0.927 | 0.706 | -2.988 | 0.678 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_window:5|text | all | 48 | 0.691 [0.684, 0.697] | 0.936 | 0.711 | -3.818 | 0.682 | 0.00 |

## Gap: acc(C) - acc(A), paired by episode

| model | split | slot | C arm | A arm | n | gap [95% CI] | Wilcoxon p | Holm p | role |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_all|text | A|full|text | 48 | 0.164 [0.152, 0.177] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_cp|image | A|full|image | 48 | 0.003 [-0.001, 0.006] | 0.43 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_cp|text | A|full|text | 48 | 0.209 [0.193, 0.225] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_window:10|text | A|full|text | 48 | 0.148 [0.134, 0.162] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_window:20|text | A|full|text | 48 | 0.114 [0.099, 0.128] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_window:5|text | A|full|text | 48 | 0.223 [0.208, 0.238] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_all|text | A|full|text | 48 | 0.163 [0.145, 0.180] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_cp|image | A|full|image | 48 | 0.114 [0.093, 0.136] | 2.2e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_cp|text | A|full|text | 48 | 0.205 [0.182, 0.228] | 2.4e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_window:10|text | A|full|text | 48 | 0.147 [0.126, 0.167] | 2.0e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_window:20|text | A|full|text | 48 | 0.115 [0.096, 0.134] | 3.9e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_window:5|text | A|full|text | 48 | 0.222 [0.201, 0.242] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_all|text | A|full|text | 48 | 0.164 [0.150, 0.177] | 1.6e-09 | 1.6e-09 | primary |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_cp|image | A|full|image | 48 | 0.058 [0.048, 0.069] | 1.8e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_cp|text | A|full|text | 48 | 0.207 [0.188, 0.225] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_window:10|text | A|full|text | 48 | 0.147 [0.132, 0.163] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_window:20|text | A|full|text | 48 | 0.114 [0.099, 0.130] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_window:5|text | A|full|text | 48 | 0.223 [0.206, 0.239] | 1.6e-09 | nan | exploratory |

Primary family (Wilcoxon signed-rank (two-sided), Holm, alpha=0.05): Qwen3-VL-32B-Instruct@0cfaf481: C|sum_all|text - A|full|text (all)

## B / B' vs A and C vs B (absolute accuracy differences)

| model | split | slot | B arm | A arm | n | B-A [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | A+|full|text | A|full|text | 48 | 0.051 [0.040, 0.062] | 2.3e-08 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | B'|full|chart | A|full|text | 48 | 0.147 [0.131, 0.164] | 1.6e-09 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | B|full|image | A|full|image | 48 | 0.001 [-0.002, 0.004] | 0.7 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | B|full|text | A|full|text | 48 | 0.159 [0.142, 0.176] | 1.7e-09 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | A+|full|text | A|full|text | 48 | 0.057 [0.038, 0.075] | 4.1e-06 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | B'|full|chart | A|full|text | 48 | 0.082 [0.058, 0.105] | 1.0e-06 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | B|full|image | A|full|image | 48 | 0.067 [0.051, 0.085] | 3.4e-08 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | B|full|text | A|full|text | 48 | 0.161 [0.142, 0.179] | 2.4e-09 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | A+|full|text | A|full|text | 48 | 0.054 [0.042, 0.066] | 1.2e-08 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | B'|full|chart | A|full|text | 48 | 0.115 [0.098, 0.132] | 1.7e-09 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | B|full|image | A|full|image | 48 | 0.034 [0.026, 0.043] | 8.2e-08 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | B|full|text | A|full|text | 48 | 0.160 [0.146, 0.175] | 1.6e-09 |

Normalised position (B-A)/(C-A) is shown only when C-A > 0.05 and the bootstrap C-A stays > 0; it need not lie in [0, 1].

| model | split | slot | B arm | C arm | n | B-A [95% CI] | C-B [95% CI] | C-A | position [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | A+|full|text | C|sum_all|text | 48 | 0.051 [0.040, 0.062] | 0.113 [0.100, 0.126] | 0.164 | 0.31 [0.25, 0.38] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | A+|full|text | C|sum_cp|text | 48 | 0.051 [0.040, 0.062] | 0.158 [0.140, 0.175] | 0.209 | 0.25 [0.19, 0.30] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | A+|full|text | C|sum_window:10|text | 48 | 0.051 [0.040, 0.062] | 0.097 [0.082, 0.112] | 0.148 | 0.35 [0.27, 0.42] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | A+|full|text | C|sum_window:20|text | 48 | 0.051 [0.040, 0.062] | 0.062 [0.049, 0.077] | 0.114 | 0.45 [0.36, 0.54] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | A+|full|text | C|sum_window:5|text | 48 | 0.051 [0.040, 0.062] | 0.172 [0.155, 0.188] | 0.223 | 0.23 [0.18, 0.28] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | B'|full|chart | C|sum_all|text | 48 | 0.147 [0.131, 0.164] | 0.017 [0.007, 0.027] | 0.164 | 0.90 [0.83, 0.96] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | B'|full|chart | C|sum_cp|text | 48 | 0.147 [0.131, 0.164] | 0.062 [0.047, 0.076] | 0.209 | 0.70 [0.64, 0.77] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | B'|full|chart | C|sum_window:10|text | 48 | 0.147 [0.131, 0.164] | 0.001 [-0.010, 0.013] | 0.148 | 0.99 [0.92, 1.07] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | B'|full|chart | C|sum_window:20|text | 48 | 0.147 [0.131, 0.164] | -0.034 [-0.047, -0.021] | 0.114 | 1.30 [1.17, 1.44] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | B'|full|chart | C|sum_window:5|text | 48 | 0.147 [0.131, 0.164] | 0.076 [0.062, 0.089] | 0.223 | 0.66 [0.60, 0.72] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | B|full|image | C|sum_cp|image | 48 | 0.001 [-0.002, 0.004] | 0.001 [-0.001, 0.004] | 0.003 | not interpretable: C-A = 0.003 does not exceed the prespecified threshold 0.05 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | B|full|text | C|sum_all|text | 48 | 0.159 [0.142, 0.176] | 0.005 [-0.006, 0.017] | 0.164 | 0.97 [0.90, 1.03] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | B|full|text | C|sum_cp|text | 48 | 0.159 [0.142, 0.176] | 0.050 [0.036, 0.064] | 0.209 | 0.76 [0.70, 0.82] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | B|full|text | C|sum_window:10|text | 48 | 0.159 [0.142, 0.176] | -0.011 [-0.023, 0.002] | 0.148 | 1.07 [0.98, 1.16] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | B|full|text | C|sum_window:20|text | 48 | 0.159 [0.142, 0.176] | -0.046 [-0.060, -0.031] | 0.114 | 1.40 [1.26, 1.56] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | B|full|text | C|sum_window:5|text | 48 | 0.159 [0.142, 0.176] | 0.064 [0.052, 0.076] | 0.223 | 0.71 [0.66, 0.76] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | A+|full|text | C|sum_all|text | 48 | 0.057 [0.038, 0.075] | 0.106 [0.092, 0.121] | 0.163 | 0.35 [0.25, 0.43] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | A+|full|text | C|sum_cp|text | 48 | 0.057 [0.038, 0.075] | 0.148 [0.129, 0.167] | 0.205 | 0.28 [0.20, 0.35] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | A+|full|text | C|sum_window:10|text | 48 | 0.057 [0.038, 0.075] | 0.090 [0.074, 0.106] | 0.147 | 0.39 [0.28, 0.48] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | A+|full|text | C|sum_window:20|text | 48 | 0.057 [0.038, 0.075] | 0.059 [0.041, 0.076] | 0.115 | 0.49 [0.36, 0.62] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | A+|full|text | C|sum_window:5|text | 48 | 0.057 [0.038, 0.075] | 0.165 [0.147, 0.183] | 0.222 | 0.26 [0.18, 0.32] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | B'|full|chart | C|sum_all|text | 48 | 0.082 [0.058, 0.105] | 0.081 [0.063, 0.098] | 0.163 | 0.50 [0.38, 0.61] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | B'|full|chart | C|sum_cp|text | 48 | 0.082 [0.058, 0.105] | 0.123 [0.101, 0.145] | 0.205 | 0.40 [0.30, 0.50] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | B'|full|chart | C|sum_window:10|text | 48 | 0.082 [0.058, 0.105] | 0.065 [0.048, 0.081] | 0.147 | 0.56 [0.43, 0.67] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | B'|full|chart | C|sum_window:20|text | 48 | 0.082 [0.058, 0.105] | 0.033 [0.014, 0.053] | 0.115 | 0.71 [0.54, 0.87] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | B'|full|chart | C|sum_window:5|text | 48 | 0.082 [0.058, 0.105] | 0.140 [0.119, 0.160] | 0.222 | 0.37 [0.27, 0.46] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | B|full|image | C|sum_cp|image | 48 | 0.067 [0.051, 0.085] | 0.047 [0.032, 0.060] | 0.114 | 0.59 [0.49, 0.70] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | B|full|text | C|sum_all|text | 48 | 0.161 [0.142, 0.179] | 0.002 [-0.007, 0.011] | 0.163 | 0.99 [0.94, 1.04] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | B|full|text | C|sum_cp|text | 48 | 0.161 [0.142, 0.179] | 0.044 [0.032, 0.056] | 0.205 | 0.79 [0.74, 0.84] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | B|full|text | C|sum_window:10|text | 48 | 0.161 [0.142, 0.179] | -0.014 [-0.025, -0.003] | 0.147 | 1.10 [1.02, 1.18] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | B|full|text | C|sum_window:20|text | 48 | 0.161 [0.142, 0.179] | -0.046 [-0.061, -0.031] | 0.115 | 1.40 [1.25, 1.59] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | B|full|text | C|sum_window:5|text | 48 | 0.161 [0.142, 0.179] | 0.061 [0.052, 0.070] | 0.222 | 0.72 [0.69, 0.76] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | A+|full|text | C|sum_all|text | 48 | 0.054 [0.042, 0.066] | 0.109 [0.097, 0.121] | 0.164 | 0.33 [0.27, 0.39] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | A+|full|text | C|sum_cp|text | 48 | 0.054 [0.042, 0.066] | 0.153 [0.135, 0.170] | 0.207 | 0.26 [0.21, 0.31] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | A+|full|text | C|sum_window:10|text | 48 | 0.054 [0.042, 0.066] | 0.093 [0.079, 0.107] | 0.147 | 0.37 [0.30, 0.44] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | A+|full|text | C|sum_window:20|text | 48 | 0.054 [0.042, 0.066] | 0.060 [0.046, 0.075] | 0.114 | 0.47 [0.38, 0.57] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | A+|full|text | C|sum_window:5|text | 48 | 0.054 [0.042, 0.066] | 0.169 [0.152, 0.184] | 0.223 | 0.24 [0.19, 0.29] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | B'|full|chart | C|sum_all|text | 48 | 0.115 [0.098, 0.132] | 0.049 [0.037, 0.061] | 0.164 | 0.70 [0.62, 0.77] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | B'|full|chart | C|sum_cp|text | 48 | 0.115 [0.098, 0.132] | 0.092 [0.076, 0.109] | 0.207 | 0.55 [0.48, 0.62] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | B'|full|chart | C|sum_window:10|text | 48 | 0.115 [0.098, 0.132] | 0.033 [0.021, 0.045] | 0.147 | 0.78 [0.70, 0.86] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | B'|full|chart | C|sum_window:20|text | 48 | 0.115 [0.098, 0.132] | -0.000 [-0.014, 0.014] | 0.114 | 1.00 [0.88, 1.13] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | B'|full|chart | C|sum_window:5|text | 48 | 0.115 [0.098, 0.132] | 0.108 [0.092, 0.123] | 0.223 | 0.52 [0.45, 0.58] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | B|full|image | C|sum_cp|image | 48 | 0.034 [0.026, 0.043] | 0.024 [0.017, 0.030] | 0.058 | 0.59 [0.49, 0.69] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | B|full|text | C|sum_all|text | 48 | 0.160 [0.146, 0.175] | 0.003 [-0.004, 0.011] | 0.164 | 0.98 [0.93, 1.03] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | B|full|text | C|sum_cp|text | 48 | 0.160 [0.146, 0.175] | 0.047 [0.035, 0.058] | 0.207 | 0.77 [0.73, 0.82] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | B|full|text | C|sum_window:10|text | 48 | 0.160 [0.146, 0.175] | -0.013 [-0.023, -0.002] | 0.147 | 1.09 [1.01, 1.16] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | B|full|text | C|sum_window:20|text | 48 | 0.160 [0.146, 0.175] | -0.046 [-0.059, -0.033] | 0.114 | 1.40 [1.27, 1.56] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | B|full|text | C|sum_window:5|text | 48 | 0.160 [0.146, 0.175] | 0.062 [0.054, 0.071] | 0.223 | 0.72 [0.69, 0.75] |

## TOST: VLM + summary vs rule alone (margin +-0.02, alpha 0.05)

Equivalence here means only that the accuracy difference lies within the margin; it does not imply trial-by-trial rule following or identical internal computation (see agreement below).

- Secondary family Qwen3-VL-32B-Instruct@0cfaf481/episodes_v3#e00b8793: members ['C|sum_all|text vs window-count rule (domains pooled)', 'C|sum_cp|text vs window-count rule (domains pooled)', 'C|sum_window:5|text vs window-count rule (domains pooled)']; K* = 5 (source analysis/h200_ho/dev_selection.json); margin 0.02; alpha 0.05; Holm (missing members counted as p = 1)

| model | split | slot | C arm | rule | n | acc VLM | acc rule | diff [90% CI] | p TOST | equiv. (uncorr.) | Holm p | equiv. (Holm) | role |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_all|text | window_counts | 48 | 0.633 | 0.632 | 0.001 [-0.001, 0.003] | 1.8e-21 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_all|text | bocpd_posterior | 48 | 0.633 | 0.677 | -0.044 [-0.051, -0.036] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_cp|image | window_counts | 48 | 0.504 | 0.678 | -0.174 [-0.193, -0.155] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_cp|image | bocpd_posterior | 48 | 0.504 | 0.677 | -0.173 [-0.190, -0.156] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_cp|text | window_counts | 48 | 0.678 | 0.678 | -0.000 [-0.001, 0.001] | 8.9e-37 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_cp|text | bocpd_posterior | 48 | 0.678 | 0.677 | 0.001 [-0.005, 0.007] | 8.3e-06 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_window:10|text | window_counts | 48 | 0.617 | 0.612 | 0.005 [0.001, 0.009] | 8.9e-09 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_window:10|text | bocpd_posterior | 48 | 0.617 | 0.677 | -0.060 [-0.067, -0.053] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_window:20|text | window_counts | 48 | 0.582 | 0.579 | 0.003 [-0.002, 0.008] | 1.2e-06 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_window:20|text | bocpd_posterior | 48 | 0.582 | 0.677 | -0.094 [-0.104, -0.084] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_window:5|text | window_counts | 48 | 0.692 | 0.692 | 0.000 [-0.001, 0.001] | 7.6e-40 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_window:5|text | bocpd_posterior | 48 | 0.692 | 0.677 | 0.015 [0.009, 0.021] | 0.1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_all|text | window_counts | 48 | 0.630 | 0.632 | -0.002 [-0.005, 0.001] | 5.2e-12 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_all|text | bocpd_posterior | 48 | 0.630 | 0.677 | -0.047 [-0.054, -0.039] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_cp|image | window_counts | 48 | 0.599 | 0.678 | -0.079 [-0.095, -0.063] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_cp|image | bocpd_posterior | 48 | 0.599 | 0.677 | -0.078 [-0.094, -0.061] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_cp|text | window_counts | 48 | 0.672 | 0.678 | -0.006 [-0.009, -0.003] | 1.7e-09 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_cp|text | bocpd_posterior | 48 | 0.672 | 0.677 | -0.004 [-0.012, 0.003] | 0.00098 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_window:10|text | window_counts | 48 | 0.614 | 0.612 | 0.002 [-0.003, 0.006] | 1.5e-08 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_window:10|text | bocpd_posterior | 48 | 0.614 | 0.677 | -0.063 [-0.070, -0.055] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_window:20|text | window_counts | 48 | 0.583 | 0.579 | 0.003 [-0.002, 0.009] | 6.0e-06 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_window:20|text | bocpd_posterior | 48 | 0.583 | 0.677 | -0.094 [-0.104, -0.084] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_window:5|text | window_counts | 48 | 0.689 | 0.692 | -0.002 [-0.005, 0.000] | 1.2e-14 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_window:5|text | bocpd_posterior | 48 | 0.689 | 0.677 | 0.013 [0.006, 0.019] | 0.042 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_all|text | window_counts | 48 | 0.631 | 0.632 | -0.000 [-0.003, 0.002] | 1.1e-18 | yes | 1.1e-18 | yes | secondary |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_all|text | bocpd_posterior | 48 | 0.631 | 0.677 | -0.045 [-0.053, -0.038] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_cp|image | window_counts | 48 | 0.551 | 0.678 | -0.127 [-0.142, -0.112] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_cp|image | bocpd_posterior | 48 | 0.551 | 0.677 | -0.125 [-0.139, -0.112] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_cp|text | window_counts | 48 | 0.675 | 0.678 | -0.003 [-0.005, -0.001] | 1.4e-20 | yes | 2.7e-20 | yes | secondary |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_cp|text | bocpd_posterior | 48 | 0.675 | 0.677 | -0.002 [-0.009, 0.005] | 4.3e-05 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_window:10|text | window_counts | 48 | 0.615 | 0.612 | 0.003 [-0.001, 0.007] | 7.5e-09 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_window:10|text | bocpd_posterior | 48 | 0.615 | 0.677 | -0.061 [-0.069, -0.054] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_window:20|text | window_counts | 48 | 0.582 | 0.579 | 0.003 [-0.002, 0.008] | 1.8e-06 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_window:20|text | bocpd_posterior | 48 | 0.582 | 0.677 | -0.094 [-0.104, -0.084] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_window:5|text | window_counts | 48 | 0.691 | 0.692 | -0.001 [-0.003, 0.000] | 8.4e-26 | yes | 2.5e-25 | yes | secondary |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_window:5|text | bocpd_posterior | 48 | 0.691 | 0.677 | 0.014 [0.008, 0.020] | 0.062 | no | nan | - | exploratory |

## Trial-level choice agreement: VLM + summary vs window-count rule (non-tied trials)

| model | split | slot | C arm | n ep | non-tied | agree (pooled) | agree ep-mean [95% CI] | disagree | VLM right | rule right | stable | volatile | VLM->A | VLM->B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_all|text | 48 | 3687 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_cp|image | 48 | 3770 | 0.503 | 0.503 [0.486, 0.519] | 1874 | 607 | 1267 | 948 | 926 | 489 | 1385 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_cp|text | 48 | 3770 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_window:10|text | 48 | 3487 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_window:20|text | 48 | 2945 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_window:5|text | 48 | 3776 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_all|text | 48 | 3687 | 0.989 | 0.989 [0.979, 0.996] | 41 | 15 | 26 | 19 | 22 | 0 | 41 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_cp|image | 48 | 3770 | 0.770 | 0.770 [0.733, 0.807] | 868 | 285 | 583 | 423 | 445 | 244 | 624 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_cp|text | 48 | 3770 | 0.986 | 0.986 [0.978, 0.993] | 53 | 16 | 37 | 22 | 31 | 18 | 35 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_window:10|text | 48 | 3487 | 0.996 | 0.996 [0.993, 0.998] | 15 | 2 | 13 | 6 | 9 | 1 | 14 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_window:20|text | 48 | 2945 | 0.981 | 0.981 [0.970, 0.991] | 55 | 28 | 27 | 21 | 34 | 14 | 41 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_window:5|text | 48 | 3776 | 0.994 | 0.994 [0.989, 0.998] | 23 | 7 | 16 | 10 | 13 | 4 | 19 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_all|text | 48 | 7374 | 0.994 | 0.994 [0.989, 0.998] | 41 | 15 | 26 | 19 | 22 | 0 | 41 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_cp|image | 48 | 7540 | 0.636 | 0.636 [0.613, 0.659] | 2742 | 892 | 1850 | 1371 | 1371 | 733 | 2009 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_cp|text | 48 | 7540 | 0.993 | 0.993 [0.989, 0.996] | 53 | 16 | 37 | 22 | 31 | 18 | 35 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_window:10|text | 48 | 6974 | 0.998 | 0.998 [0.996, 0.999] | 15 | 2 | 13 | 6 | 9 | 1 | 14 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_window:20|text | 48 | 5890 | 0.991 | 0.991 [0.985, 0.996] | 55 | 28 | 27 | 21 | 34 | 14 | 41 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_window:5|text | 48 | 7552 | 0.997 | 0.997 [0.995, 0.999] | 23 | 7 | 16 | 10 | 13 | 4 | 19 |

## Per-scenario gap (domain_slot = scenario only; pet reported separately above)

Each held-out text scenario has 8 episode pairs; the cross-scenario sign test has 6 units (one per text scenario; minimum two-sided p = 0.03125). Pet is reported separately and is not part of the cross-scenario test.

| model | split | C arm | scenario | n pairs | gap [95% CI] |
|---|---|---|---|---|---|
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_all|text | color | 8 | 0.169 [0.134, 0.199] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_all|text | medical | 8 | 0.168 [0.106, 0.229] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_all|text | news | 8 | 0.101 [0.070, 0.133] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_all|text | sports | 8 | 0.158 [0.125, 0.191] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_all|text | stock | 8 | 0.193 [0.163, 0.225] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_all|text | weather | 8 | 0.187 [0.158, 0.220] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_all|text | sign test | 6 scen. | 6 positive, p=0.031 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|image | color | 8 | 0.209 [0.149, 0.261] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|image | medical | 8 | 0.149 [0.120, 0.184] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|image | news | 8 | 0.100 [0.065, 0.142] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|image | sports | 8 | 0.101 [0.071, 0.133] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|image | stock | 8 | 0.076 [0.052, 0.100] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|image | weather | 8 | 0.049 [0.019, 0.079] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|image | sign test | 6 scen. | 6 positive, p=0.031 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|text | color | 8 | 0.217 [0.157, 0.264] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|text | medical | 8 | 0.217 [0.158, 0.275] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|text | news | 8 | 0.120 [0.066, 0.176] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|text | sports | 8 | 0.199 [0.158, 0.236] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|text | stock | 8 | 0.244 [0.206, 0.283] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|text | weather | 8 | 0.233 [0.191, 0.272] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|text | sign test | 6 scen. | 6 positive, p=0.031 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:10|text | color | 8 | 0.147 [0.103, 0.187] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:10|text | medical | 8 | 0.152 [0.081, 0.222] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:10|text | news | 8 | 0.085 [0.051, 0.116] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:10|text | sports | 8 | 0.128 [0.093, 0.169] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:10|text | stock | 8 | 0.191 [0.153, 0.228] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:10|text | weather | 8 | 0.176 [0.149, 0.204] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:10|text | sign test | 6 scen. | 6 positive, p=0.031 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:20|text | color | 8 | 0.116 [0.085, 0.146] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:20|text | medical | 8 | 0.106 [0.035, 0.174] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:20|text | news | 8 | 0.066 [0.032, 0.103] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:20|text | sports | 8 | 0.116 [0.073, 0.160] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:20|text | stock | 8 | 0.160 [0.136, 0.180] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:20|text | weather | 8 | 0.128 [0.090, 0.169] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:20|text | sign test | 6 scen. | 6 positive, p=0.031 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:5|text | color | 8 | 0.220 [0.166, 0.264] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:5|text | medical | 8 | 0.225 [0.165, 0.288] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:5|text | news | 8 | 0.155 [0.108, 0.191] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:5|text | sports | 8 | 0.220 [0.180, 0.267] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:5|text | stock | 8 | 0.263 [0.228, 0.296] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:5|text | weather | 8 | 0.250 [0.218, 0.283] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:5|text | sign test | 6 scen. | 6 positive, p=0.031 |

## History baselines (same trial mask as the reference A arm)

Baselines marked "yes" use generator information the VLM does not have and are not fair competitors.

| model | split | baseline | acc [95% CI] | n ep | uses generator info | description |
|---|---|---|---|---|---|---|
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | truth_quota | 0.655 [0.650, 0.661] | 48 | yes | knows the generator's 5/5 truth quota per 10-trial block; predicts the label with more remaining quota and follows the reporter who reports it (no reliability tracking) |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | always_A | 0.648 [0.647, 0.650] | 48 | yes | diagnostic only: always follows latent role A (role identity is hidden from the VLM) |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | known_pA | 0.800 [0.799, 0.801] | 48 | yes | knows the true current p_A and follows the more reliable reporter (reference ceiling) |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | bocpd_posterior | 0.677 [0.669, 0.684] | 48 | no | BOCPD posterior predictive p_A from the outcome history |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | bocpd_map_window | 0.678 [0.667, 0.688] | 48 | no | majority within the BOCPD MAP run-length window (= the sum_cp summary rule) |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | count_all | 0.632 [0.628, 0.636] | 48 | no | majority of all past outcomes (= the sum_all summary rule) |

## Readout quality (candidate mass, greedy output follows candidate format)

| model | split | arm | slot | format | n | mass mean | mass min | frac mass<0.5 | greedy follows |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | A+|full|text | pet | text | 7680 | 0.997 | 0.959 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | A+|full|text | scenario | text | 7680 | 0.998 | 0.980 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | A|full|image | pet | image | 7680 | 0.999 | 0.994 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | A|full|image | scenario | image | 7680 | 0.994 | 0.927 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | A|full|text | pet | text | 7680 | 0.994 | 0.881 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | A|full|text | scenario | text | 7680 | 0.996 | 0.939 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | B'|full|chart | pet | chart | 7680 | 0.964 | 0.807 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | B'|full|chart | scenario | chart | 7680 | 0.980 | 0.854 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | B|full|image | pet | image | 7680 | 0.999 | 0.990 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | B|full|image | scenario | image | 7680 | 0.995 | 0.935 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | B|full|text | pet | text | 7680 | 0.991 | 0.926 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | B|full|text | scenario | text | 7680 | 0.994 | 0.913 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_all|text | pet | text | 7680 | 1.000 | 0.997 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_all|text | scenario | text | 7680 | 1.000 | 0.991 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|image | pet | image | 7680 | 0.999 | 0.985 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|image | scenario | image | 7680 | 0.999 | 0.985 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|text | pet | text | 7680 | 1.000 | 0.997 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_cp|text | scenario | text | 7680 | 1.000 | 0.987 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:10|text | pet | text | 7680 | 1.000 | 0.997 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:10|text | scenario | text | 7680 | 1.000 | 0.993 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:20|text | pet | text | 7680 | 1.000 | 0.997 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:20|text | scenario | text | 7680 | 1.000 | 0.988 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:5|text | pet | text | 7680 | 1.000 | 0.997 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_window:5|text | scenario | text | 7680 | 1.000 | 0.993 | 0.000 | 1.000 |

Rule-window mismatches (recomputed vs logged window_start): 0; identical duplicate rows merged: 0; legacy y-only rows (approx logits): 0.
