# Raw vs. summarized history gap (analyze_main.py)

mode: formal; runs: ho_full_chart, ho_full_coded, ho_full_image, ho_full_rawcoded, ho_full_text, ho_sum_all, ho_sum_cp, ho_sum_window10, ho_sum_window20, ho_sum_window5

## Episode-level metrics (twin-averaged in logit space, t >= 2)

"all" = equal-weight mean of the pet and scenario slots. approx = share of trials whose logit was reconstructed from clipped y (legacy rows; saturates at +-13.8).

| model | data | split | arm | slot | n ep | acc [95% CI] | FB stable | FB volatile | log score | AUC10 | approx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | A+|full|text | pet | 48 | 0.556 [0.545, 0.568] | 0.718 | 0.515 | -1.251 | 0.512 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | A+|full|text | scenario | 48 | 0.564 [0.551, 0.576] | 0.719 | 0.512 | -0.990 | 0.507 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | A+|full|text | all | 48 | 0.560 [0.549, 0.570] | 0.719 | 0.513 | -1.120 | 0.510 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | A|full|image | pet | 48 | 0.502 [0.484, 0.520] | 0.500 | 0.504 | -1.916 | 0.513 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | A|full|image | scenario | 48 | 0.495 [0.479, 0.511] | 0.540 | 0.490 | -0.825 | 0.492 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | A|full|image | all | 48 | 0.499 [0.483, 0.514] | 0.520 | 0.497 | -1.371 | 0.503 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | A|full|text | pet | 48 | 0.553 [0.540, 0.565] | 0.683 | 0.477 | -0.928 | 0.481 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | A|full|text | scenario | 48 | 0.545 [0.531, 0.560] | 0.684 | 0.479 | -0.908 | 0.486 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | A|full|text | all | 48 | 0.549 [0.538, 0.560] | 0.683 | 0.478 | -0.918 | 0.483 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | B'|full|chart | pet | 48 | 0.629 [0.619, 0.638] | 0.728 | 0.494 | -0.719 | 0.478 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | B'|full|chart | scenario | 48 | 0.580 [0.562, 0.597] | 0.644 | 0.496 | -0.724 | 0.487 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | B'|full|chart | all | 48 | 0.605 [0.593, 0.615] | 0.686 | 0.495 | -0.721 | 0.483 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | B|full|image | pet | 48 | 0.502 [0.484, 0.520] | 0.514 | 0.506 | -2.122 | 0.514 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | B|full|image | scenario | 48 | 0.544 [0.525, 0.562] | 0.621 | 0.501 | -1.099 | 0.496 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | B|full|image | all | 48 | 0.523 [0.507, 0.539] | 0.567 | 0.503 | -1.611 | 0.505 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | B|full|text | pet | 48 | 0.626 [0.618, 0.633] | 0.909 | 0.524 | -1.668 | 0.504 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | B|full|text | scenario | 48 | 0.624 [0.614, 0.633] | 0.834 | 0.507 | -1.044 | 0.485 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | B|full|text | all | 48 | 0.625 [0.617, 0.632] | 0.871 | 0.515 | -1.356 | 0.495 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_all|text | pet | 48 | 0.633 [0.628, 0.637] | 0.984 | 0.483 | -3.618 | 0.483 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_all|text | scenario | 48 | 0.633 [0.629, 0.637] | 0.973 | 0.487 | -2.508 | 0.483 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_all|text | all | 48 | 0.633 [0.629, 0.637] | 0.978 | 0.485 | -3.063 | 0.483 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_cp|image | pet | 48 | 0.515 [0.497, 0.532] | 0.574 | 0.518 | -2.326 | 0.525 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_cp|image | scenario | 48 | 0.573 [0.550, 0.596] | 0.662 | 0.601 | -1.525 | 0.603 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_cp|image | all | 48 | 0.544 [0.527, 0.560] | 0.618 | 0.560 | -1.926 | 0.564 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_cp|text | pet | 48 | 0.678 [0.666, 0.688] | 0.924 | 0.719 | -3.748 | 0.696 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_cp|text | scenario | 48 | 0.677 [0.666, 0.688] | 0.920 | 0.718 | -2.632 | 0.695 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_cp|text | all | 48 | 0.677 [0.666, 0.688] | 0.922 | 0.718 | -3.190 | 0.696 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_window:10|text | pet | 48 | 0.617 [0.611, 0.622] | 0.926 | 0.497 | -4.176 | 0.438 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_window:10|text | scenario | 48 | 0.615 [0.609, 0.621] | 0.924 | 0.500 | -2.994 | 0.442 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_window:10|text | all | 48 | 0.616 [0.610, 0.622] | 0.925 | 0.499 | -3.585 | 0.440 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_window:20|text | pet | 48 | 0.582 [0.571, 0.593] | 0.931 | 0.446 | -3.143 | 0.406 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_window:20|text | scenario | 48 | 0.581 [0.569, 0.592] | 0.927 | 0.451 | -2.255 | 0.410 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_window:20|text | all | 48 | 0.581 [0.570, 0.593] | 0.929 | 0.449 | -2.699 | 0.408 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_window:5|text | pet | 48 | 0.692 [0.686, 0.698] | 0.945 | 0.716 | -3.252 | 0.685 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_window:5|text | scenario | 48 | 0.691 [0.685, 0.697] | 0.942 | 0.715 | -2.410 | 0.684 | 0.00 |
| InternVL3_5-38B-HF@7c830fc2 | episodes_v3#e00b8793 | held_out | C|sum_window:5|text | all | 48 | 0.691 [0.685, 0.697] | 0.943 | 0.715 | -2.831 | 0.685 | 0.00 |

## Gap: acc(C) - acc(A), paired by episode

| model | split | slot | C arm | A arm | n | gap [95% CI] | Wilcoxon p | Holm p | role |
|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_all|text | A|full|text | 48 | 0.080 [0.068, 0.093] | 2.1e-09 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_cp|image | A|full|image | 48 | 0.013 [0.007, 0.018] | 0.00013 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_cp|text | A|full|text | 48 | 0.125 [0.109, 0.141] | 1.6e-09 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_window:10|text | A|full|text | 48 | 0.064 [0.050, 0.078] | 2.9e-08 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_window:20|text | A|full|text | 48 | 0.030 [0.015, 0.045] | 0.0021 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_window:5|text | A|full|text | 48 | 0.139 [0.124, 0.154] | 1.6e-09 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_all|text | A|full|text | 48 | 0.088 [0.075, 0.100] | 2.5e-09 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_cp|image | A|full|image | 48 | 0.078 [0.054, 0.104] | 1.6e-06 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_cp|text | A|full|text | 48 | 0.132 [0.117, 0.147] | 2.4e-09 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_window:10|text | A|full|text | 48 | 0.070 [0.055, 0.085] | 6.2e-08 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_window:20|text | A|full|text | 48 | 0.035 [0.018, 0.053] | 0.00093 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_window:5|text | A|full|text | 48 | 0.146 [0.130, 0.161] | 1.6e-09 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_all|text | A|full|text | 48 | 0.084 [0.074, 0.094] | 1.6e-09 | 1.6e-09 | primary |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_cp|image | A|full|image | 48 | 0.045 [0.033, 0.059] | 2.7e-07 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_cp|text | A|full|text | 48 | 0.128 [0.115, 0.142] | 1.6e-09 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_window:10|text | A|full|text | 48 | 0.067 [0.055, 0.079] | 4.7e-09 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_window:20|text | A|full|text | 48 | 0.032 [0.019, 0.047] | 0.00015 | nan | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_window:5|text | A|full|text | 48 | 0.142 [0.129, 0.155] | 1.6e-09 | nan | exploratory |

Primary family (Wilcoxon signed-rank (two-sided), Holm, alpha=0.05): InternVL3_5-38B-HF@7c830fc2: C|sum_all|text - A|full|text (all)

## B / B' vs A and C vs B (absolute accuracy differences)

| model | split | slot | B arm | A arm | n | B-A [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | A+|full|text | A|full|text | 48 | 0.003 [-0.007, 0.014] | 0.73 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | B'|full|chart | A|full|text | 48 | 0.076 [0.062, 0.091] | 1.2e-08 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | B|full|image | A|full|image | 48 | 0.000 [0.000, 0.000] | 1 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | B|full|text | A|full|text | 48 | 0.073 [0.058, 0.087] | 2.0e-08 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | A+|full|text | A|full|text | 48 | 0.018 [0.004, 0.032] | 0.022 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | B'|full|chart | A|full|text | 48 | 0.035 [0.015, 0.055] | 0.0015 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | B|full|image | A|full|image | 48 | 0.049 [0.030, 0.070] | 0.00012 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | B|full|text | A|full|text | 48 | 0.079 [0.064, 0.094] | 3.4e-09 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | A+|full|text | A|full|text | 48 | 0.011 [0.002, 0.020] | 0.024 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | B'|full|chart | A|full|text | 48 | 0.056 [0.043, 0.068] | 6.2e-08 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | B|full|image | A|full|image | 48 | 0.025 [0.015, 0.035] | 0.00012 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | B|full|text | A|full|text | 48 | 0.076 [0.065, 0.087] | 1.6e-09 |

Normalised position (B-A)/(C-A) is shown only when C-A > 0.05 and the bootstrap C-A stays > 0; it need not lie in [0, 1].

| model | split | slot | B arm | C arm | n | B-A [95% CI] | C-B [95% CI] | C-A | position [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | A+|full|text | C|sum_all|text | 48 | 0.003 [-0.007, 0.014] | 0.077 [0.066, 0.088] | 0.080 | 0.04 [-0.09, 0.16] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | A+|full|text | C|sum_cp|text | 48 | 0.003 [-0.007, 0.014] | 0.122 [0.107, 0.136] | 0.125 | 0.03 [-0.06, 0.10] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | A+|full|text | C|sum_window:10|text | 48 | 0.003 [-0.007, 0.014] | 0.061 [0.049, 0.073] | 0.064 | 0.05 [-0.12, 0.19] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | A+|full|text | C|sum_window:20|text | 48 | 0.003 [-0.007, 0.014] | 0.026 [0.013, 0.040] | 0.030 | not interpretable: C-A = 0.030 does not exceed the prespecified threshold 0.05 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | A+|full|text | C|sum_window:5|text | 48 | 0.003 [-0.007, 0.014] | 0.136 [0.123, 0.148] | 0.139 | 0.02 [-0.05, 0.09] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | B'|full|chart | C|sum_all|text | 48 | 0.076 [0.062, 0.091] | 0.004 [-0.006, 0.013] | 0.080 | 0.95 [0.83, 1.08] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | B'|full|chart | C|sum_cp|text | 48 | 0.076 [0.062, 0.091] | 0.049 [0.034, 0.063] | 0.125 | 0.61 [0.51, 0.71] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | B'|full|chart | C|sum_window:10|text | 48 | 0.076 [0.062, 0.091] | -0.012 [-0.023, -0.001] | 0.064 | 1.19 [1.02, 1.40] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | B'|full|chart | C|sum_window:20|text | 48 | 0.076 [0.062, 0.091] | -0.047 [-0.061, -0.032] | 0.030 | not interpretable: C-A = 0.030 does not exceed the prespecified threshold 0.05 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | B'|full|chart | C|sum_window:5|text | 48 | 0.076 [0.062, 0.091] | 0.063 [0.051, 0.075] | 0.139 | 0.55 [0.47, 0.62] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | B|full|image | C|sum_cp|image | 48 | 0.000 [0.000, 0.000] | 0.013 [0.007, 0.018] | 0.013 | not interpretable: C-A = 0.013 does not exceed the prespecified threshold 0.05 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | B|full|text | C|sum_all|text | 48 | 0.073 [0.058, 0.087] | 0.007 [-0.002, 0.016] | 0.080 | 0.91 [0.80, 1.02] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | B|full|text | C|sum_cp|text | 48 | 0.073 [0.058, 0.087] | 0.052 [0.041, 0.063] | 0.125 | 0.58 [0.50, 0.66] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | B|full|text | C|sum_window:10|text | 48 | 0.073 [0.058, 0.087] | -0.009 [-0.017, -0.001] | 0.064 | 1.14 [1.01, 1.28] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | B|full|text | C|sum_window:20|text | 48 | 0.073 [0.058, 0.087] | -0.043 [-0.056, -0.030] | 0.030 | not interpretable: C-A = 0.030 does not exceed the prespecified threshold 0.05 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | B|full|text | C|sum_window:5|text | 48 | 0.073 [0.058, 0.087] | 0.066 [0.058, 0.075] | 0.139 | 0.52 [0.45, 0.59] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | A+|full|text | C|sum_all|text | 48 | 0.018 [0.004, 0.032] | 0.069 [0.058, 0.080] | 0.088 | 0.21 [0.06, 0.34] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | A+|full|text | C|sum_cp|text | 48 | 0.018 [0.004, 0.032] | 0.114 [0.099, 0.128] | 0.132 | 0.14 [0.04, 0.23] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | A+|full|text | C|sum_window:10|text | 48 | 0.018 [0.004, 0.032] | 0.052 [0.039, 0.065] | 0.070 | 0.26 [0.07, 0.43] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | A+|full|text | C|sum_window:20|text | 48 | 0.018 [0.004, 0.032] | 0.017 [0.006, 0.030] | 0.035 | not interpretable: C-A = 0.035 does not exceed the prespecified threshold 0.05 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | A+|full|text | C|sum_window:5|text | 48 | 0.018 [0.004, 0.032] | 0.128 [0.112, 0.142] | 0.146 | 0.12 [0.03, 0.21] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | B'|full|chart | C|sum_all|text | 48 | 0.035 [0.015, 0.055] | 0.053 [0.036, 0.070] | 0.088 | 0.40 [0.18, 0.59] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | B'|full|chart | C|sum_cp|text | 48 | 0.035 [0.015, 0.055] | 0.097 [0.077, 0.117] | 0.132 | 0.26 [0.11, 0.40] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | B'|full|chart | C|sum_window:10|text | 48 | 0.035 [0.015, 0.055] | 0.035 [0.019, 0.051] | 0.070 | 0.50 [0.23, 0.72] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | B'|full|chart | C|sum_window:20|text | 48 | 0.035 [0.015, 0.055] | 0.001 [-0.017, 0.019] | 0.035 | not interpretable: C-A = 0.035 does not exceed the prespecified threshold 0.05 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | B'|full|chart | C|sum_window:5|text | 48 | 0.035 [0.015, 0.055] | 0.111 [0.092, 0.131] | 0.146 | 0.24 [0.10, 0.36] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | B|full|image | C|sum_cp|image | 48 | 0.049 [0.030, 0.070] | 0.029 [0.014, 0.046] | 0.078 | 0.63 [0.46, 0.80] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | B|full|text | C|sum_all|text | 48 | 0.079 [0.064, 0.094] | 0.009 [-0.002, 0.019] | 0.088 | 0.90 [0.78, 1.02] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | B|full|text | C|sum_cp|text | 48 | 0.079 [0.064, 0.094] | 0.053 [0.040, 0.066] | 0.132 | 0.60 [0.51, 0.69] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | B|full|text | C|sum_window:10|text | 48 | 0.079 [0.064, 0.094] | -0.009 [-0.019, 0.001] | 0.070 | 1.13 [0.98, 1.30] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | B|full|text | C|sum_window:20|text | 48 | 0.079 [0.064, 0.094] | -0.044 [-0.059, -0.028] | 0.035 | not interpretable: C-A = 0.035 does not exceed the prespecified threshold 0.05 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | B|full|text | C|sum_window:5|text | 48 | 0.079 [0.064, 0.094] | 0.067 [0.057, 0.077] | 0.146 | 0.54 [0.47, 0.61] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | A+|full|text | C|sum_all|text | 48 | 0.011 [0.002, 0.020] | 0.073 [0.064, 0.082] | 0.084 | 0.13 [0.03, 0.22] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | A+|full|text | C|sum_cp|text | 48 | 0.011 [0.002, 0.020] | 0.118 [0.105, 0.131] | 0.128 | 0.08 [0.02, 0.15] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | A+|full|text | C|sum_window:10|text | 48 | 0.011 [0.002, 0.020] | 0.056 [0.045, 0.067] | 0.067 | 0.16 [0.03, 0.28] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | A+|full|text | C|sum_window:20|text | 48 | 0.011 [0.002, 0.020] | 0.022 [0.011, 0.033] | 0.032 | not interpretable: C-A = 0.032 does not exceed the prespecified threshold 0.05 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | A+|full|text | C|sum_window:5|text | 48 | 0.011 [0.002, 0.020] | 0.132 [0.119, 0.144] | 0.142 | 0.08 [0.02, 0.13] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | B'|full|chart | C|sum_all|text | 48 | 0.056 [0.043, 0.068] | 0.028 [0.018, 0.039] | 0.084 | 0.66 [0.54, 0.78] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | B'|full|chart | C|sum_cp|text | 48 | 0.056 [0.043, 0.068] | 0.073 [0.058, 0.088] | 0.128 | 0.43 [0.34, 0.53] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | B'|full|chart | C|sum_window:10|text | 48 | 0.056 [0.043, 0.068] | 0.011 [0.001, 0.022] | 0.067 | 0.83 [0.68, 0.98] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | B'|full|chart | C|sum_window:20|text | 48 | 0.056 [0.043, 0.068] | -0.023 [-0.036, -0.010] | 0.032 | not interpretable: C-A = 0.032 does not exceed the prespecified threshold 0.05 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | B'|full|chart | C|sum_window:5|text | 48 | 0.056 [0.043, 0.068] | 0.087 [0.073, 0.100] | 0.142 | 0.39 [0.31, 0.47] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | B|full|image | C|sum_cp|image | 48 | 0.025 [0.015, 0.035] | 0.021 [0.012, 0.030] | 0.045 | not interpretable: C-A = 0.045 does not exceed the prespecified threshold 0.05 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | B|full|text | C|sum_all|text | 48 | 0.076 [0.065, 0.087] | 0.008 [-0.001, 0.016] | 0.084 | 0.90 [0.81, 1.01] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | B|full|text | C|sum_cp|text | 48 | 0.076 [0.065, 0.087] | 0.053 [0.041, 0.064] | 0.128 | 0.59 [0.52, 0.67] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | B|full|text | C|sum_window:10|text | 48 | 0.076 [0.065, 0.087] | -0.009 [-0.017, -0.001] | 0.067 | 1.13 [1.01, 1.27] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | B|full|text | C|sum_window:20|text | 48 | 0.076 [0.065, 0.087] | -0.043 [-0.057, -0.030] | 0.032 | not interpretable: C-A = 0.032 does not exceed the prespecified threshold 0.05 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | B|full|text | C|sum_window:5|text | 48 | 0.076 [0.065, 0.087] | 0.067 [0.058, 0.075] | 0.142 | 0.53 [0.48, 0.58] |

## TOST: VLM + summary vs rule alone (margin +-0.02, alpha 0.05)

Equivalence here means only that the accuracy difference lies within the margin; it does not imply trial-by-trial rule following or identical internal computation (see agreement below).

- Secondary family InternVL3_5-38B-HF@7c830fc2/episodes_v3#e00b8793: members ['C|sum_all|text vs window-count rule (domains pooled)', 'C|sum_cp|text vs window-count rule (domains pooled)', 'C|sum_window:5|text vs window-count rule (domains pooled)']; K* = 5 (source analysis/h200_ho/dev_selection.json); margin 0.02; alpha 0.05; Holm (missing members counted as p = 1)

| model | split | slot | C arm | rule | n | acc VLM | acc rule | diff [90% CI] | p TOST | equiv. (uncorr.) | Holm p | equiv. (Holm) | role |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_all|text | window_counts | 48 | 0.633 | 0.632 | 0.001 [-0.001, 0.003] | 1.8e-21 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_all|text | bocpd_posterior | 48 | 0.633 | 0.677 | -0.044 [-0.051, -0.036] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_cp|image | window_counts | 48 | 0.515 | 0.678 | -0.163 [-0.181, -0.145] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_cp|image | bocpd_posterior | 48 | 0.515 | 0.677 | -0.162 [-0.178, -0.145] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_cp|text | window_counts | 48 | 0.678 | 0.678 | -0.000 [-0.001, 0.001] | 8.9e-37 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_cp|text | bocpd_posterior | 48 | 0.678 | 0.677 | 0.001 [-0.005, 0.007] | 9.6e-06 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_window:10|text | window_counts | 48 | 0.617 | 0.612 | 0.005 [0.001, 0.008] | 7.5e-09 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_window:10|text | bocpd_posterior | 48 | 0.617 | 0.677 | -0.060 [-0.067, -0.053] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_window:20|text | window_counts | 48 | 0.582 | 0.579 | 0.003 [-0.003, 0.008] | 2.1e-06 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_window:20|text | bocpd_posterior | 48 | 0.582 | 0.677 | -0.094 [-0.105, -0.084] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_window:5|text | window_counts | 48 | 0.692 | 0.692 | 0.000 [-0.001, 0.001] | 7.6e-40 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_window:5|text | bocpd_posterior | 48 | 0.692 | 0.677 | 0.015 [0.009, 0.021] | 0.1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_all|text | window_counts | 48 | 0.633 | 0.632 | 0.001 [-0.001, 0.004] | 3.0e-17 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_all|text | bocpd_posterior | 48 | 0.633 | 0.677 | -0.044 [-0.051, -0.037] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_cp|image | window_counts | 48 | 0.573 | 0.678 | -0.105 [-0.127, -0.084] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_cp|image | bocpd_posterior | 48 | 0.573 | 0.677 | -0.104 [-0.125, -0.083] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_cp|text | window_counts | 48 | 0.677 | 0.678 | -0.001 [-0.002, 0.000] | 1.1e-36 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_cp|text | bocpd_posterior | 48 | 0.677 | 0.677 | 0.001 [-0.006, 0.007] | 4.3e-06 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_window:10|text | window_counts | 48 | 0.615 | 0.612 | 0.003 [-0.000, 0.007] | 2.8e-10 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_window:10|text | bocpd_posterior | 48 | 0.615 | 0.677 | -0.061 [-0.069, -0.054] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_window:20|text | window_counts | 48 | 0.581 | 0.579 | 0.001 [-0.004, 0.007] | 4.2e-07 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_window:20|text | bocpd_posterior | 48 | 0.581 | 0.677 | -0.096 [-0.106, -0.085] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_window:5|text | window_counts | 48 | 0.691 | 0.692 | -0.001 [-0.001, 0.000] | 1.4e-39 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_window:5|text | bocpd_posterior | 48 | 0.691 | 0.677 | 0.015 [0.008, 0.021] | 0.082 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_all|text | window_counts | 48 | 0.633 | 0.632 | 0.001 [-0.001, 0.003] | 1.4e-20 | yes | 1.4e-20 | yes | secondary |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_all|text | bocpd_posterior | 48 | 0.633 | 0.677 | -0.044 [-0.051, -0.036] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_cp|image | window_counts | 48 | 0.544 | 0.678 | -0.134 [-0.151, -0.117] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_cp|image | bocpd_posterior | 48 | 0.544 | 0.677 | -0.133 [-0.148, -0.117] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_cp|text | window_counts | 48 | 0.677 | 0.678 | -0.001 [-0.001, 0.000] | 8.6e-44 | yes | 2.6e-43 | yes | secondary |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_cp|text | bocpd_posterior | 48 | 0.677 | 0.677 | 0.001 [-0.006, 0.007] | 6.0e-06 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_window:10|text | window_counts | 48 | 0.616 | 0.612 | 0.004 [0.001, 0.008] | 7.0e-10 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_window:10|text | bocpd_posterior | 48 | 0.616 | 0.677 | -0.061 [-0.068, -0.053] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_window:20|text | window_counts | 48 | 0.581 | 0.579 | 0.002 [-0.003, 0.007] | 6.0e-07 | yes | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_window:20|text | bocpd_posterior | 48 | 0.581 | 0.677 | -0.095 [-0.105, -0.085] | 1 | no | nan | - | exploratory |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_window:5|text | window_counts | 48 | 0.691 | 0.692 | -0.000 [-0.001, 0.000] | 9.7e-44 | yes | 2.6e-43 | yes | secondary |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_window:5|text | bocpd_posterior | 48 | 0.691 | 0.677 | 0.015 [0.009, 0.021] | 0.091 | no | nan | - | exploratory |

## Trial-level choice agreement: VLM + summary vs window-count rule (non-tied trials)

| model | split | slot | C arm | n ep | non-tied | agree (pooled) | agree ep-mean [95% CI] | disagree | VLM right | rule right | stable | volatile | VLM->A | VLM->B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_all|text | 48 | 3687 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_cp|image | 48 | 3770 | 0.521 | 0.521 [0.504, 0.538] | 1805 | 594 | 1211 | 887 | 918 | 487 | 1318 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_cp|text | 48 | 3770 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_window:10|text | 48 | 3487 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_window:20|text | 48 | 2945 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | pet | C|sum_window:5|text | 48 | 3776 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_all|text | 48 | 3687 | 0.999 | 0.999 [0.997, 1.000] | 4 | 2 | 2 | 2 | 2 | 0 | 4 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_cp|image | 48 | 3770 | 0.674 | 0.674 [0.616, 0.734] | 1230 | 417 | 813 | 622 | 608 | 319 | 911 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_cp|text | 48 | 3770 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_window:10|text | 48 | 3487 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_window:20|text | 48 | 2945 | 0.999 | 0.999 [0.998, 1.000] | 2 | 2 | 0 | 0 | 2 | 0 | 2 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | scenario | C|sum_window:5|text | 48 | 3776 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_all|text | 48 | 7374 | 0.999 | 0.999 [0.999, 1.000] | 4 | 2 | 2 | 2 | 2 | 0 | 4 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_cp|image | 48 | 7540 | 0.597 | 0.598 [0.564, 0.631] | 3035 | 1011 | 2024 | 1509 | 1526 | 806 | 2229 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_cp|text | 48 | 7540 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_window:10|text | 48 | 6974 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_window:20|text | 48 | 5890 | 1.000 | 1.000 [0.999, 1.000] | 2 | 2 | 0 | 0 | 2 | 0 | 2 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | all | C|sum_window:5|text | 48 | 7552 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Per-scenario gap (domain_slot = scenario only; pet reported separately above)

Each held-out text scenario has 8 episode pairs; the cross-scenario sign test has 6 units (one per text scenario; minimum two-sided p = 0.03125). Pet is reported separately and is not part of the cross-scenario test.

| model | split | C arm | scenario | n pairs | gap [95% CI] |
|---|---|---|---|---|---|
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_all|text | color | 8 | 0.055 [0.025, 0.089] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_all|text | medical | 8 | 0.087 [0.065, 0.117] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_all|text | news | 8 | 0.068 [0.040, 0.097] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_all|text | sports | 8 | 0.087 [0.060, 0.114] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_all|text | stock | 8 | 0.127 [0.106, 0.144] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_all|text | weather | 8 | 0.101 [0.070, 0.133] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_all|text | sign test | 6 scen. | 6 positive, p=0.031 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|image | color | 8 | 0.176 [0.122, 0.220] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|image | medical | 8 | 0.179 [0.146, 0.210] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|image | news | 8 | -0.009 [-0.028, 0.009] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|image | sports | 8 | 0.040 [0.011, 0.070] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|image | stock | 8 | 0.059 [0.022, 0.085] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|image | weather | 8 | 0.027 [-0.013, 0.066] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|image | sign test | 6 scen. | 5 positive, p=0.22 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|text | color | 8 | 0.103 [0.070, 0.130] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|text | medical | 8 | 0.134 [0.112, 0.161] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|text | news | 8 | 0.095 [0.062, 0.130] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|text | sports | 8 | 0.134 [0.112, 0.158] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|text | stock | 8 | 0.177 [0.152, 0.201] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|text | weather | 8 | 0.147 [0.098, 0.195] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|text | sign test | 6 scen. | 6 positive, p=0.031 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:10|text | color | 8 | 0.035 [0.011, 0.062] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:10|text | medical | 8 | 0.062 [0.032, 0.093] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:10|text | news | 8 | 0.044 [0.006, 0.082] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:10|text | sports | 8 | 0.063 [0.032, 0.095] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:10|text | stock | 8 | 0.125 [0.108, 0.142] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:10|text | weather | 8 | 0.090 [0.060, 0.117] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:10|text | sign test | 6 scen. | 6 positive, p=0.031 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:20|text | color | 8 | 0.002 [-0.036, 0.054] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:20|text | medical | 8 | 0.014 [-0.025, 0.063] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:20|text | news | 8 | 0.027 [-0.009, 0.057] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:20|text | sports | 8 | 0.033 [0.003, 0.062] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:20|text | stock | 8 | 0.093 [0.055, 0.130] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:20|text | weather | 8 | 0.043 [0.003, 0.081] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:20|text | sign test | 6 scen. | 6 positive, p=0.031 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:5|text | color | 8 | 0.106 [0.079, 0.134] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:5|text | medical | 8 | 0.142 [0.119, 0.169] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:5|text | news | 8 | 0.114 [0.073, 0.150] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:5|text | sports | 8 | 0.152 [0.114, 0.191] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:5|text | stock | 8 | 0.196 [0.179, 0.214] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:5|text | weather | 8 | 0.165 [0.125, 0.203] |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:5|text | sign test | 6 scen. | 6 positive, p=0.031 |

## History baselines (same trial mask as the reference A arm)

Baselines marked "yes" use generator information the VLM does not have and are not fair competitors.

| model | split | baseline | acc [95% CI] | n ep | uses generator info | description |
|---|---|---|---|---|---|---|
| InternVL3_5-38B-HF@7c830fc2 | held_out | truth_quota | 0.655 [0.650, 0.661] | 48 | yes | knows the generator's 5/5 truth quota per 10-trial block; predicts the label with more remaining quota and follows the reporter who reports it (no reliability tracking) |
| InternVL3_5-38B-HF@7c830fc2 | held_out | always_A | 0.648 [0.647, 0.650] | 48 | yes | diagnostic only: always follows latent role A (role identity is hidden from the VLM) |
| InternVL3_5-38B-HF@7c830fc2 | held_out | known_pA | 0.800 [0.799, 0.801] | 48 | yes | knows the true current p_A and follows the more reliable reporter (reference ceiling) |
| InternVL3_5-38B-HF@7c830fc2 | held_out | bocpd_posterior | 0.677 [0.669, 0.684] | 48 | no | BOCPD posterior predictive p_A from the outcome history |
| InternVL3_5-38B-HF@7c830fc2 | held_out | bocpd_map_window | 0.678 [0.667, 0.688] | 48 | no | majority within the BOCPD MAP run-length window (= the sum_cp summary rule) |
| InternVL3_5-38B-HF@7c830fc2 | held_out | count_all | 0.632 [0.628, 0.636] | 48 | no | majority of all past outcomes (= the sum_all summary rule) |

## Readout quality (candidate mass, greedy output follows candidate format)

| model | split | arm | slot | format | n | mass mean | mass min | frac mass<0.5 | greedy follows |
|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-38B-HF@7c830fc2 | held_out | A+|full|text | pet | text | 7680 | 1.000 | 0.950 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | A+|full|text | scenario | text | 7680 | 1.000 | 0.958 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | A|full|image | pet | image | 7680 | 1.000 | 0.995 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | A|full|image | scenario | image | 7680 | 1.000 | 0.998 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | A|full|text | pet | text | 7680 | 1.000 | 0.995 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | A|full|text | scenario | text | 7680 | 1.000 | 0.985 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | B'|full|chart | pet | chart | 7680 | 1.000 | 0.980 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | B'|full|chart | scenario | chart | 7680 | 1.000 | 0.950 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | B|full|image | pet | image | 7680 | 1.000 | 0.855 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | B|full|image | scenario | image | 7680 | 1.000 | 0.998 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | B|full|text | pet | text | 7680 | 1.000 | 0.994 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | B|full|text | scenario | text | 7680 | 1.000 | 0.985 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_all|text | pet | text | 7680 | 1.000 | 0.990 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_all|text | scenario | text | 7680 | 1.000 | 0.985 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|image | pet | image | 7680 | 1.000 | 0.684 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|image | scenario | image | 7680 | 1.000 | 0.999 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|text | pet | text | 7680 | 1.000 | 0.990 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_cp|text | scenario | text | 7680 | 1.000 | 0.985 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:10|text | pet | text | 7680 | 1.000 | 0.990 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:10|text | scenario | text | 7680 | 1.000 | 0.985 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:20|text | pet | text | 7680 | 1.000 | 0.990 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:20|text | scenario | text | 7680 | 1.000 | 0.985 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:5|text | pet | text | 7680 | 1.000 | 0.988 | 0.000 | 1.000 |
| InternVL3_5-38B-HF@7c830fc2 | held_out | C|sum_window:5|text | scenario | text | 7680 | 1.000 | 0.985 | 0.000 | 1.000 |

Rule-window mismatches (recomputed vs logged window_start): 0; identical duplicate rows merged: 0; legacy y-only rows (approx logits): 0.
