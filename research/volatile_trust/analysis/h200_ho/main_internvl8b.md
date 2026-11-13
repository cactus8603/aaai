# Raw vs. summarized history gap (analyze_main.py)

mode: formal; runs: ho_full_chart, ho_full_coded, ho_full_image, ho_full_rawcoded, ho_full_text, ho_sum_all, ho_sum_cp, ho_sum_window10, ho_sum_window20, ho_sum_window5

## Episode-level metrics (twin-averaged in logit space, t >= 2)

"all" = equal-weight mean of the pet and scenario slots. approx = share of trials whose logit was reconstructed from clipped y (legacy rows; saturates at +-13.8).

| model | data | split | arm | slot | n ep | acc [95% CI] | FB stable | FB volatile | log score | AUC10 | approx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A+|full|text | pet | 48 | 0.491 [0.474, 0.507] | 0.598 | 0.506 | -1.024 | 0.499 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A+|full|text | scenario | 48 | 0.525 [0.509, 0.541] | 0.591 | 0.509 | -0.916 | 0.503 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A+|full|text | all | 48 | 0.508 [0.494, 0.521] | 0.594 | 0.508 | -0.970 | 0.501 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|full|image | pet | 48 | 0.453 [0.441, 0.464] | 0.491 | 0.486 | -0.903 | 0.486 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|full|image | scenario | 48 | 0.487 [0.473, 0.500] | 0.509 | 0.497 | -0.818 | 0.496 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|full|image | all | 48 | 0.470 [0.460, 0.480] | 0.500 | 0.492 | -0.861 | 0.491 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|full|text | pet | 48 | 0.460 [0.445, 0.475] | 0.540 | 0.493 | -0.933 | 0.490 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|full|text | scenario | 48 | 0.497 [0.482, 0.512] | 0.535 | 0.504 | -0.845 | 0.502 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|full|text | all | 48 | 0.479 [0.468, 0.489] | 0.537 | 0.499 | -0.889 | 0.496 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | B'|full|chart | pet | 48 | 0.529 [0.513, 0.546] | 0.601 | 0.503 | -0.750 | 0.495 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | B'|full|chart | scenario | 48 | 0.540 [0.521, 0.558] | 0.579 | 0.502 | -0.724 | 0.494 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | B'|full|chart | all | 48 | 0.534 [0.523, 0.546] | 0.590 | 0.502 | -0.737 | 0.495 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | B|full|image | pet | 48 | 0.546 [0.528, 0.565] | 0.673 | 0.509 | -1.207 | 0.503 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | B|full|image | scenario | 48 | 0.542 [0.520, 0.562] | 0.624 | 0.510 | -1.041 | 0.511 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | B|full|image | all | 48 | 0.544 [0.528, 0.561] | 0.648 | 0.509 | -1.124 | 0.507 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | B|full|text | pet | 48 | 0.639 [0.630, 0.646] | 0.910 | 0.512 | -1.444 | 0.519 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | B|full|text | scenario | 48 | 0.633 [0.624, 0.640] | 0.843 | 0.502 | -1.031 | 0.508 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | B|full|text | all | 48 | 0.636 [0.628, 0.642] | 0.877 | 0.507 | -1.238 | 0.513 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_all|text | pet | 48 | 0.639 [0.634, 0.643] | 0.912 | 0.499 | -1.129 | 0.484 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_all|text | scenario | 48 | 0.631 [0.624, 0.636] | 0.838 | 0.495 | -0.944 | 0.480 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_all|text | all | 48 | 0.635 [0.630, 0.638] | 0.875 | 0.497 | -1.036 | 0.482 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_cp|image | pet | 48 | 0.648 [0.634, 0.661] | 0.820 | 0.636 | -1.027 | 0.620 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_cp|image | scenario | 48 | 0.628 [0.611, 0.643] | 0.727 | 0.596 | -1.032 | 0.590 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_cp|image | all | 48 | 0.638 [0.625, 0.650] | 0.773 | 0.616 | -1.029 | 0.605 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_cp|text | pet | 48 | 0.678 [0.667, 0.688] | 0.896 | 0.685 | -1.093 | 0.662 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_cp|text | scenario | 48 | 0.669 [0.658, 0.680] | 0.834 | 0.642 | -0.902 | 0.624 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_cp|text | all | 48 | 0.673 [0.662, 0.684] | 0.865 | 0.663 | -0.998 | 0.643 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_window:10|text | pet | 48 | 0.617 [0.611, 0.622] | 0.890 | 0.498 | -1.141 | 0.445 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_window:10|text | scenario | 48 | 0.609 [0.602, 0.616] | 0.813 | 0.495 | -0.942 | 0.453 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_window:10|text | all | 48 | 0.613 [0.608, 0.617] | 0.852 | 0.496 | -1.042 | 0.449 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_window:20|text | pet | 48 | 0.590 [0.579, 0.601] | 0.883 | 0.465 | -0.941 | 0.424 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_window:20|text | scenario | 48 | 0.581 [0.572, 0.590] | 0.817 | 0.472 | -0.842 | 0.440 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_window:20|text | all | 48 | 0.586 [0.578, 0.594] | 0.850 | 0.468 | -0.891 | 0.432 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_window:5|text | pet | 48 | 0.692 [0.686, 0.698] | 0.868 | 0.679 | -0.899 | 0.654 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_window:5|text | scenario | 48 | 0.670 [0.659, 0.681] | 0.782 | 0.639 | -0.793 | 0.620 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_window:5|text | all | 48 | 0.681 [0.674, 0.688] | 0.825 | 0.659 | -0.846 | 0.637 | 0.00 |

## Gap: acc(C) - acc(A), paired by episode

| model | split | slot | C arm | A arm | n | gap [95% CI] | Wilcoxon p | Holm p | role |
|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_all|text | A|full|text | 48 | 0.179 [0.164, 0.194] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_cp|image | A|full|image | 48 | 0.195 [0.177, 0.213] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_cp|text | A|full|text | 48 | 0.218 [0.199, 0.236] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_window:10|text | A|full|text | 48 | 0.157 [0.141, 0.174] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_window:20|text | A|full|text | 48 | 0.130 [0.112, 0.149] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_window:5|text | A|full|text | 48 | 0.232 [0.214, 0.249] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_all|text | A|full|text | 48 | 0.133 [0.118, 0.149] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_cp|image | A|full|image | 48 | 0.141 [0.122, 0.160] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_cp|text | A|full|text | 48 | 0.171 [0.153, 0.190] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_window:10|text | A|full|text | 48 | 0.111 [0.095, 0.128] | 2.4e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_window:20|text | A|full|text | 48 | 0.084 [0.069, 0.100] | 4.2e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_window:5|text | A|full|text | 48 | 0.173 [0.154, 0.193] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_all|text | A|full|text | 48 | 0.156 [0.145, 0.167] | 1.6e-09 | 1.6e-09 | primary |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_cp|image | A|full|image | 48 | 0.168 [0.153, 0.182] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_cp|text | A|full|text | 48 | 0.195 [0.179, 0.210] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_window:10|text | A|full|text | 48 | 0.134 [0.123, 0.145] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_window:20|text | A|full|text | 48 | 0.107 [0.095, 0.119] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_window:5|text | A|full|text | 48 | 0.202 [0.188, 0.216] | 1.6e-09 | nan | exploratory |

Primary family (Wilcoxon signed-rank (two-sided), Holm, alpha=0.05): InternVL3_5-8B-HF@741a7d03: C|sum_all|text - A|full|text (all)

## B / B' vs A and C vs B (absolute accuracy differences)

| model | split | slot | B arm | A arm | n | B-A [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | A+|full|text | A|full|text | 48 | 0.031 [0.016, 0.046] | 0.00039 |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | B'|full|chart | A|full|text | 48 | 0.069 [0.050, 0.088] | 2.8e-07 |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | B|full|image | A|full|image | 48 | 0.094 [0.075, 0.112] | 7.0e-09 |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | B|full|text | A|full|text | 48 | 0.179 [0.162, 0.194] | 1.7e-09 |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | A+|full|text | A|full|text | 48 | 0.028 [0.008, 0.047] | 0.0067 |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | B'|full|chart | A|full|text | 48 | 0.042 [0.018, 0.066] | 0.0019 |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | B|full|image | A|full|image | 48 | 0.055 [0.037, 0.074] | 5.1e-06 |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | B|full|text | A|full|text | 48 | 0.135 [0.117, 0.153] | 1.9e-09 |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | A+|full|text | A|full|text | 48 | 0.029 [0.016, 0.043] | 0.00038 |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | B'|full|chart | A|full|text | 48 | 0.056 [0.042, 0.069] | 1.8e-07 |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | B|full|image | A|full|image | 48 | 0.074 [0.061, 0.089] | 4.3e-09 |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | B|full|text | A|full|text | 48 | 0.157 [0.143, 0.170] | 1.7e-09 |

Normalised position (B-A)/(C-A) is shown only when C-A > 0.05 and the bootstrap C-A stays > 0; it need not lie in [0, 1].

| model | split | slot | B arm | C arm | n | B-A [95% CI] | C-B [95% CI] | C-A | position [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | A+|full|text | C|sum_all|text | 48 | 0.031 [0.016, 0.046] | 0.148 [0.132, 0.164] | 0.179 | 0.17 [0.09, 0.25] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | A+|full|text | C|sum_cp|text | 48 | 0.031 [0.016, 0.046] | 0.187 [0.169, 0.204] | 0.218 | 0.14 [0.08, 0.21] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | A+|full|text | C|sum_window:10|text | 48 | 0.031 [0.016, 0.046] | 0.126 [0.109, 0.144] | 0.157 | 0.20 [0.10, 0.29] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | A+|full|text | C|sum_window:20|text | 48 | 0.031 [0.016, 0.046] | 0.099 [0.079, 0.120] | 0.130 | 0.24 [0.13, 0.35] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | A+|full|text | C|sum_window:5|text | 48 | 0.031 [0.016, 0.046] | 0.201 [0.182, 0.219] | 0.232 | 0.13 [0.07, 0.20] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | B'|full|chart | C|sum_all|text | 48 | 0.069 [0.050, 0.088] | 0.109 [0.094, 0.125] | 0.179 | 0.39 [0.29, 0.48] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | B'|full|chart | C|sum_cp|text | 48 | 0.069 [0.050, 0.088] | 0.148 [0.128, 0.169] | 0.218 | 0.32 [0.24, 0.40] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | B'|full|chart | C|sum_window:10|text | 48 | 0.069 [0.050, 0.088] | 0.088 [0.072, 0.104] | 0.157 | 0.44 [0.34, 0.54] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | B'|full|chart | C|sum_window:20|text | 48 | 0.069 [0.050, 0.088] | 0.061 [0.047, 0.075] | 0.130 | 0.53 [0.42, 0.63] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | B'|full|chart | C|sum_window:5|text | 48 | 0.069 [0.050, 0.088] | 0.162 [0.144, 0.181] | 0.232 | 0.30 [0.22, 0.37] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | B|full|image | C|sum_cp|image | 48 | 0.094 [0.075, 0.112] | 0.101 [0.082, 0.121] | 0.195 | 0.48 [0.40, 0.57] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | B|full|text | C|sum_all|text | 48 | 0.179 [0.162, 0.194] | 0.000 [-0.007, 0.008] | 0.179 | 1.00 [0.95, 1.04] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | B|full|text | C|sum_cp|text | 48 | 0.179 [0.162, 0.194] | 0.039 [0.028, 0.050] | 0.218 | 0.82 [0.77, 0.87] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | B|full|text | C|sum_window:10|text | 48 | 0.179 [0.162, 0.194] | -0.022 [-0.030, -0.013] | 0.157 | 1.14 [1.08, 1.20] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | B|full|text | C|sum_window:20|text | 48 | 0.179 [0.162, 0.194] | -0.049 [-0.061, -0.036] | 0.130 | 1.37 [1.25, 1.52] |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | B|full|text | C|sum_window:5|text | 48 | 0.179 [0.162, 0.194] | 0.053 [0.044, 0.062] | 0.232 | 0.77 [0.73, 0.81] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | A+|full|text | C|sum_all|text | 48 | 0.028 [0.008, 0.047] | 0.105 [0.089, 0.122] | 0.133 | 0.21 [0.06, 0.33] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | A+|full|text | C|sum_cp|text | 48 | 0.028 [0.008, 0.047] | 0.144 [0.124, 0.163] | 0.171 | 0.16 [0.05, 0.26] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | A+|full|text | C|sum_window:10|text | 48 | 0.028 [0.008, 0.047] | 0.084 [0.066, 0.101] | 0.111 | 0.25 [0.08, 0.40] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | A+|full|text | C|sum_window:20|text | 48 | 0.028 [0.008, 0.047] | 0.056 [0.039, 0.074] | 0.084 | 0.33 [0.10, 0.53] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | A+|full|text | C|sum_window:5|text | 48 | 0.028 [0.008, 0.047] | 0.145 [0.126, 0.165] | 0.173 | 0.16 [0.05, 0.26] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | B'|full|chart | C|sum_all|text | 48 | 0.042 [0.018, 0.066] | 0.091 [0.074, 0.108] | 0.133 | 0.32 [0.14, 0.47] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | B'|full|chart | C|sum_cp|text | 48 | 0.042 [0.018, 0.066] | 0.129 [0.111, 0.148] | 0.171 | 0.25 [0.11, 0.37] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | B'|full|chart | C|sum_window:10|text | 48 | 0.042 [0.018, 0.066] | 0.069 [0.050, 0.088] | 0.111 | 0.38 [0.17, 0.56] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | B'|full|chart | C|sum_window:20|text | 48 | 0.042 [0.018, 0.066] | 0.042 [0.022, 0.061] | 0.084 | 0.50 [0.23, 0.74] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | B'|full|chart | C|sum_window:5|text | 48 | 0.042 [0.018, 0.066] | 0.131 [0.112, 0.150] | 0.173 | 0.24 [0.11, 0.36] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | B|full|image | C|sum_cp|image | 48 | 0.055 [0.037, 0.074] | 0.086 [0.065, 0.107] | 0.141 | 0.39 [0.27, 0.51] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | B|full|text | C|sum_all|text | 48 | 0.135 [0.117, 0.153] | -0.002 [-0.011, 0.007] | 0.133 | 1.02 [0.95, 1.09] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | B|full|text | C|sum_cp|text | 48 | 0.135 [0.117, 0.153] | 0.036 [0.023, 0.049] | 0.171 | 0.79 [0.72, 0.86] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | B|full|text | C|sum_window:10|text | 48 | 0.135 [0.117, 0.153] | -0.024 [-0.033, -0.015] | 0.111 | 1.22 [1.13, 1.31] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | B|full|text | C|sum_window:20|text | 48 | 0.135 [0.117, 0.153] | -0.051 [-0.065, -0.038] | 0.084 | 1.61 [1.42, 1.86] |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | B|full|text | C|sum_window:5|text | 48 | 0.135 [0.117, 0.153] | 0.038 [0.025, 0.049] | 0.173 | 0.78 [0.72, 0.85] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | A+|full|text | C|sum_all|text | 48 | 0.029 [0.016, 0.043] | 0.127 [0.114, 0.139] | 0.156 | 0.19 [0.10, 0.27] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | A+|full|text | C|sum_cp|text | 48 | 0.029 [0.016, 0.043] | 0.165 [0.149, 0.181] | 0.195 | 0.15 [0.08, 0.22] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | A+|full|text | C|sum_window:10|text | 48 | 0.029 [0.016, 0.043] | 0.105 [0.091, 0.118] | 0.134 | 0.22 [0.12, 0.31] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | A+|full|text | C|sum_window:20|text | 48 | 0.029 [0.016, 0.043] | 0.078 [0.062, 0.093] | 0.107 | 0.27 [0.15, 0.40] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | A+|full|text | C|sum_window:5|text | 48 | 0.029 [0.016, 0.043] | 0.173 [0.157, 0.189] | 0.202 | 0.14 [0.08, 0.21] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | B'|full|chart | C|sum_all|text | 48 | 0.056 [0.042, 0.069] | 0.100 [0.090, 0.111] | 0.156 | 0.36 [0.28, 0.43] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | B'|full|chart | C|sum_cp|text | 48 | 0.056 [0.042, 0.069] | 0.139 [0.124, 0.153] | 0.195 | 0.29 [0.22, 0.35] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | B'|full|chart | C|sum_window:10|text | 48 | 0.056 [0.042, 0.069] | 0.078 [0.066, 0.091] | 0.134 | 0.42 [0.32, 0.50] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | B'|full|chart | C|sum_window:20|text | 48 | 0.056 [0.042, 0.069] | 0.051 [0.040, 0.062] | 0.107 | 0.52 [0.41, 0.62] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | B'|full|chart | C|sum_window:5|text | 48 | 0.056 [0.042, 0.069] | 0.147 [0.132, 0.161] | 0.202 | 0.28 [0.21, 0.34] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | B|full|image | C|sum_cp|image | 48 | 0.074 [0.061, 0.089] | 0.094 [0.077, 0.111] | 0.168 | 0.44 [0.37, 0.52] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | B|full|text | C|sum_all|text | 48 | 0.157 [0.143, 0.170] | -0.001 [-0.007, 0.006] | 0.156 | 1.01 [0.96, 1.05] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | B|full|text | C|sum_cp|text | 48 | 0.157 [0.143, 0.170] | 0.038 [0.026, 0.049] | 0.195 | 0.81 [0.76, 0.86] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | B|full|text | C|sum_window:10|text | 48 | 0.157 [0.143, 0.170] | -0.023 [-0.029, -0.016] | 0.134 | 1.17 [1.12, 1.22] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | B|full|text | C|sum_window:20|text | 48 | 0.157 [0.143, 0.170] | -0.050 [-0.061, -0.039] | 0.107 | 1.47 [1.35, 1.61] |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | B|full|text | C|sum_window:5|text | 48 | 0.157 [0.143, 0.170] | 0.045 [0.038, 0.053] | 0.202 | 0.78 [0.74, 0.81] |

## TOST: VLM + summary vs rule alone (margin +-0.02, alpha 0.05)

Equivalence here means only that the accuracy difference lies within the margin; it does not imply trial-by-trial rule following or identical internal computation (see agreement below).

- Secondary family InternVL3_5-8B-HF@741a7d03/episodes_v3#e00b8793: members ['C|sum_all|text vs window-count rule (domains pooled)', 'C|sum_cp|text vs window-count rule (domains pooled)', 'C|sum_window:5|text vs window-count rule (domains pooled)']; K* = 5 (source analysis/h200_ho/dev_selection.json); margin 0.02; alpha 0.05; Holm (missing members counted as p = 1)

| model | split | slot | C arm | rule | n | acc VLM | acc rule | diff [90% CI] | p TOST | equiv. (uncorr.) | Holm p | equiv. (Holm) | role |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_all|text | window_counts | 48 | 0.639 | 0.632 | 0.007 [0.004, 0.011] | 2.1e-07 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_all|text | bocpd_posterior | 48 | 0.639 | 0.677 | -0.038 [-0.045, -0.031] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_cp|image | window_counts | 48 | 0.648 | 0.678 | -0.030 [-0.041, -0.020] | 0.94 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_cp|image | bocpd_posterior | 48 | 0.648 | 0.677 | -0.029 [-0.040, -0.018] | 0.91 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_cp|text | window_counts | 48 | 0.678 | 0.678 | -0.000 [-0.001, 0.001] | 8.9e-37 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_cp|text | bocpd_posterior | 48 | 0.678 | 0.677 | 0.001 [-0.005, 0.007] | 8.3e-06 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_window:10|text | window_counts | 48 | 0.617 | 0.612 | 0.005 [0.001, 0.009] | 8.9e-09 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_window:10|text | bocpd_posterior | 48 | 0.617 | 0.677 | -0.060 [-0.067, -0.053] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_window:20|text | window_counts | 48 | 0.590 | 0.579 | 0.011 [0.005, 0.017] | 0.007 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_window:20|text | bocpd_posterior | 48 | 0.590 | 0.677 | -0.086 [-0.097, -0.076] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_window:5|text | window_counts | 48 | 0.692 | 0.692 | 0.000 [-0.001, 0.001] | 7.6e-40 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_window:5|text | bocpd_posterior | 48 | 0.692 | 0.677 | 0.015 [0.009, 0.021] | 0.1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_all|text | window_counts | 48 | 0.631 | 0.632 | -0.001 [-0.005, 0.003] | 2.7e-10 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_all|text | bocpd_posterior | 48 | 0.631 | 0.677 | -0.046 [-0.054, -0.038] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_cp|image | window_counts | 48 | 0.628 | 0.678 | -0.050 [-0.064, -0.037] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_cp|image | bocpd_posterior | 48 | 0.628 | 0.677 | -0.049 [-0.062, -0.036] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_cp|text | window_counts | 48 | 0.669 | 0.678 | -0.009 [-0.014, -0.005] | 0.00015 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_cp|text | bocpd_posterior | 48 | 0.669 | 0.677 | -0.008 [-0.015, -0.001] | 0.0056 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_window:10|text | window_counts | 48 | 0.609 | 0.612 | -0.003 [-0.009, 0.002] | 4.9e-06 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_window:10|text | bocpd_posterior | 48 | 0.609 | 0.677 | -0.068 [-0.077, -0.059] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_window:20|text | window_counts | 48 | 0.581 | 0.579 | 0.002 [-0.005, 0.009] | 7.4e-05 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_window:20|text | bocpd_posterior | 48 | 0.581 | 0.677 | -0.095 [-0.106, -0.085] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_window:5|text | window_counts | 48 | 0.670 | 0.692 | -0.021 [-0.031, -0.013] | 0.59 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_window:5|text | bocpd_posterior | 48 | 0.670 | 0.677 | -0.006 [-0.018, 0.005] | 0.027 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_all|text | window_counts | 48 | 0.635 | 0.632 | 0.003 [0.000, 0.006] | 2.2e-14 | yes | 4.4e-14 | yes | secondary |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_all|text | bocpd_posterior | 48 | 0.635 | 0.677 | -0.042 [-0.049, -0.035] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_cp|image | window_counts | 48 | 0.638 | 0.678 | -0.040 [-0.050, -0.031] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_cp|image | bocpd_posterior | 48 | 0.638 | 0.677 | -0.039 [-0.049, -0.029] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_cp|text | window_counts | 48 | 0.673 | 0.678 | -0.005 [-0.007, -0.003] | 5.5e-15 | yes | 1.7e-14 | yes | secondary |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_cp|text | bocpd_posterior | 48 | 0.673 | 0.677 | -0.003 [-0.010, 0.003] | 7.8e-05 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_window:10|text | window_counts | 48 | 0.613 | 0.612 | 0.001 [-0.002, 0.004] | 6.4e-15 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_window:10|text | bocpd_posterior | 48 | 0.613 | 0.677 | -0.064 [-0.071, -0.057] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_window:20|text | window_counts | 48 | 0.586 | 0.579 | 0.006 [0.002, 0.010] | 4.8e-07 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_window:20|text | bocpd_posterior | 48 | 0.586 | 0.677 | -0.091 [-0.100, -0.081] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_window:5|text | window_counts | 48 | 0.681 | 0.692 | -0.011 [-0.016, -0.006] | 0.00098 | yes | 0.00098 | yes | secondary |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_window:5|text | bocpd_posterior | 48 | 0.681 | 0.677 | 0.004 [-0.003, 0.012] | 0.0011 | yes | nan | - | exploratory |

## Trial-level choice agreement: VLM + summary vs window-count rule (non-tied trials)

| model | split | slot | C arm | n ep | non-tied | agree (pooled) | agree ep-mean [95% CI] | disagree | VLM right | rule right | stable | volatile | VLM->A | VLM->B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_all|text | 48 | 3687 | 0.989 | 0.989 [0.984, 0.993] | 40 | 31 | 9 | 8 | 32 | 5 | 35 |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_cp|image | 48 | 3770 | 0.914 | 0.914 [0.899, 0.927] | 324 | 105 | 219 | 137 | 187 | 103 | 221 |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_cp|text | 48 | 3770 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_window:10|text | 48 | 3487 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_window:20|text | 48 | 2945 | 0.984 | 0.984 [0.977, 0.990] | 46 | 38 | 8 | 1 | 45 | 11 | 35 |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_window:5|text | 48 | 3776 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_all|text | 48 | 3687 | 0.968 | 0.967 [0.948, 0.982] | 119 | 55 | 64 | 59 | 60 | 10 | 109 |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_cp|image | 48 | 3770 | 0.854 | 0.854 [0.818, 0.889] | 550 | 179 | 371 | 238 | 312 | 168 | 382 |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_cp|text | 48 | 3770 | 0.970 | 0.970 [0.959, 0.980] | 114 | 38 | 76 | 56 | 58 | 41 | 73 |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_window:10|text | 48 | 3487 | 0.978 | 0.978 [0.966, 0.988] | 78 | 37 | 41 | 25 | 53 | 26 | 52 |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_window:20|text | 48 | 2945 | 0.960 | 0.960 [0.945, 0.974] | 118 | 72 | 46 | 29 | 89 | 39 | 79 |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_window:5|text | 48 | 3776 | 0.954 | 0.954 [0.933, 0.972] | 175 | 46 | 129 | 77 | 98 | 61 | 114 |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_all|text | 48 | 7374 | 0.978 | 0.978 [0.968, 0.986] | 159 | 86 | 73 | 67 | 92 | 15 | 144 |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_cp|image | 48 | 7540 | 0.884 | 0.884 [0.864, 0.903] | 874 | 284 | 590 | 375 | 499 | 271 | 603 |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_cp|text | 48 | 7540 | 0.985 | 0.985 [0.979, 0.990] | 114 | 38 | 76 | 56 | 58 | 41 | 73 |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_window:10|text | 48 | 6974 | 0.989 | 0.989 [0.983, 0.994] | 78 | 37 | 41 | 25 | 53 | 26 | 52 |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_window:20|text | 48 | 5890 | 0.972 | 0.972 [0.964, 0.980] | 164 | 110 | 54 | 30 | 134 | 50 | 114 |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_window:5|text | 48 | 7552 | 0.977 | 0.977 [0.967, 0.986] | 175 | 46 | 129 | 77 | 98 | 61 | 114 |

## Per-scenario gap (domain_slot = scenario only; pet reported separately above)

Each held-out text scenario has 8 episode pairs; the cross-scenario sign test has 6 units (one per text scenario; minimum two-sided p = 0.03125). Pet is reported separately and is not part of the cross-scenario test.

| model | split | C arm | scenario | n pairs | gap [95% CI] |
|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_all|text | color | 8 | 0.172 [0.149, 0.193] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_all|text | medical | 8 | 0.101 [0.071, 0.131] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_all|text | news | 8 | 0.116 [0.082, 0.150] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_all|text | sports | 8 | 0.117 [0.082, 0.147] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_all|text | stock | 8 | 0.136 [0.101, 0.185] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_all|text | weather | 8 | 0.157 [0.117, 0.195] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_all|text | sign test | 6 scen. | 6 positive, p=0.031 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|image | color | 8 | 0.218 [0.190, 0.247] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|image | medical | 8 | 0.111 [0.098, 0.127] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|image | news | 8 | 0.142 [0.100, 0.184] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|image | sports | 8 | 0.160 [0.128, 0.188] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|image | stock | 8 | 0.120 [0.070, 0.169] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|image | weather | 8 | 0.095 [0.055, 0.138] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|image | sign test | 6 scen. | 6 positive, p=0.031 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|text | color | 8 | 0.214 [0.163, 0.255] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|text | medical | 8 | 0.155 [0.125, 0.190] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|text | news | 8 | 0.141 [0.111, 0.174] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|text | sports | 8 | 0.171 [0.123, 0.212] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|text | stock | 8 | 0.171 [0.117, 0.229] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|text | weather | 8 | 0.177 [0.141, 0.214] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|text | sign test | 6 scen. | 6 positive, p=0.031 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:10|text | color | 8 | 0.144 [0.109, 0.171] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:10|text | medical | 8 | 0.078 [0.051, 0.104] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:10|text | news | 8 | 0.095 [0.063, 0.130] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:10|text | sports | 8 | 0.109 [0.078, 0.136] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:10|text | stock | 8 | 0.112 [0.078, 0.158] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:10|text | weather | 8 | 0.130 [0.079, 0.177] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:10|text | sign test | 6 scen. | 6 positive, p=0.031 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:20|text | color | 8 | 0.117 [0.093, 0.142] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:20|text | medical | 8 | 0.059 [0.035, 0.084] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:20|text | news | 8 | 0.070 [0.040, 0.101] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:20|text | sports | 8 | 0.097 [0.068, 0.134] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:20|text | stock | 8 | 0.060 [0.024, 0.104] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:20|text | weather | 8 | 0.101 [0.052, 0.147] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:20|text | sign test | 6 scen. | 6 positive, p=0.031 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:5|text | color | 8 | 0.217 [0.172, 0.253] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:5|text | medical | 8 | 0.150 [0.120, 0.180] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:5|text | news | 8 | 0.160 [0.120, 0.199] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:5|text | sports | 8 | 0.119 [0.081, 0.166] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:5|text | stock | 8 | 0.199 [0.165, 0.245] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:5|text | weather | 8 | 0.193 [0.141, 0.242] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:5|text | sign test | 6 scen. | 6 positive, p=0.031 |

## History baselines (same trial mask as the reference A arm)

Baselines marked "yes" use generator information the VLM does not have and are not fair competitors.

| model | split | baseline | acc [95% CI] | n ep | uses generator info | description |
|---|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03 | held_out | truth_quota | 0.655 [0.650, 0.661] | 48 | yes | knows the generator's 5/5 truth quota per 10-trial block; predicts the label with more remaining quota and follows the reporter who reports it (no reliability tracking) |
| InternVL3_5-8B-HF@741a7d03 | held_out | always_A | 0.648 [0.647, 0.650] | 48 | yes | diagnostic only: always follows latent role A (role identity is hidden from the VLM) |
| InternVL3_5-8B-HF@741a7d03 | held_out | known_pA | 0.800 [0.799, 0.801] | 48 | yes | knows the true current p_A and follows the more reliable reporter (reference ceiling) |
| InternVL3_5-8B-HF@741a7d03 | held_out | bocpd_posterior | 0.677 [0.669, 0.684] | 48 | no | BOCPD posterior predictive p_A from the outcome history |
| InternVL3_5-8B-HF@741a7d03 | held_out | bocpd_map_window | 0.678 [0.667, 0.688] | 48 | no | majority within the BOCPD MAP run-length window (= the sum_cp summary rule) |
| InternVL3_5-8B-HF@741a7d03 | held_out | count_all | 0.632 [0.628, 0.636] | 48 | no | majority of all past outcomes (= the sum_all summary rule) |

## Readout quality (candidate mass, greedy output follows candidate format)

| model | split | arm | slot | format | n | mass mean | mass min | frac mass<0.5 | greedy follows |
|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03 | held_out | A+|full|text | pet | text | 7680 | 0.999 | 0.993 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | A+|full|text | scenario | text | 7680 | 0.999 | 0.982 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | A|full|image | pet | image | 7680 | 0.999 | 0.957 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | A|full|image | scenario | image | 7680 | 0.999 | 0.967 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | A|full|text | pet | text | 7680 | 0.999 | 0.997 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | A|full|text | scenario | text | 7680 | 0.999 | 0.988 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | B'|full|chart | pet | chart | 7680 | 0.999 | 0.928 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | B'|full|chart | scenario | chart | 7680 | 0.999 | 0.686 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | B|full|image | pet | image | 7680 | 0.987 | 0.177 | 0.001 | 0.998 |
| InternVL3_5-8B-HF@741a7d03 | held_out | B|full|image | scenario | image | 7680 | 0.996 | 0.658 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | B|full|text | pet | text | 7680 | 0.994 | 0.650 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | B|full|text | scenario | text | 7680 | 0.997 | 0.806 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_all|text | pet | text | 7680 | 0.996 | 0.875 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_all|text | scenario | text | 7680 | 0.998 | 0.952 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|image | pet | image | 7680 | 0.997 | 0.579 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|image | scenario | image | 7680 | 0.999 | 0.942 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|text | pet | text | 7680 | 0.995 | 0.875 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_cp|text | scenario | text | 7680 | 0.997 | 0.943 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:10|text | pet | text | 7680 | 0.995 | 0.883 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:10|text | scenario | text | 7680 | 0.997 | 0.938 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:20|text | pet | text | 7680 | 0.995 | 0.895 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:20|text | scenario | text | 7680 | 0.997 | 0.945 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:5|text | pet | text | 7680 | 0.996 | 0.909 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_window:5|text | scenario | text | 7680 | 0.998 | 0.974 | 0.000 | 1.000 |

Rule-window mismatches (recomputed vs logged window_start): 0; identical duplicate rows merged: 0; legacy y-only rows (approx logits): 0.
