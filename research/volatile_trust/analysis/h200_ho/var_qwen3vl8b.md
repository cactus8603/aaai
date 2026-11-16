# Raw vs. summarized history gap (analyze_main.py)

mode: formal; runs: ho_cp_hard, ho_full_text, ho_sum_all, ho_sum_oracle, ho_window10

## Episode-level metrics (twin-averaged in logit space, t >= 2)

"all" = equal-weight mean of the pet and scenario slots. approx = share of trials whose logit was reconstructed from clipped y (legacy rows; saturates at +-13.8).

| model | data | split | arm | slot | n ep | acc [95% CI] | FB stable | FB volatile | log score | AUC10 | approx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | A|cp_hard|text | pet | 48 | 0.479 [0.461, 0.496] | 0.553 | 0.521 | -1.997 | 0.515 | 0.00 |
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | A|cp_hard|text | scenario | 48 | 0.476 [0.459, 0.493] | 0.547 | 0.517 | -1.948 | 0.518 | 0.00 |
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | A|cp_hard|text | all | 48 | 0.477 [0.461, 0.493] | 0.550 | 0.519 | -1.973 | 0.516 | 0.00 |
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | A|full|text | pet | 48 | 0.485 [0.470, 0.499] | 0.562 | 0.484 | -1.537 | 0.482 | 0.00 |
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | A|full|text | scenario | 48 | 0.488 [0.472, 0.504] | 0.564 | 0.484 | -1.320 | 0.487 | 0.00 |
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | A|full|text | all | 48 | 0.486 [0.473, 0.500] | 0.563 | 0.484 | -1.428 | 0.484 | 0.00 |
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | A|window:10|text | pet | 48 | 0.456 [0.438, 0.474] | 0.523 | 0.474 | -2.171 | 0.460 | 0.00 |
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | A|window:10|text | scenario | 48 | 0.459 [0.443, 0.476] | 0.520 | 0.485 | -2.190 | 0.480 | 0.00 |
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | A|window:10|text | all | 48 | 0.458 [0.442, 0.474] | 0.521 | 0.480 | -2.180 | 0.470 | 0.00 |
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | C|sum_all|text | pet | 48 | 0.633 [0.628, 0.638] | 0.928 | 0.497 | -1.414 | 0.483 | 0.00 |
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | C|sum_all|text | scenario | 48 | 0.608 [0.594, 0.622] | 0.832 | 0.494 | -1.293 | 0.475 | 0.00 |
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | C|sum_all|text | all | 48 | 0.621 [0.612, 0.629] | 0.880 | 0.496 | -1.353 | 0.479 | 0.00 |
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | C|sum_oracle|text | pet | 48 | 0.744 [0.741, 0.748] | 0.967 | 0.820 | -1.259 | 0.808 | 0.00 |
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | C|sum_oracle|text | scenario | 48 | 0.710 [0.690, 0.728] | 0.885 | 0.745 | -1.157 | 0.732 | 0.00 |
| Qwen3-VL-8B-Instruct@0c351dd0 | episodes_v3#e00b8793 | held_out | C|sum_oracle|text | all | 48 | 0.727 [0.717, 0.737] | 0.926 | 0.783 | -1.208 | 0.770 | 0.00 |

## Gap: acc(C) - acc(A), paired by episode

| model | split | slot | C arm | A arm | n | gap [95% CI] | Wilcoxon p | Holm p | role |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | pet | C|sum_all|text | A|full|text | 48 | 0.148 [0.133, 0.164] | 1.7e-09 | nan | exploratory |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | pet | C|sum_oracle|text | A|full|text | 48 | 0.260 [0.243, 0.276] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | scenario | C|sum_all|text | A|full|text | 48 | 0.121 [0.104, 0.136] | 5.0e-09 | nan | exploratory |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | scenario | C|sum_oracle|text | A|full|text | 48 | 0.222 [0.201, 0.242] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | all | C|sum_all|text | A|full|text | 48 | 0.134 [0.122, 0.146] | 2.4e-09 | 2.4e-09 | primary |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | all | C|sum_oracle|text | A|full|text | 48 | 0.241 [0.227, 0.255] | 1.6e-09 | nan | exploratory |

Primary family (Wilcoxon signed-rank (two-sided), Holm, alpha=0.05): Qwen3-VL-8B-Instruct@0c351dd0: C|sum_all|text - A|full|text (all)

## B / B' vs A and C vs B (absolute accuracy differences)

| model | split | slot | B arm | A arm | n | B-A [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|---|

## TOST: VLM + summary vs rule alone (margin +-0.02, alpha 0.05)

Equivalence here means only that the accuracy difference lies within the margin; it does not imply trial-by-trial rule following or identical internal computation (see agreement below).

- Secondary family Qwen3-VL-8B-Instruct@0c351dd0/episodes_v3#e00b8793: members ['C|sum_all|text vs window-count rule (domains pooled)', 'C|sum_cp|text vs window-count rule (domains pooled)', 'C|sum_window:5|text vs window-count rule (domains pooled)']; K* = 5 (source analysis/h200_ho/dev_selection.json); margin 0.02; alpha 0.05; Holm (missing members counted as p = 1); MISSING ['sum_cp', 'sum_window:5']

| model | split | slot | C arm | rule | n | acc VLM | acc rule | diff [90% CI] | p TOST | equiv. (uncorr.) | Holm p | equiv. (Holm) | role |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | pet | C|sum_all|text | window_counts | 48 | 0.633 | 0.632 | 0.001 [-0.002, 0.005] | 1.5e-11 | yes | nan | - | exploratory |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | pet | C|sum_all|text | bocpd_posterior | 48 | 0.633 | 0.677 | -0.044 [-0.051, -0.036] | 1 | no | nan | - | exploratory |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | pet | C|sum_oracle|text | window_counts | 48 | 0.744 | 0.744 | 0.001 [-0.001, 0.003] | 2.4e-21 | yes | nan | - | exploratory |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | pet | C|sum_oracle|text | bocpd_posterior | 48 | 0.744 | 0.677 | 0.068 [0.061, 0.075] | 1 | no | nan | - | exploratory |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | scenario | C|sum_all|text | window_counts | 48 | 0.608 | 0.632 | -0.023 [-0.035, -0.013] | 0.69 | no | nan | - | exploratory |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | scenario | C|sum_all|text | bocpd_posterior | 48 | 0.608 | 0.677 | -0.068 [-0.081, -0.055] | 1 | no | nan | - | exploratory |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | scenario | C|sum_oracle|text | window_counts | 48 | 0.710 | 0.744 | -0.034 [-0.050, -0.019] | 0.92 | no | nan | - | exploratory |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | scenario | C|sum_oracle|text | bocpd_posterior | 48 | 0.710 | 0.677 | 0.033 [0.016, 0.050] | 0.9 | no | nan | - | exploratory |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | all | C|sum_all|text | window_counts | 48 | 0.621 | 0.632 | -0.011 [-0.017, -0.005] | 0.009 | yes | 0.027 | yes | secondary |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | all | C|sum_all|text | bocpd_posterior | 48 | 0.621 | 0.677 | -0.056 [-0.065, -0.047] | 1 | no | nan | - | exploratory |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | all | C|sum_oracle|text | window_counts | 48 | 0.727 | 0.744 | -0.017 [-0.025, -0.009] | 0.23 | no | nan | - | exploratory |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | all | C|sum_oracle|text | bocpd_posterior | 48 | 0.727 | 0.677 | 0.051 [0.040, 0.061] | 1 | no | nan | - | exploratory |

## Trial-level choice agreement: VLM + summary vs window-count rule (non-tied trials)

| model | split | slot | C arm | n ep | non-tied | agree (pooled) | agree ep-mean [95% CI] | disagree | VLM right | rule right | stable | volatile | VLM->A | VLM->B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | pet | C|sum_all|text | 48 | 3687 | 0.979 | 0.979 [0.966, 0.989] | 77 | 39 | 38 | 21 | 56 | 4 | 73 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | pet | C|sum_oracle|text | 48 | 3676 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | scenario | C|sum_all|text | 48 | 3687 | 0.903 | 0.902 [0.857, 0.941] | 357 | 129 | 228 | 168 | 189 | 5 | 352 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | scenario | C|sum_oracle|text | 48 | 3676 | 0.937 | 0.936 [0.898, 0.969] | 233 | 56 | 177 | 93 | 140 | 76 | 157 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | all | C|sum_all|text | 48 | 7374 | 0.941 | 0.940 [0.917, 0.961] | 434 | 168 | 266 | 189 | 245 | 9 | 425 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | all | C|sum_oracle|text | 48 | 7352 | 0.968 | 0.968 [0.949, 0.985] | 233 | 56 | 177 | 93 | 140 | 76 | 157 |

## Per-scenario gap (domain_slot = scenario only; pet reported separately above)

Each held-out text scenario has 8 episode pairs; the cross-scenario sign test has 6 units (one per text scenario; minimum two-sided p = 0.03125). Pet is reported separately and is not part of the cross-scenario test.

| model | split | C arm | scenario | n pairs | gap [95% CI] |
|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_all|text | color | 8 | 0.109 [0.077, 0.138] |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_all|text | medical | 8 | 0.097 [0.027, 0.152] |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_all|text | news | 8 | 0.134 [0.108, 0.161] |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_all|text | sports | 8 | 0.131 [0.103, 0.160] |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_all|text | stock | 8 | 0.157 [0.125, 0.193] |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_all|text | weather | 8 | 0.095 [0.070, 0.122] |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_all|text | sign test | 6 scen. | 6 positive, p=0.031 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_oracle|text | color | 8 | 0.217 [0.184, 0.245] |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_oracle|text | medical | 8 | 0.228 [0.171, 0.280] |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_oracle|text | news | 8 | 0.239 [0.210, 0.267] |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_oracle|text | sports | 8 | 0.247 [0.215, 0.278] |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_oracle|text | stock | 8 | 0.282 [0.245, 0.320] |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_oracle|text | weather | 8 | 0.122 [0.089, 0.153] |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_oracle|text | sign test | 6 scen. | 6 positive, p=0.031 |

## History baselines (same trial mask as the reference A arm)

Baselines marked "yes" use generator information the VLM does not have and are not fair competitors.

| model | split | baseline | acc [95% CI] | n ep | uses generator info | description |
|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | truth_quota | 0.655 [0.650, 0.661] | 48 | yes | knows the generator's 5/5 truth quota per 10-trial block; predicts the label with more remaining quota and follows the reporter who reports it (no reliability tracking) |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | always_A | 0.648 [0.647, 0.650] | 48 | yes | diagnostic only: always follows latent role A (role identity is hidden from the VLM) |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | known_pA | 0.800 [0.799, 0.801] | 48 | yes | knows the true current p_A and follows the more reliable reporter (reference ceiling) |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | bocpd_posterior | 0.677 [0.669, 0.684] | 48 | no | BOCPD posterior predictive p_A from the outcome history |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | bocpd_map_window | 0.678 [0.667, 0.688] | 48 | no | majority within the BOCPD MAP run-length window (= the sum_cp summary rule) |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | count_all | 0.632 [0.628, 0.636] | 48 | no | majority of all past outcomes (= the sum_all summary rule) |

## Readout quality (candidate mass, greedy output follows candidate format)

| model | split | arm | slot | format | n | mass mean | mass min | frac mass<0.5 | greedy follows |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | A|cp_hard|text | pet | text | 7680 | 1.000 | 1.000 | 0.000 | 1.000 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | A|cp_hard|text | scenario | text | 7680 | 0.999 | 0.587 | 0.000 | 1.000 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | A|full|text | pet | text | 7680 | 1.000 | 1.000 | 0.000 | 1.000 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | A|full|text | scenario | text | 7680 | 0.999 | 0.587 | 0.000 | 1.000 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | A|window:10|text | pet | text | 7680 | 1.000 | 1.000 | 0.000 | 1.000 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | A|window:10|text | scenario | text | 7680 | 0.999 | 0.587 | 0.000 | 1.000 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_all|text | pet | text | 7680 | 1.000 | 1.000 | 0.000 | 1.000 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_all|text | scenario | text | 7680 | 0.999 | 0.587 | 0.000 | 1.000 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_oracle|text | pet | text | 7680 | 1.000 | 1.000 | 0.000 | 1.000 |
| Qwen3-VL-8B-Instruct@0c351dd0 | held_out | C|sum_oracle|text | scenario | text | 7680 | 0.999 | 0.587 | 0.000 | 1.000 |

Rule-window mismatches (recomputed vs logged window_start): 0; identical duplicate rows merged: 0; legacy y-only rows (approx logits): 0.
