# Raw vs. summarized history gap (analyze_main.py)

mode: formal; runs: ho_full_text, ho_sum_all, ho_sum_oracle, ho_window10

## Episode-level metrics (twin-averaged in logit space, t >= 2)

"all" = equal-weight mean of the pet and scenario slots. approx = share of trials whose logit was reconstructed from clipped y (legacy rows; saturates at +-13.8).

| model | data | split | arm | slot | n ep | acc [95% CI] | FB stable | FB volatile | log score | AUC10 | approx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A|full|text | pet | 48 | 0.469 [0.456, 0.481] | 0.545 | 0.481 | -1.320 | 0.488 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A|full|text | scenario | 48 | 0.467 [0.449, 0.486] | 0.582 | 0.476 | -1.279 | 0.483 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A|full|text | all | 48 | 0.468 [0.454, 0.482] | 0.564 | 0.479 | -1.300 | 0.485 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A|window:10|text | pet | 48 | 0.428 [0.411, 0.444] | 0.488 | 0.462 | -2.886 | 0.463 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A|window:10|text | scenario | 48 | 0.429 [0.412, 0.446] | 0.495 | 0.468 | -2.806 | 0.466 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | A|window:10|text | all | 48 | 0.429 [0.412, 0.444] | 0.491 | 0.465 | -2.846 | 0.465 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_all|text | pet | 48 | 0.633 [0.629, 0.637] | 0.985 | 0.482 | -5.167 | 0.483 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_all|text | scenario | 48 | 0.630 [0.625, 0.635] | 0.966 | 0.481 | -3.463 | 0.482 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_all|text | all | 48 | 0.631 [0.627, 0.636] | 0.975 | 0.482 | -4.315 | 0.482 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_oracle|text | pet | 48 | 0.744 [0.741, 0.748] | 0.978 | 0.847 | -3.924 | 0.836 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_oracle|text | scenario | 48 | 0.736 [0.729, 0.743] | 0.960 | 0.824 | -2.650 | 0.813 | 0.00 |
| Qwen3-VL-32B-Instruct@0cfaf481 | episodes_v3#e00b8793 | held_out | C|sum_oracle|text | all | 48 | 0.740 [0.735, 0.745] | 0.969 | 0.836 | -3.287 | 0.825 | 0.00 |

## Gap: acc(C) - acc(A), paired by episode

| model | split | slot | C arm | A arm | n | gap [95% CI] | Wilcoxon p | Holm p | role |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_all|text | A|full|text | 48 | 0.164 [0.152, 0.177] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_oracle|text | A|full|text | 48 | 0.276 [0.264, 0.289] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_all|text | A|full|text | 48 | 0.163 [0.145, 0.180] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_oracle|text | A|full|text | 48 | 0.269 [0.249, 0.289] | 1.6e-09 | nan | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_all|text | A|full|text | 48 | 0.164 [0.150, 0.177] | 1.6e-09 | 1.6e-09 | primary |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_oracle|text | A|full|text | 48 | 0.272 [0.258, 0.287] | 1.6e-09 | nan | exploratory |

Primary family (Wilcoxon signed-rank (two-sided), Holm, alpha=0.05): Qwen3-VL-32B-Instruct@0cfaf481: C|sum_all|text - A|full|text (all)

## B / B' vs A and C vs B (absolute accuracy differences)

| model | split | slot | B arm | A arm | n | B-A [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|---|

## TOST: VLM + summary vs rule alone (margin +-0.02, alpha 0.05)

Equivalence here means only that the accuracy difference lies within the margin; it does not imply trial-by-trial rule following or identical internal computation (see agreement below).

- Secondary family Qwen3-VL-32B-Instruct@0cfaf481/episodes_v3#e00b8793: members ['C|sum_all|text vs window-count rule (domains pooled)', 'C|sum_cp|text vs window-count rule (domains pooled)', 'C|sum_window:5|text vs window-count rule (domains pooled)']; K* = 5 (source analysis/h200_ho/dev_selection.json); margin 0.02; alpha 0.05; Holm (missing members counted as p = 1); MISSING ['sum_cp', 'sum_window:5']

| model | split | slot | C arm | rule | n | acc VLM | acc rule | diff [90% CI] | p TOST | equiv. (uncorr.) | Holm p | equiv. (Holm) | role |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_all|text | window_counts | 48 | 0.633 | 0.632 | 0.001 [-0.001, 0.003] | 1.8e-21 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_all|text | bocpd_posterior | 48 | 0.633 | 0.677 | -0.044 [-0.051, -0.036] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_oracle|text | window_counts | 48 | 0.744 | 0.744 | 0.001 [-0.001, 0.003] | 2.4e-21 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_oracle|text | bocpd_posterior | 48 | 0.744 | 0.677 | 0.068 [0.061, 0.075] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_all|text | window_counts | 48 | 0.630 | 0.632 | -0.002 [-0.005, 0.001] | 5.2e-12 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_all|text | bocpd_posterior | 48 | 0.630 | 0.677 | -0.047 [-0.054, -0.039] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_oracle|text | window_counts | 48 | 0.736 | 0.744 | -0.008 [-0.013, -0.003] | 0.00015 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_oracle|text | bocpd_posterior | 48 | 0.736 | 0.677 | 0.060 [0.051, 0.068] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_all|text | window_counts | 48 | 0.631 | 0.632 | -0.000 [-0.003, 0.002] | 1.1e-18 | yes | 3.3e-18 | yes | secondary |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_all|text | bocpd_posterior | 48 | 0.631 | 0.677 | -0.045 [-0.053, -0.038] | 1 | no | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_oracle|text | window_counts | 48 | 0.740 | 0.744 | -0.004 [-0.007, -0.001] | 1.9e-11 | yes | nan | - | exploratory |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_oracle|text | bocpd_posterior | 48 | 0.740 | 0.677 | 0.064 [0.057, 0.071] | 1 | no | nan | - | exploratory |

## Trial-level choice agreement: VLM + summary vs window-count rule (non-tied trials)

| model | split | slot | C arm | n ep | non-tied | agree (pooled) | agree ep-mean [95% CI] | disagree | VLM right | rule right | stable | volatile | VLM->A | VLM->B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_all|text | 48 | 3687 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | pet | C|sum_oracle|text | 48 | 3676 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_all|text | 48 | 3687 | 0.989 | 0.989 [0.979, 0.996] | 41 | 15 | 26 | 19 | 22 | 0 | 41 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | scenario | C|sum_oracle|text | 48 | 3676 | 0.983 | 0.983 [0.974, 0.991] | 63 | 16 | 47 | 22 | 41 | 17 | 46 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_all|text | 48 | 7374 | 0.994 | 0.994 [0.989, 0.998] | 41 | 15 | 26 | 19 | 22 | 0 | 41 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | all | C|sum_oracle|text | 48 | 7352 | 0.991 | 0.991 [0.987, 0.995] | 63 | 16 | 47 | 22 | 41 | 17 | 46 |

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
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_oracle|text | color | 8 | 0.278 [0.244, 0.309] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_oracle|text | medical | 8 | 0.275 [0.212, 0.339] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_oracle|text | news | 8 | 0.196 [0.158, 0.231] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_oracle|text | sports | 8 | 0.247 [0.209, 0.288] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_oracle|text | stock | 8 | 0.313 [0.283, 0.339] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_oracle|text | weather | 8 | 0.304 [0.278, 0.332] |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_oracle|text | sign test | 6 scen. | 6 positive, p=0.031 |

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
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | A|full|text | pet | text | 7680 | 0.994 | 0.881 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | A|full|text | scenario | text | 7680 | 0.996 | 0.939 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | A|window:10|text | pet | text | 7680 | 0.992 | 0.881 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | A|window:10|text | scenario | text | 7680 | 0.996 | 0.939 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_all|text | pet | text | 7680 | 1.000 | 0.997 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_all|text | scenario | text | 7680 | 1.000 | 0.991 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_oracle|text | pet | text | 7680 | 1.000 | 0.996 | 0.000 | 1.000 |
| Qwen3-VL-32B-Instruct@0cfaf481 | held_out | C|sum_oracle|text | scenario | text | 7680 | 1.000 | 0.988 | 0.000 | 1.000 |

Rule-window mismatches (recomputed vs logged window_start): 0; identical duplicate rows merged: 0; legacy y-only rows (approx logits): 0.
