# Raw vs. summarized history gap (analyze_main.py)

mode: formal; runs: ho_cp_hard, ho_full_text, ho_sum_all, ho_sum_oracle, ho_window10

## Episode-level metrics (twin-averaged in logit space, t >= 2)

"all" = equal-weight mean of the pet and scenario slots. approx = share of trials whose logit was reconstructed from clipped y (legacy rows; saturates at +-13.8).

| model | data | split | arm | slot | n ep | acc [95% CI] | FB stable | FB volatile | log score | AUC10 | approx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|cp_hard|text | pet | 48 | 0.455 [0.440, 0.472] | 0.537 | 0.509 | -1.282 | 0.508 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|cp_hard|text | scenario | 48 | 0.489 [0.471, 0.507] | 0.544 | 0.527 | -1.014 | 0.525 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|cp_hard|text | all | 48 | 0.472 [0.459, 0.486] | 0.540 | 0.518 | -1.148 | 0.516 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|full|text | pet | 48 | 0.460 [0.445, 0.475] | 0.540 | 0.493 | -0.933 | 0.490 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|full|text | scenario | 48 | 0.497 [0.482, 0.512] | 0.535 | 0.504 | -0.845 | 0.502 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|full|text | all | 48 | 0.479 [0.468, 0.489] | 0.537 | 0.499 | -0.889 | 0.496 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|window:10|text | pet | 48 | 0.439 [0.425, 0.452] | 0.516 | 0.467 | -1.347 | 0.458 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|window:10|text | scenario | 48 | 0.495 [0.480, 0.509] | 0.542 | 0.493 | -0.999 | 0.485 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | A|window:10|text | all | 48 | 0.467 [0.455, 0.478] | 0.529 | 0.480 | -1.173 | 0.471 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_all|text | pet | 48 | 0.639 [0.634, 0.643] | 0.912 | 0.499 | -1.129 | 0.484 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_all|text | scenario | 48 | 0.631 [0.624, 0.636] | 0.838 | 0.495 | -0.944 | 0.480 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_all|text | all | 48 | 0.635 [0.630, 0.638] | 0.875 | 0.497 | -1.036 | 0.482 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_oracle|text | pet | 48 | 0.745 [0.741, 0.749] | 0.946 | 0.794 | -0.891 | 0.782 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_oracle|text | scenario | 48 | 0.729 [0.722, 0.737] | 0.874 | 0.714 | -0.759 | 0.704 | 0.00 |
| InternVL3_5-8B-HF@741a7d03 | episodes_v3#e00b8793 | held_out | C|sum_oracle|text | all | 48 | 0.737 [0.733, 0.741] | 0.910 | 0.754 | -0.825 | 0.743 | 0.00 |

## Gap: acc(C) - acc(A), paired by episode

| model | split | slot | C arm | A arm | n | gap [95% CI] | Wilcoxon p | Holm p | role |
|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_all|text | A|full|text | 48 | 0.179 [0.164, 0.194] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_oracle|text | A|full|text | 48 | 0.285 [0.270, 0.300] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_all|text | A|full|text | 48 | 0.133 [0.118, 0.149] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_oracle|text | A|full|text | 48 | 0.232 [0.217, 0.248] | 1.6e-09 | nan | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_all|text | A|full|text | 48 | 0.156 [0.145, 0.167] | 1.6e-09 | 1.6e-09 | primary |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_oracle|text | A|full|text | 48 | 0.258 [0.248, 0.269] | 1.6e-09 | nan | exploratory |

Primary family (Wilcoxon signed-rank (two-sided), Holm, alpha=0.05): InternVL3_5-8B-HF@741a7d03: C|sum_all|text - A|full|text (all)

## B / B' vs A and C vs B (absolute accuracy differences)

| model | split | slot | B arm | A arm | n | B-A [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|---|

## TOST: VLM + summary vs rule alone (margin +-0.02, alpha 0.05)

Equivalence here means only that the accuracy difference lies within the margin; it does not imply trial-by-trial rule following or identical internal computation (see agreement below).

- Secondary family InternVL3_5-8B-HF@741a7d03/episodes_v3#e00b8793: members ['C|sum_all|text vs window-count rule (domains pooled)', 'C|sum_cp|text vs window-count rule (domains pooled)', 'C|sum_window:5|text vs window-count rule (domains pooled)']; K* = 5 (source analysis/h200_ho/dev_selection.json); margin 0.02; alpha 0.05; Holm (missing members counted as p = 1); MISSING ['sum_cp', 'sum_window:5']

| model | split | slot | C arm | rule | n | acc VLM | acc rule | diff [90% CI] | p TOST | equiv. (uncorr.) | Holm p | equiv. (Holm) | role |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_all|text | window_counts | 48 | 0.639 | 0.632 | 0.007 [0.004, 0.011] | 2.1e-07 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_all|text | bocpd_posterior | 48 | 0.639 | 0.677 | -0.038 [-0.045, -0.031] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_oracle|text | window_counts | 48 | 0.745 | 0.744 | 0.001 [-0.001, 0.003] | 9.9e-21 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_oracle|text | bocpd_posterior | 48 | 0.745 | 0.677 | 0.068 [0.061, 0.075] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_all|text | window_counts | 48 | 0.631 | 0.632 | -0.001 [-0.005, 0.003] | 2.7e-10 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_all|text | bocpd_posterior | 48 | 0.631 | 0.677 | -0.046 [-0.054, -0.038] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_oracle|text | window_counts | 48 | 0.729 | 0.744 | -0.015 [-0.021, -0.008] | 0.082 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_oracle|text | bocpd_posterior | 48 | 0.729 | 0.677 | 0.053 [0.043, 0.063] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_all|text | window_counts | 48 | 0.635 | 0.632 | 0.003 [0.000, 0.006] | 2.2e-14 | yes | 6.6e-14 | yes | secondary |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_all|text | bocpd_posterior | 48 | 0.635 | 0.677 | -0.042 [-0.049, -0.035] | 1 | no | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_oracle|text | window_counts | 48 | 0.737 | 0.744 | -0.007 [-0.010, -0.004] | 1.5e-09 | yes | nan | - | exploratory |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_oracle|text | bocpd_posterior | 48 | 0.737 | 0.677 | 0.060 [0.053, 0.068] | 1 | no | nan | - | exploratory |

## Trial-level choice agreement: VLM + summary vs window-count rule (non-tied trials)

| model | split | slot | C arm | n ep | non-tied | agree (pooled) | agree ep-mean [95% CI] | disagree | VLM right | rule right | stable | volatile | VLM->A | VLM->B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_all|text | 48 | 3687 | 0.989 | 0.989 [0.984, 0.993] | 40 | 31 | 9 | 8 | 32 | 5 | 35 |
| InternVL3_5-8B-HF@741a7d03 | held_out | pet | C|sum_oracle|text | 48 | 3676 | 1.000 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_all|text | 48 | 3687 | 0.968 | 0.967 [0.948, 0.982] | 119 | 55 | 64 | 59 | 60 | 10 | 109 |
| InternVL3_5-8B-HF@741a7d03 | held_out | scenario | C|sum_oracle|text | 48 | 3676 | 0.971 | 0.971 [0.960, 0.980] | 108 | 29 | 79 | 27 | 81 | 54 | 54 |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_all|text | 48 | 7374 | 0.978 | 0.978 [0.968, 0.986] | 159 | 86 | 73 | 67 | 92 | 15 | 144 |
| InternVL3_5-8B-HF@741a7d03 | held_out | all | C|sum_oracle|text | 48 | 7352 | 0.985 | 0.985 [0.980, 0.990] | 108 | 29 | 79 | 27 | 81 | 54 | 54 |

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
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_oracle|text | color | 8 | 0.277 [0.252, 0.299] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_oracle|text | medical | 8 | 0.222 [0.188, 0.255] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_oracle|text | news | 8 | 0.214 [0.176, 0.252] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_oracle|text | sports | 8 | 0.209 [0.179, 0.236] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_oracle|text | stock | 8 | 0.245 [0.214, 0.288] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_oracle|text | weather | 8 | 0.226 [0.185, 0.263] |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_oracle|text | sign test | 6 scen. | 6 positive, p=0.031 |

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
| InternVL3_5-8B-HF@741a7d03 | held_out | A|cp_hard|text | pet | text | 7680 | 0.999 | 0.997 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | A|cp_hard|text | scenario | text | 7680 | 0.999 | 0.986 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | A|full|text | pet | text | 7680 | 0.999 | 0.997 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | A|full|text | scenario | text | 7680 | 0.999 | 0.988 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | A|window:10|text | pet | text | 7680 | 0.999 | 0.997 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | A|window:10|text | scenario | text | 7680 | 0.999 | 0.988 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_all|text | pet | text | 7680 | 0.996 | 0.875 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_all|text | scenario | text | 7680 | 0.998 | 0.952 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_oracle|text | pet | text | 7680 | 0.996 | 0.875 | 0.000 | 1.000 |
| InternVL3_5-8B-HF@741a7d03 | held_out | C|sum_oracle|text | scenario | text | 7680 | 0.997 | 0.919 | 0.000 | 1.000 |

Rule-window mismatches (recomputed vs logged window_start): 0; identical duplicate rows merged: 0; legacy y-only rows (approx logits): 0.
