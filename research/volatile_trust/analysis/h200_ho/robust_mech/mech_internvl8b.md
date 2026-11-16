# Mechanism analysis (analyze_mechanism.py)

All results are behavioural descriptions: "behaviour consistent with" a strategy and "effective weight on past outcomes", not evidence about internal computation. C-arm kernels reflect the external summary and the model jointly.

## Behaviour consistent with simple strategies (domains pooled): agreement / chance / kappa [95% CI]

| model/data/split/readout/arm | copy_prev_truth | follow_prev_correct | count_all | last3 | last5 | last10 | bocpd | oracle | label_prior | most consistent (episodes) |
|---|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03/episodes_robust_v3#9e0f4ff9/held_out/choice/A|full|text | 0.66/0.50/**0.31** [0.27, 0.36] | 0.53/0.50/**0.06** [0.02, 0.10] | 0.55/0.53/**0.06** [0.02, 0.09] | 0.54/0.51/**0.06** [0.03, 0.09] | 0.54/0.51/**0.06** [0.03, 0.09] | 0.54/0.51/**0.06** [0.03, 0.09] | 0.53/0.51/**0.05** [0.02, 0.08] | 0.54/0.51/**0.07** [0.03, 0.11] | 0.53/0.50/**0.06** [-0.13, 0.25] | copy_prev_truth:16, label_prior:8 |

## Same trial mask: kappa_common (model vs strategy) and strategy-by-strategy kappa

**InternVL3_5-8B-HF@741a7d03/episodes_robust_v3#9e0f4ff9/held_out/choice/A|full|text** (common mask: 3350 of 3792 trials, 0.88)

| | model | copy_prev_truth | follow_prev_correct | count_all | last3 | last5 | last10 | bocpd | oracle | label_prior |
|---|---|---|---|---|---|---|---|---|---|---|
| copy_prev_truth | 0.32 | 1.00 | 0.01 | -0.04 | -0.02 | -0.03 | -0.03 | -0.03 | -0.03 | 0.01 |
| follow_prev_correct | 0.05 | 0.01 | 1.00 | 0.23 | 0.62 | 0.54 | 0.42 | 0.62 | 0.52 | -0.01 |
| count_all | 0.06 | -0.04 | 0.23 | 1.00 | 0.35 | 0.41 | 0.48 | 0.43 | 0.31 | -0.01 |
| last3 | 0.07 | -0.02 | 0.62 | 0.35 | 1.00 | 0.78 | 0.62 | 0.83 | 0.62 | -0.02 |
| last5 | 0.06 | -0.03 | 0.54 | 0.41 | 0.78 | 1.00 | 0.77 | 0.85 | 0.63 | -0.03 |
| last10 | 0.06 | -0.03 | 0.42 | 0.48 | 0.62 | 0.77 | 1.00 | 0.78 | 0.50 | -0.01 |
| bocpd | 0.06 | -0.03 | 0.62 | 0.43 | 0.83 | 0.85 | 0.78 | 1.00 | 0.62 | -0.01 |
| oracle | 0.07 | -0.03 | 0.52 | 0.31 | 0.62 | 0.63 | 0.50 | 0.62 | 1.00 | 0.01 |
| label_prior | 0.09 | 0.01 | -0.01 | -0.01 | -0.02 | -0.03 | -0.01 | -0.01 | 0.01 | 1.00 |

## Effective weight on past outcomes (soft logistic kernel, domains pooled, episode-bootstrap 95% CI)

| model/data/split/readout/arm | w(s_1) | w(s_2) | w(s_3) | w(s_5) | w(s_10) | sum w(s_1..K) | exp fit lam (R2) | copy_prev_truth | label_prior | a_listed_first | intercept |
|---|---|---|---|---|---|---|---|---|---|---|---|
| InternVL3_5-8B-HF@741a7d03/episodes_robust_v3#9e0f4ff9/held_out/choice/A|full|text | 0.06 [0.01, 0.11] | 0.06 [0.02, 0.09] | 0.04 [-0.00, 0.08] | 0.01 [-0.04, 0.05] | 0.02 [-0.02, 0.06] | 0.33 | 0.83 (0.14) | 0.41 [0.35, 0.47] | 0.08 [-0.16, 0.32] | 0.11 [0.03, 0.19] | 0.07 [0.02, 0.14] |
