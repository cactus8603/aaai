# xiong2023 "Changing bandit": frozen-VLM replay protocol

## Data
Provenance (URLs, revisions, licenses and sha256) is in `.cache/datasets/psych101_xiong2023/provenance.json`.
- **Psych-101 text slice:** 30 participants, Apache-2.0.
- **Raw parquet:** `marcelbinz/xiong2023neural`. It has **no declared license**, so we use it for analysis only and do not redistribute it.
- The two sources match trial for trial in all 180 games.

## Protocol
- **Episodes.** We use the three 100-trial games per participant (h = 0.1, 0.2, 0.4). These are Psych-101 games 1, 3 and 5. That gives 90 episodes and 9,000 trials.
  - Each 100-trial game directly precedes a 1500-trial game with the same h. By games 3 and 5 the human has already played 1,600 and 3,200 trials. The h order is counterbalanced over 6 orders, with 4–6 participants each.
  - Each episode stores `position` (1/2/3 = game 1/3/5). The (h, position) cells hold 9–11 episodes each.
  - The 1500-trial games are in `data/episodes_long.jsonl`. They are used only for human reference fits (`--human-long`).
- **Teacher-forced replay.** At trial t the model sees the original instructions, a game header and the human's trials 1..t-1 with the points the human received. We then ask "Trial t of this game: which slot machine do you press? Answer with exactly one letter: X or Y."
  - The readout is P(letter), normalized over the two letters, from raw first-token logits.
  - The model and the human see the same **within-game choice and feedback history**. That is the only sense in which their evidence is identical. The humans were also told h, and at positions 2 and 3 they had prior experience with the task (below). The models had neither in the primary `hidden` condition. In `told` they get the stated h but still no prior experience.
  - The readout uses the raw letter logits. Each row saves `logits` (both letters) and `logit_diff` (slot 1 − slot 2). The analysis uses P = sigmoid(logit_diff) when present, so saturated softmax outputs lose no information.
- **Standalone games.** Each game is shown alone as "Game 1."; the participant's other games are not shown.
  - **Limitation (experience mismatch):** at positions 2 and 3 the human had 1,600 or 3,200 trials of prior experience with the task, including a 1500-trial block at a different h. The model has none. The analysis therefore adjusts for position (see below), but cannot remove the mismatch.
- **Machine letters.**
  - Each participant keeps their own Psych-101 letters. All 26 uppercase letters are single tokens in every local Qwen2.5-VL, Qwen3-VL and InternVL3.5 tokenizer, so there is no remap (`LABEL_REMAP = {}`).
  - The instructions keep the participant's original intro order ("labeled X and Y"). The question lists the letters in that same order. The order is stored per episode as `intro_letters`; for 12 of 30 participants it is not alphabetical.
- **Hazard variants. `hidden` is the PRIMARY condition** (the `run_xiong.py` default). Only in `hidden` must the volatility be inferred from experience; under `told` a model can use the stated number. Humans were told h, so the `told` − `hidden` contrast (secondary) measures how much the description helps the model: a description–experience contrast.
  - **What each human–model comparison matches.** The analysis reports three comparisons together (`human_model_comparisons`):
    - **Model hidden vs human told (primary).** This is a *descriptive cross-condition difference*: both the hazard information and the prior experience differ. On its own it does not show a human–model gap in learning ability.
    - **Model told vs human told.** The hazard information matches; the prior experience still differs.
    - **Model told − model hidden.** Same model and same histories, so this isolates the effect of stating h.
  - `told` (secondary): the participant's Psych-101 intro verbatim. The only changes are the game header ("Game 1." for every standalone game; the original reads "Game 1/3/5.") and the appended question line.
  - `hidden` (primary): removes "with a hazard rate (which you will be told)", the "For example, if the hazard rate is 0.1…" line and "The hazard rate is h." from the header.

## Formats (`render_xiong.py`; PIL only, deterministic, DejaVu fonts with sha256 recorded)
| format | history | image px (non-InternVL; InternVL always 448) | Qwen3-VL-8B tokens (t=1 / 50 / 100, mean) |
|---|---|---|---|
| text (told) | Psych-101 lines | none | 214 / 799 / 1398, mean 805 |
| text (hidden) | same | none | 169 / 754 / 1353, mean 760 |
| image_outcome | text, but the last N=5 outcomes are machine pictures with a digital display | 224 | 238 / 1088 / 1687, mean 1087 |
| chart | one scatter chart of trials 1..t-1 (x = trial on a fixed 1..100 axis, y = points; blue circle and orange triangle per machine; legend) | 448 | 467 / 469 / 470, mean 469 |

- **Why N = 5 rather than only the previous outcome.**
  - Under h = 0.4, the decision-relevant values, including the last-seen value of the other machine, usually lie 2 to 5 trials back. A single picture would let the model ignore the image channel.
  - Five pictures cost about 290 extra tokens.
  - `--n-img 1` gives the single-picture variant.
- **Chart resolution.** At 224 px the chart is only 7x7 vision tokens for Qwen3-VL, so the chart format defaults to 448 px.
- **Resolution by model type.** `run_xiong.py` reads `model_type` from the model's `config.json`. The InternVL backend accepts only one 448 px tile, so InternVL uses 448 px for every format, including text and perception. Other models keep the per-format defaults above. The effective px is stored in every row and in the tag.

## Perception controls
- **perc_outcome:** 5 items per episode, 450 in total. The model sees one machine picture and picks which of 4 numbers is on the display (A to D). The distractors are one near value (within 10) and two far values.
- **perc_chart:** 6 items per episode, 540 in total.
  - *compare*: at trials k-1 and k different machines were pressed (|difference| ≥ 10); which machine gave more points?
  - *identify*: which machine was pressed at trial k?
  - The chart covers trials 1..K, with K ≥ k.
  - The chart shows only the chosen machine's points, so "which machine was higher at trial k" is made answerable by restricting it to adjacent switch trials.

## Analysis (`analyze_xiong.py`)
- We use the "myopic model-free agent" of Xiong et al. (2023): it updates the chosen value and forgets the unchosen value toward 50. We add perseveration (κ). The same model is fit separately for each hazard rate.
- **Comparable fits.** Humans give binary choices; models give soft P.
  - **Primary model fit:** for each episode we draw 20 binary choice sequences from the model's P; the history stays the human's. We fit `xiong_mf` to each sequence and take the per-episode median (`sampled_fit`).
  - **Secondary:** the soft-target fit (cross-entropy on P), reported under `pooled` / `per_participant`.
  - **β is reported alongside φ** (median per h, plus a β trend test), so β differences cannot pass as φ differences.
  - **Recovery at the model's own β** (`phi_recovery_at_model_beta`): closed-loop simulations at the model's median β and κ per h, with (a) φ = h and (b) constant φ = 0.2. Case (b) must show no slope.
- **ONE primary model-vs-human test within this protocol** (`model_vs_human_phi_slope`, primary condition = `hidden`). For the paper it is a supporting, descriptive comparison; see the H02 decision under "Calibration of the paired slope test":
  - compute each participant's OLS slope of φ on h (model sampled-fit φ vs human binary-fit φ, on the same 30 histories);
  - take the paired difference model − human;
  - test it with Wilcoxon signed-rank and give a participant-bootstrap 95% CI of the median difference.
  - In the `hidden` condition this is a **descriptive cross-condition difference** (see Hazard variants). It is reported next to model told vs human told and the model told − hidden contrast.
  - Everything else is secondary.
  - **Monte Carlo error of the 20 draws** (`mc_error_of_draws`). We resample the 20 draws of each episode with replacement and recompute the medians, the slopes and the paired statistic. We report the SD of the median difference, the 5–95% range of the Wilcoxon p, and split-half (draws 1–10 vs 11–20) estimates.
  - **Calibration** (`--calibrate N`, written to `calibration.json`; see "Calibration of the paired slope test" below). Read the primary p-value only together with this calibration.
- **Human learning rates are our re-analysis.** The per-h φ fits and trend tests for humans are our own analysis of the data. Xiong et al. (2023) report no per-h learning rates (see Known mismatches).
- **Fit diagnostics.** Every fit stores `converged` (scipy success) and `at_bound` (parameters within 1e-3 of a box bound on the unconstrained scale).
  - The report gives the failed-fit and boundary-fit rates per source and per h (`fit_diagnostics`), and for the sampled fits.
  - α → 1 (`alpha_hi`) is the normative value for noise-free rewards. It is reported but not counted as a problem (`frac_boundary_excl_alpha_hi`).
  - The fits use an exact closed-form NLL with an analytic gradient. It equals the sequential recursion to within 1e-13 (checked in the selftest) and is about 50 times faster than the finite-difference version.
- **Seeds.** Every random draw uses its own fixed seed, `rng_for(condition, purpose, episode)`. Adding a condition, or changing which conditions are analysed, does not change any other condition's draws or bootstraps. The first k draws of an episode are the same for any `--n-samples` ≥ k.
- **Hazard sets.** A model source needs complete episodes at all three h. Otherwise it is listed under `skipped_sources` and not analysed. `trend_tests` omits Friedman and Page when fewer than three hazards are given.
- The rewards are noise-free: reward = the chosen machine's true value on all 155,400 rows. So the normative Bayesian learner has α = 1 and forgetting rate φ = h. **φ is the hazard-sensitive learning rate.** We also report α and a plain delta rule (φ = 0).
- Tests:
  - pooled fits with a participant bootstrap;
  - per-participant fits, with one statistic per participant (no pseudo-replication):
    - the per-participant OLS slope of the parameter on h, tested with Wilcoxon against 0;
    - Friedman, and Page's L for the ordered increase;
  - **position adjustment**: a within-participant fixed-effects OLS, param ~ h + position, with a participant-bootstrap CI for the h coefficient, plus medians per (h, position) cell;
  - the paper's model-agnostic signature, P(shift | last reward < 50) by h.
- `--human-long` also fits humans on their 1500-trial games and reports them next to the 100-trial fits (`human_short_vs_long`).
- Self-test (`--selftest`):
  - recovers α and φ from closed-loop simulations on the real value sequences;
  - **null check**: every simulated participant has φ = 0.2 at every h (β = 0.4, about the human level). Per-participant fits then show no spurious h trend: slope-Wilcoxon and Page p > 0.05 in 5 of 5 seeds.
  - **plumbing**: the closed-form NLL equals the sequential recursion and the analytic gradient equals finite differences; two-hazard input skips Friedman and Page; per-condition seeds are independent of other conditions; the `logit_diff` readout is used without saturation.
- `run_xiong.py --selftest` uses a fake backend on CPU. It checks: px chosen by model type; full-design key counts (9,000 / 450 / 540); logits saved; a truncated line is cut and re-scored; metadata is appended; a changed data file, a changed model revision or foreign rows are refused; and the marker is written only for the full design.

## Run cost
| set | queries | 8B @ 0.3 s/row | 32B @ 0.75 s/row |
|---|---|---|---|
| one format × hazard variant | 9,000 | 0.75 h | 1.9 h |
| core: text hidden + told, image_outcome hidden, chart hidden, perception | 36,990 | 3.1 h | 7.7 h |
| full: 3 formats × 2 variants + perception | 54,990 | 4.6 h | 11.5 h |

The prompts are longer than in the volatile-trust task (mean of about 800 to 1,100 tokens for text and image_outcome), so expect s/row above these rates. The chart is cheapest.

## Known mismatches with the paper
- The paper reports 29 participants doing 3 × 1500-trial blocks after a 300-trial practice. The raw data have 33 participants and Psych-101 has 30; raw ids 3, 5 and 28 are absent from Psych-101. Raw participant 31 has 500-trial long blocks.
- The paper does not describe the 100-trial games. They are not "the paper's practice": that practice is described as 300 trials *before* the blocks, whereas each 100-trial game here sits directly before a 1500-trial block with the same h. We replay the 100-trial games because of context length.
- In the 1500-trial blocks the pooled human φ is flat (about 0.01–0.02 at every h), while per-participant medians and slopes still rise with h. Report both.
- The paper fits LR, MF, model-based and RNN models but no per-h learning rates. Its human finding is "high-stay, low-shift": more shifting after low rewards when h is higher. It reports no learning-rate increase.
- The instruction text says the same machine "will not always give you the same points". In the data, a machine's points stay the same until a change point.

## Run outputs
- **File names.** `results_<tag>_shard<i>of<n>.jsonl`. For the choice formats `<tag>` is `<format>_<hazard_info>_px<px>_n<n_img>`; for the perception formats it is `<format>_px<px>`. Each row also stores `image_px`, `n_img` and `tag`.
- **Run fingerprint.** `fingerprint_<stem>.json` holds the runner version, model id and revision, model type, the sha256 of `episodes.jsonl`, the sha256 of `render_xiong.py` (prompt and renderer version), the font sha256s, the format, the px, and for choice formats `n_img` and `hazard_info`. Every row carries the fingerprint hash (`run_fingerprint`).
- **Resume.** The truncated last line is cut off first. Then the run refuses to append, and exits, if the stored fingerprint differs, if rows exist without a fingerprint file (legacy), or if any row carries another hash. To change a setting, use a new `--output`.
- **Metadata.** `run_metadata_<stem>.jsonl` gets one appended line per launch (backend metadata, fingerprint, args, rows already present). It is never overwritten.
- **Completion marker.** `_COMPLETE_<stem>` is written only when all of these hold:
  - the run covers the full design: 30 participants × 3 h × 100 trials = 9,000 keys over all shards for each choice format, or 450 / 540 perception items;
  - the shard's unique rows equal its expected key set, with no duplicates, no missing keys and no foreign-fingerprint rows.
  Subset runs (`--max-participants`, `--hazards`, `--trials`) never write it.
- **Analysis grouping.** `analyze_xiong.py` groups by (model, format, hazard_info, px, n_img). It stops with an error on duplicate keys or on a group that mixes run fingerprints.

## Human result (our re-analysis): which test gives which p
Source: `analysis_human.json`; its `p_value_legend` maps each field to its test.

| test | 100-trial games | 1500-trial games |
|---|---|---|
| **PRIMARY:** per-participant φ slope on h, Wilcoxon vs 0 (`slope_wilcoxon_p`) | p = 0.0076 | p = 0.029 |
| Page's L, ordered trend (`page_p`) | p = 0.017 | p = 0.066 |
| Within-participant OLS φ ~ h + position, b_h with bootstrap 95% CI | 0.64 [0.23, 1.05] | 0.33 [−0.08, 0.68] |
| Pooled φ at h = 0.1 / 0.2 / 0.4 | 0.054 / 0.080 / 0.282 | 0.025 / 0.014 / 0.012 |
| Per-participant median φ at h = 0.1 / 0.2 / 0.4 | 0.079 / 0.110 / 0.255 | 0.076 / 0.114 / 0.131 |

- Fit diagnostics: no failed fits. Boundary fits other than α → 1 make up 11.7% of the 100-trial fits and 9.8% of the 1500-trial fits, mostly φ at its lower bound (φ ≈ 0).
- Rerun with the analytic-gradient fitter and per-condition seeds: the point estimates and p-values are unchanged; only the bootstrap CIs moved, by ≤ 0.02.

The p = 0.015 quoted in an earlier note was the Wilcoxon of φ(0.4) − φ(0.1). It is superseded and no longer reported.

## Calibration of the paired slope test (H02; `calibration.json`)
**Setup.** 200 seeds × 3 arms; this took 14 min on 40 CPU cores.
- **Truth.** Each real human's own per-h xiong_mf fit, clipped to φ ∈ [0.01, 0.9], β ∈ [0.05, 4] and |κ|, |b| ≤ 3.
- **Simulated humans.** They play closed loop on the real value sequences.
- **Simulated model.** The same strategy, with φ_h shifted by δ·(h − mean h). It is replayed teacher-forced on the simulated human's history.
- **Pipeline.** We run the full pipeline: the pooled warm start, the human per-participant fits, the 20-draw sampled fits, then the Wilcoxon test with a bootstrap CI.

| arm (true median slope diff) | reject p<.05 | bias of median diff | 95% CI coverage |
|---|---|---|---|
| δ = 0 (0) | **0.060** (MC SE 0.017) | +0.092 | 0.935 |
| δ = −0.5 (−0.50) | 0.485 | +0.158 | 0.870 |
| δ = +0.5 (+0.39 after clipping) | 0.590 | +0.075 | 0.960 |
| single-draw variant, δ = 0 / −0.5 / +0.5 | 0.035 / 0.275 / 0.265 | +0.018 / +0.134 / −0.039 | 0.945 / 0.920 / 0.935 |

**Reading.**
- **False positives.** Under no true difference the test is close to nominal: 6.0% ± 1.7%.
- **Bias.** The 20-draw median estimator is biased upward by about +0.09 per unit h relative to the human single-sequence fit. The model-side slope bias is +0.075 and the human-side bias is +0.042. The bias also shrinks a true negative difference, so at δ = −0.5 coverage falls to 0.87.
- **Power is modest.** For a slope difference about as large as the human slope itself (≈0.4–0.5), power is 0.49–0.59. A non-significant result therefore cannot be read as "no difference".
- **Boundary fits.** 22–28% of the simulated fits hit a φ or β bound; none failed.

**Decision (H02; binding for the paper and `EXPERIMENTS.md` E6/C4).**
- The model-vs-human comparison is **supporting and descriptive**, not a primary or confirmatory result of the paper.
- We report the paired slope-difference estimate (median model − human) with its participant-bootstrap 95% CI and the Wilcoxon p, always next to this calibration: false-positive rate 6.0% ± 1.7% at δ = 0, CI coverage 0.935, an upward bias of about +0.09 in the 20-draw estimator, and power 0.49–0.59 for |δ| = 0.5.
- We make **no strong claim** of a human–model difference or of human–model similarity. A difference of about 0.09 could be produced by the estimator bias alone.
- A null result is **uninterpretable** at this power: it is not evidence that models and humans adjust their learning rates alike.
- The sampled-fit selftest also gave slope p = 0.033 for β = 1.5 with constant φ (a null case; `analyze_xiong.py --selftest`, `sampled_fit_recovery/beta1.5_phi_const_0.2`). This is consistent with the upward bias above and is a further reason for the descriptive framing.
- If the fixes below are implemented before the freeze, they are reported as sensitivity analyses only; they do not raise the level of the claim.

**Proposed fix (not yet applied to the primary statistic).**
- Report a calibration-based bias-corrected difference: the observed difference minus the null-arm bias, estimated at the model's own fitted parameters.
- Or replace the Wilcoxon p with a parametric-bootstrap null: simulate both pipelines under δ = 0 at the fitted parameters and compare the observed statistic with that null.
- The single-draw variant is nearly unbiased but loses about half the power.
