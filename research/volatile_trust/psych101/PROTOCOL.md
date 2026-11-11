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
  - The model and the human therefore see identical evidence.
- **Standalone games.** Each game is shown alone as "Game 1."; the participant's other games are not shown.
  - **Limitation (experience mismatch):** at positions 2 and 3 the human had 1,600 or 3,200 trials of prior experience with the task, including a 1500-trial block at a different h. The model has none. The analysis therefore adjusts for position (see below), but cannot remove the mismatch.
- **Machine letters.**
  - Each participant keeps their own Psych-101 letters. All 26 uppercase letters are single tokens in every local Qwen2.5-VL, Qwen3-VL and InternVL3.5 tokenizer, so there is no remap (`LABEL_REMAP = {}`).
  - The instructions keep the participant's original intro order ("labeled X and Y"). The question lists the letters in that same order. The order is stored per episode as `intro_letters`; for 12 of 30 participants it is not alphabetical.
- **Hazard variants.**
  - `told`: the participant's Psych-101 intro verbatim. The only changes are the game header ("Game 1." for every standalone game; the original reads "Game 1/3/5.") and the appended question line.
  - `hidden`: removes "with a hazard rate (which you will be told)", the "For example, if the hazard rate is 0.1…" line and "The hazard rate is h." from the header.

## Formats (`render_xiong.py`; PIL only, deterministic, DejaVu fonts with sha256 recorded)
| format | history | image px | Qwen3-VL-8B tokens (t=1 / 50 / 100, mean) |
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

## Perception controls
- **perc_outcome:** 5 items per episode, 450 in total. The model sees one machine picture and picks which of 4 numbers is on the display (A to D). The distractors are one near value (within 10) and two far values.
- **perc_chart:** 6 items per episode, 540 in total.
  - *compare*: at trials k-1 and k different machines were pressed (|difference| ≥ 10); which machine gave more points?
  - *identify*: which machine was pressed at trial k?
  - The chart covers trials 1..K, with K ≥ k.
  - The chart shows only the chosen machine's points, so "which machine was higher at trial k" is made answerable by restricting it to adjacent switch trials.

## Analysis (`analyze_xiong.py`)
- We use the "myopic model-free agent" of Xiong et al. (2023): it updates the chosen value and forgets the unchosen value toward 50. We add perseveration (κ). The same model is fit to human choices and to the VLM's soft P, separately for each hazard rate.
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

## Run cost
| set | queries | 8B @ 0.3 s/row | 32B @ 0.75 s/row |
|---|---|---|---|
| one format × hazard variant | 9,000 | 0.75 h | 1.9 h |
| core: text told + hidden, image_outcome told, chart told, perception | 36,990 | 3.1 h | 7.7 h |
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
- **Completion marker.** `_COMPLETE_<tag>_shard<i>of<n>` is written only when the unique rows equal the expected keys for the selected episodes, with no duplicates. Debug `--trials` runs never write or remove it.
- **Resume.** A truncated last line is dropped on resume.
- **Analysis grouping.** `analyze_xiong.py` groups by (model, format, hazard_info, px, n_img) and stops with an error on duplicate keys.
