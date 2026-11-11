# xiong2023 "Changing bandit": frozen-VLM replay protocol

## Data
Provenance (URLs, revisions, licenses and sha256) is in `.cache/datasets/psych101_xiong2023/provenance.json`.
- **Psych-101 text slice:** 30 participants, Apache-2.0.
- **Raw parquet:** `marcelbinz/xiong2023neural`. It has **no declared license**, so we use it for analysis only and do not redistribute it.
- The two sources match trial for trial in all 180 games.

## Protocol
- **Episodes.** We use the three 100-trial games per participant (h = 0.1, 0.2, 0.4). These are Psych-101 games 1, 3 and 5, i.e. the paper's 300-trial practice. That gives 90 episodes and 9,000 trials.
  - The 1500-trial games are in `data/episodes_long.jsonl`. They are used only for human reference fits (`--human-long`).
- **Teacher-forced replay.** At trial t the model sees the original instructions, a game header and the human's trials 1..t-1 with the points the human received. We then ask "Trial t of this game: which slot machine do you press? Answer with exactly one letter: X or Y."
  - The readout is P(letter), normalized over the two letters, from raw first-token logits.
  - The model and the human therefore see identical evidence.
- **Standalone games.** Each game is shown alone as "Game 1."; the participant's other games are not shown.
- **Machine letters.**
  - Each participant keeps their own Psych-101 letters. All 26 uppercase letters are single tokens in every local Qwen2.5-VL, Qwen3-VL and InternVL3.5 tokenizer, so there is no remap (`LABEL_REMAP = {}`).
  - In the instructions and the question, the letters are listed alphabetically. Alphabetical order is independent of the slot index: the first letter is slot 1 in 108 of 180 games.
- **Hazard variants.**
  - `told`: the verbatim Psych-101 wording.
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
  - per-participant fits with a Wilcoxon test of h = 0.4 vs 0.1 and a within-participant Spearman trend;
  - the paper's model-agnostic signature, P(shift | last reward < 50) by h.
- The self-test (`--selftest`) recovers α and φ from closed-loop simulations on the real value sequences.

## Run cost
| set | queries | 8B @ 0.3 s/row | 32B @ 0.75 s/row |
|---|---|---|---|
| one format × hazard variant | 9,000 | 0.75 h | 1.9 h |
| core: text told + hidden, image_outcome told, chart told, perception | 36,990 | 3.1 h | 7.7 h |
| full: 3 formats × 2 variants + perception | 54,990 | 4.6 h | 11.5 h |

The prompts are longer than in the volatile-trust task (mean of about 800 to 1,100 tokens for text and image_outcome), so expect s/row above these rates. The chart is cheapest.

## Known mismatches with the paper
- The paper reports 29 participants doing 3 × 1500-trial blocks after a 300-trial practice. The raw data have 33 participants and Psych-101 has 30; raw ids 3, 5 and 28 are absent from Psych-101. Raw participant 31 has 500-trial long blocks.
- We replay the 100-trial (practice) games because of context length.
- The paper fits LR, MF, model-based and RNN models but no per-h learning rates. Its human finding is "high-stay, low-shift": more shifting after low rewards when h is higher. It reports no learning-rate increase.
- The instruction text says the same machine "will not always give you the same points". In the data, a machine's points stay the same until a change point.
