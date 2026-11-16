# Review: advice-taking / history-format paper (NeuroAI workshop, AAAI)

The Summary−Raw gap, the Correctness-only benefit and the masking replication hold up arithmetically. The main risks are four:

- overall accuracy does not show that models track *changes*;
- the "slower than Bayesian" claim compares unlike quantities;
- several claims hold only for the bias-cancelled readout or the quota design;
- the human links are stated more strongly than the data allow.

All fixes below need only existing logits or rewording.

## 1. Logic problems and overclaims (ranked)

**1.1 Accuracy does not demonstrate tracking of changing accuracy** (Abstract; Table `tab:main`; "Correctness-only recovers most…").
- Problem: "Always follow A" scores 64.8% (Table `tab:normsim`), above every open-model cell in Table `tab:main` (max 63.6%) and above the all-trial count rule (63.2%). A strategy that never updates therefore matches the best Correctness-only and Summary results.
- Problem: "Fraction recovered" (89–101%) treats Summary as a ceiling, although Correctness-only and Summary reach the same accuracy with different kernels (lag-10/lag-1 of 23% vs 52%).
- Fix: add always-A as a reference row in Table `tab:main`. Report accuracy or following share by stable/volatile phase; App. C promises "following probability by phase" but it appears nowhere.
- Fix: reword to "Correctness-only reached the level of the all-trial count rule through a recency-weighted strategy, but stayed 6–9 points below the last-five rule."

**1.2 "Adapted more slowly than a Bayesian volatility learner"** (Abstract; "Only recent counts…"; Discussion; Fig. `fig:curves`).
- Problem: the learner's curve is a share of argmax choices (97%), while the models' is a mean choice probability (.60/.56). These are not comparable.
- Problem: the Correctness-only kernel is nearly as recency-weighted as the learner's (23% vs 16%), which suggests weaker rather than slower use.
- Problem: curves average both reversal directions. For count-based inputs these behave oppositely (proprietary models given Summary: 50% early, 49% late; Table `tab:apibeh`), so "crossed .5 after six trials" is not an adaptation time. No uncertainty is shown.
- Fix: recompute model curves as share of argmax choices, add bootstrap bands, and split by reversal direction or state the averaging. Write "less completely" unless the discrete curves support "slower". Drop "versus six" in the Discussion.

**1.3 The Bayesian learner reads as a near-optimal benchmark, but it is not** (Abstract; contribution 3; "Proprietary…").
- Problem: the last-five rule beats it (69.2 vs 66.8; 67.9 vs 62.9 on the API subset), as does the BOCPD rule (67.7), and WSLS alone gets 62.3 on the subset. It was also added after the plan commit (App. J).
- Fix: "…tracked the advisors from Raw (63%, equal to a Bayesian reference learner; a last-five count rule scores 68%)." State in Methods that the learner was added post hoc with fixed settings and is not an ideal observer for this generator.

**1.4 Quota dependence is understated** ("The gap does not depend on the outcome quotas"; Abstract).
- Problem: the gap roughly halves without quotas (8.4–16.4 → 3.6–8.3). Correctness-only, Raw+Correctness and masking were never run on the robustness set, so the central claims rest on episodes where outcome labels are anti-predictive.
- Problem: truncated Raw at 39.1% (App. E) is far below chance and unexplained. A plausible cause is that quotas penalize label-frequency heuristics.
- Fix: retitle the paragraph "The gap is smaller but persists without outcome quotas". Add the 3.6–8.3 range to the Abstract and a Limitations sentence that the other formats and masking were tested only with quotas. Compute the repeat weight and κ for Raw on the robustness set from existing logits.

**1.5 Count-rule claims hold only for the two-copy logit average** ("Following the count rule…").
- Problem: the text reports 93.5–99.9% choice agreement, but per-copy κ with the all-trial count rule is .04–.68 for open models (Table `tab:apibeh`). Per-copy Summary accuracy is 55.9–63.1% (Table `tab:readout_sensitivity`) against the rule's 63.2%. Per-copy recovery by Correctness-only is 73–103% (Qwen3-VL-32B: 10.1/13.9).
- Fix: "After averaging logits over the two counterbalanced copies, which cancels name and answer-word biases, models chose…; single prompts agree less (App. J)." Add one sentence explaining the low per-copy κ.

**1.6 "The visible outcome label drives the repetition"** (Abstract; contribution 2; masking paragraph; Discussion).
- Problem: masking the label alone recovers only 40–60% of the Raw+Correctness→Correctness-only difference. Masking reports alone lowers the repeat weight in two models (.44→.26, .87→.65). The compact record lowers it with the label still present.
- Problem: masking was done on Raw+Correctness, not Raw, so "interference rather than missing information" is shown only for Raw+Correctness.
- Problem: the "four predictions" are never listed, and they postdate both the commit and inspection of held-out outputs.
- Fix: "is the main trigger of repetition in this prose layout". List the predictions verbatim in App. I. Change contribution 2 to "predictions written after development results and before the held-out masking runs (not part of the committed plan)".

**1.7 "Reasoning" is implied as the moderator** (Abstract; contribution 3; heading).
- Problem: GPT-6-Astra produced a median of 5 output tokens *including reasoning* and matched Claude's accuracy. App. J also says Claude "always reasons" but shows visible reasoning on 47% of calls.
- Fix: call them "two proprietary models (low reasoning effort)" throughout. Drop "without reasoning" from the Abstract, or add "one proprietary model also answered in about five tokens".

**1.8 Human and neuroscience links** (Abstract; contribution 4; Discussion; Conclusion).
- Problem: "behavioral signatures expected of people" and "behavior expected of people" contradict "it is not human performance" and "no human data on this variant".
- Problem: the Abstract and contribution state the bandit contrast without saying no difference survived correction.
- Problem: the stale-value regression "checked the same point", but human discounting was strongest at the lowest hazard (−.263 at h=0.1, about 0 at 0.4). That runs opposite to hazard-dependent forgetting.
- Problem: the BOLD sentence is followed by "we make no claim", so it reads as decoration.
- Fix (Conclusion): "…against normative references from the human literature."
- Fix (Abstract): "In an exploratory replay of a Psych-101 changing bandit, participants' fitted forgetting rose with hazard rate; model slopes were near zero, but paired differences were not significant after correction (power ≈ 0.5)."
- Fix: cut the BOLD sentence and the stale-value sentences from the main text.

**1.9 "Learning" language.**
- Problem: the heading "Proprietary reasoning models learn from Raw", "when it had to be learned", and "context-dependent learning" (used to mean in-context adaptation). Volatility-dependent learning is untestable here by your own App. F.
- Fix: use "track advisor accuracy from trial-by-trial feedback in context". Say in the Introduction that the task tests tracking across reversals, not volatility-dependent learning rates.

**1.10 Compact record "but not in the three larger models".**
- Problem: there is no matched original-layout baseline (App. I). Gemma's compact gap (+6.4) is under half its held-out gap (+14.7).
- Fix: "the gap remained +6.4 to +12.0 in the three larger models (no matched baseline)". Label the result "development episodes" in the Abstract.

**1.11 Image reports fail your own positive control** (image paragraph; App. G).
- Problem: four of five models fall below the 80% described-accuracy criterion with image reports, yet that result is interpreted. The chart is excluded for failing its criterion.
- Fix: state the criterion in the main text. Frame the result as a position/binding failure at the decision stage, not a deficit in advisor tracking.

**1.12 Smaller items.**
- "which explains Raw accuracy below 50%" (κ = .22–.32): change to "is consistent with".
- The Stroop analogy invites a simpler account (copying the most recent answer token). Hedge to "loosely analogous" or cut.
- "Summary… is not a description": the described-counts control uses the same sentence. Say "description-style counts computed from the realized history".

## 2. Numerical inconsistencies

| Place A | Place B | Issue and fix |
|---|---|---|
| Masking text: label masking "+2.7–6.4" | Table `tab:masking_accuracy`: 58.6−56.0 = 2.6 | Probably rounding; add "from unrounded values" or change to 2.6 |
| Image text: "6.4–10.2" | Table `tab:mm`: 60.2−53.7 = 6.5 | Same |
| Table `tab:main`: +13.4, +16.4 | 62.1−48.6 = 13.5; 63.1−46.8 = 16.3 | Same; add a caption note |
| Table `tab:tests`: Gemma −1.4; Qwen-8B last-5 −2.4 | 61.7−63.2 = −1.5; 66.7−69.2 = −2.5 | Same |
| App. C: "near chance only within 3 pp" | Qwen3-VL-32B Raw 46.8; three CIs exclude 50 (Table `tab:decomp`) | Write "at or below chance" |
| Main: "image naming 98.8–100%" | App. C: "photo naming 98.6–100%" | State pooled vs photo once |
| App. B: development stories are color, weather, stock, sports | App. G: medical icons selected on "development items" | Explain how the medical items were built |
| Table `tab:cover` caption: App. G "identifies results obtained with the original set" | App. G: original outputs "not reported" | Reconcile |
| App. C: "held-out episodes are run once" | App. G: Table `tab:mm` uses "complete reruns"; masking and API runs came later | Say "each condition was run once; conditions added later are marked exploratory" |
| Results: learner crosses .5 "three trials after"; variants "two or three" | Discussion: "within three trials", "three… versus six" | Define the offset (0 = first post-reversal trial) and use one convention |
| Kernel: 15 lags (open models) | App. J: lags 1–10 (API models) | State the difference or refit |
| Main: κ = .22–.32 "in four models" | Table `tab:apibeh`: .08–.31 (five, subset) | Name the fifth model and its κ |
| Source comment: "final numbers from held-out only" | Abstract's compact-record claim is development-only | Label it |

## 3. Missing information, and what to cut

**Missing (all from existing data or text):**
- Raw repeat weights on held-out with CIs, in the main text. The core repetition evidence currently lives only in appendix Figs. `fig:kernel_models` and `fig:strategies`.
- A definition of "easy history" (how many recent trials, and what precedes them in the 80-trial version), plus items per control cell.
- The kernel regression specification: how a probability is the dependent variable, and whether fits are pooled.
- The direction of the Gemma readout bias. It is conservative (51% vs 59% on low-mass trials), and the "both families" criterion rests on this one model. Say so in Methods.
- The synthetic all-trial-rule kernel ratio at lag 10. A flat kernel would give 100%, so "52%, consistent with counting all trials" needs this support.
- Bandit fits: under replay, whose choices define "chosen/unchosen" and the perseveration term when fitting sampled model choices.
- The InternVL prompt differences (Chinese system message, suffix), and which models "reasoning disabled where the template permits" excludes.
- Open-model subset accuracies next to the API rows in Table `tab:main`, so readers do not compare 62.8 with 48.6.

**Cut or demote (risky, not load-bearing):**
- Human bandit: move Fig. `fig:human` to App. H and keep three hedged sentences. It is null, about 50% powered, upward-biased, and the effect weakens in the 1,500-trial games. This frees about 0.4 page for the items above.
- The correctness chart: remove from Methods and Results, since it is not interpreted.
- The PhotoGuider sentence in Related Work.
- App. F "Effective learning rate": say which agent the statistic was applied to and what the range "0.84–0.90" spans, or cut it.
- "Outputs with the original icons… not reported" reads as selective reporting. Change to "available in the released outputs".
- Strip the LaTeX comments (terminology block, "Do not restore…") if the source is uploaded.

## 4. Clarity and structure

- **Results order.** Fig. `fig:gap` appears first, but its paragraph is seventh. Suggested order:
  1. controls;
  2. gap and Correctness-only;
  3. repetition and masking, merged;
  4. boundaries (robustness set, compact record, proprietary models);
  5. integration vs references;
  6. images;
  7. bandit.
- **"Correctness-only recovers most…"** carries seven claims. Move the compact-record sentences to the boundaries paragraph.
- **Table `tab:masking_accuracy`** repeats "Reports… Reports… Reports… Reports" in its header. Use a 2×2 header or "none / reports / label / both".
- **"The effect has boundaries"** (Abstract): "the effect" is undefined. Write "The Summary−Raw gap".
- **"Episode pair"** is defined three times, twice within the Measures paragraph. Keep one.
- **One concept, several names:**
  - "label-aligned logits" / "counterbalanced logits" / "candidate logits";
  - "masked" / "replaced by a dash";
  - "repeating the previous outcome" / "outcome repetition" / "repetition";
  - "probability of change" / "change probability" / "hazard rate";
  - "Bayesian volatility learner" / "Bayesian learner" / "the learner";
  - "held-out set" / "full set" / "main set";
  - "held-out subset" / "API subset".
- **One name, several concepts:**
  - "accuracy" covers prediction accuracy, advisor accuracy and control-item accuracy;
  - "gap" covers Summary−Raw, description–history and the Correctness-only difference;
  - "history-format effect" covers both Summary−Raw and Correctness-only−Raw.
- **"Implications for NeuroAI"** opens with agent engineering. Split it into "For agents" and "For NeuroAI" or retitle it.

## 5. Likely skeptical-reviewer objections and text-only preemption

1. **"Why VLMs? The main results are text-only."** Say in the Introduction: "Primary analyses use text reports; image reports are an exploratory extension that exposes a separate position/binding failure."
2. **"It's a prompt-format artifact."** Lead the boundaries paragraph with the compact-record result. Note that Raw stayed below the count rule in every wording, and that Correctness-only and Summary also fell (Summary agreement with the count rule dropped to 72% for InternVL3.5-8B).
3. **"It's a quota artifact."** Use the fixes in 1.4 and report the robustness-set gap beside the headline numbers.
4. **"Accuracy barely separates strategies."** State the scale once in the main text: 50 chance, 63.2 all-trial count, 64.8 always-A, 66.8 learner, 69.2 last-five, 80 known accuracy. Then rest the claims on kernels and reversal windows.
5. **"This is not volatility learning, and there are no human data."** Say this in the Introduction rather than only in Limitations, and remove "expected of people".
6. **"Open models were never allowed to reason; four of five share one language model."** This is already in Limitations. Add "we do not claim the deficit persists with reasoning enabled" and avoid "VLMs" as a general subject in the Conclusion.
7. **"Many analyses are post hoc."** Add one sentence in Methods mapping the status of each analysis:
   - confirmatory: the primary test;
   - prespecified secondary: TOST and the recovery ratio;
   - later-specified replication: masking;
   - exploratory: everything else, including the Bayesian learner.