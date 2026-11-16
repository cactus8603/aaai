## 1. Verdict

**Strengths.** The format decomposition is clean. The design is careful: counterbalanced copies, a held-out set, a primary test fixed in advance, rule-only baselines, and a masking experiment with predictions written beforehand. The appendices are unusually honest.

**Weaknesses.** The headline ("the bottleneck is judging raw outcomes, not integrating them") is stated for models in general, but three things contradict it:
- the new API result;
- the draft's own numbers, which show a second, integration loss of 3–7 points;
- the draft's own layout check, where the gap nearly vanishes with a compact record.

The human comparison is indirect and partly mismatched. There are no human data on this task, the reference is an ideal observer, and the "shortcut" was never tested in people. The eight pages carry too many threads (binding, Psych-101, controller). The Psych-101 test is underpowered and the mapping to cortex overreaches.

**Likely score.** Borderline or weak accept as written; a clear accept if reframed and scoped.

## 2. Recommended framing

**Central claim.** The format decomposition is a diagnostic: which step limits learning whom to trust depends on the system, and the decomposition identifies it.
- **Open-weight models answering directly.** For five open-weight VLMs answering with one word and no reasoning, the largest loss is evaluating outcomes across many trials, where they copy the visible outcome label. A smaller loss remains in integrating at the right time scale.
- **API reasoning models.** Two proprietary reasoning models show no evaluation loss. They match the Bayesian volatility learner from the raw record (.63 vs .629), gain nothing from correctness labels, and are capped by an all-trial summary at its rule (Claude .585, exactly the rule's value).

**Title.** Something like "Learning Whom to Trust from Feedback: Which Step Fails Depends on the Model". Drop "Vision-Language Models" from the title, since all confirmatory results use text reports.

**What must no longer be claimed:**
- That models or VLMs in general cannot learn from raw experience.
- That the Bayesian learner beats "every model", or that models match it "only" with an external rule.
- That integration must sit outside the model.
- That reasoning explains the difference. It is confounded with scale, training and readout; only the queued Qwen3-VL-8B-Thinking run speaks to it, so write both outcomes now.
- That the API models "solve" the task. They are about 5 points below the last-5 count on the same subset (.679).

## 3. Prioritized revisions

1. **Abstract, "The bottleneck is judging raw outcomes, not integrating them."** Replace with: "In open-weight models answering directly, most of the loss lies in judging raw outcomes; two proprietary reasoning models, queried on a subset, showed no such loss and matched the Bayesian learner from the raw record (.63 vs .63)." Mirror this in the introduction's second contribution bullet and the first Discussion paragraph.

2. **"Integration is thus not where most accuracy is lost" and the abstract.** The draft's own numbers show two losses. From Table 1 (my calculation), raw to correctness-only is 8–16 points, and correctness-only to the Bayesian learner is a further 3–7 points. Evaluation is therefore 64–83% of the shortfall. Volatile-phase following is near .50 in all formats, and long correctness-only controls reach only .67–.90. Replace with: "Outcome evaluation accounts for about two-thirds to four-fifths of the shortfall from the Bayesian learner; integration at the right time scale accounts for the rest."

3. **Methods, "All models run without reasoning and answer with one word."** This is now false. Replace with: "The five open models answer with one word without reasoning. The two proprietary models are reasoning models (effort 'low'); their choice is read from the generated answer."
   - **New paragraph.** State the subset, per-pair CIs, valid-answer rate and the three confounds.
   - **Table.** Put the API rows in a separate block with open models and reference rules on the same subset.
   - **Wording.** Say "did not differ from", not "reached" or "equivalent", unless a CI over the 24 pairs supports it.
   - **GPT-6-Astra.** Report it only when complete, or on identical trials and flagged as partial. Its summary value (.606) currently exceeds the rule (.585), probably because the trial set differs.

4. **"A learner that describes people solves the same episodes" and "a shortcut human participants did not use."** This comparison has three problems:
   - The Behrens regressor is "follow the advisor correct on the previous trial". That is the draft's win-stay/lose-shift strategy, where models also show κ ≈ 0. Repeating the outcome label was never tested in people.
   - The learner is an ideal observer whose estimates explained choices in a richer task. The .668 is its argmax, not human accuracy.
   - It was added after the commit, so it is exploratory.

   Retitle the paragraph "A Bayesian volatility learner solves the same episodes". Replace the second quote with: "a shortcut a learner that tracks the advisors would not use". Delete "our models instead repeated the last outcome" as a human contrast.

5. **Reference baselines in Table 1.** Add "always follow A" (.648) and the last-5 rule (.692). A reader should see that a constant policy beats every open-model condition except recent-trial summaries, and that a 5-trial window beats the "Bayes-optimal" learner. Add one sentence explaining why: the learner is optimal under its own generative model, not the block schedule, and K=5 was chosen on episodes with the same schedule. Drop the Chart column, which is "not interpreted".

6. **"A summary passes on its rule" and "Implications."** The API result strengthens the first and undercuts the second. Add: "For the reasoning models, an all-trial summary lowered accuracy below the raw record to the level of its rule (Claude .585 vs .629)." Replace "the results suggest the division of labour described in people … integration rule outside the model" with: "For models that fail at outcome evaluation, an explicit correctness record and an external rule are a remedy; for models that do not, a lossy summary is a cost." Delete "described in people".

7. **Masking paragraph and the kernel-figure caption.**
   - "7–10 points" (development) is a main effect; "3–6 points" (held-out) is a simple effect. Like for like, the simple effects are 4.2 and 7.2 (development) against 4.0 and 5.3 (held-out) for the same two models.
   - "Recovers most of the gap" is unsupported. It is 40% and 65% on development and 40–60% on held-out.
   - "Replacing the reports did not" holds only while the outcome is visible. Once the outcome is masked, masking the reports adds 2–6 points (my calculation from the masking table).
   - Replace the last sentence with: "The visible outcome label drives the repetition; reports matter once it is removed."

8. **Layout sensitivity.** This is buried in an appendix and one Limitations clause. With the compact record, raw accuracy is .55–.56 and the gap is +.03 and +.01 [−.02, .05]. Add to Results: "The size of the gap depends on layout: with a one-line record it shrank to 1–3 points in the two 8B models (Appendix J); raw accuracy stayed below the count rule in every layout."

9. **Discussion, "the models fail at forming this error" and the cortex mapping.** Models name the correct advisor on a single trial at 1.00, and the API models do it across trials. Cut the fMRI sentence, or reduce it to: "In the human study this step corresponds to an advisor prediction error; we draw no parallel beyond the computational level."
   - **"Which also explains why summaries help in bandit tasks."** Delete. Bandit histories already give the evaluated outcome, so this step is absent there.

10. **Psych-101.** Remove it from the abstract or scope it to one clause.
    - Change "with calibrated power" to "with a simulation-based check of the test (power about 0.5)".
    - Change "The result was not null" to: "In four of five models the slope difference was significant (p = .011–.031; .070 in the fifth); two bootstrap CIs include zero."
    - State that φ ≈ 0 is interpretable only if the model's choices depend on payoffs. Report the fitted value sensitivity.

11. **"Models repeat the previous outcome."** κ is .13–.32, and learning-model fits are near ln 2. Write "tended to repeat". Also reconcile "the kernel peaked at the most recent trial and decayed" with "counting over all past trials was usually the best model" and with the long post-reversal lag. A 15-lag kernel cannot show an all-trial count; say so.

12. **Image paragraph.** "Binding problem" is one reading. The pilot showed a first-listed-image bias (1.00 vs .03). Retitle it "Image reports lower accuracy (exploratory)" and cut it to two sentences.

13. **Abstract, "described accuracy was used almost perfectly."** This holds for development items and text reports only (.59–.84 with images). Say so. Mark the Bayesian reference and the masking experiment as added after the plan was fixed.

14. **Smaller wording points.**
    - "Both language-model families": the second family is one model. Write "in Gemma-3 and in all four Qwen3-based models".
    - "Biologically grounded": write "adapted from human neuroscience".
    - "Can be measured in a model in the same way and compared with people": the draft itself shows the volatility signature is undetectable in 80 trials.

15. **Terms and codes.**
    - Define Psych-101 and hazard rate in the abstract.
    - Define or remove: "CPT regressor", "hard accuracy", "label-aligned logits", "the 0.8 gate".
    - Remove codes: "v3.1/v3", "deviation D1", "M0–M4", "RTX 3090".
    - Appendix H must list the correctness-only condition now queued.

## 4. Overreaching or unsupported statements

Items 1–14 above cover the main quotes. Further ones:

- **"more accurate (.668) than every model … models matched it only when an external rule summarized recent trials"** — contradicted by the API result.
- **"Models use described accuracy but not raw experience"** (figure caption and paragraph head) — true only for the open models answering directly, with text reports.
- **"matched the Bayesian learner that describes people"** — the match comes from a window tuned on the same schedule, and the learner is not human performance.
- **"so adapting the time scale of integration to the environment is also left to whatever sits outside the model"** — rests on open models in an underpowered replay.
- **"This refines the description–history gap"** — not tested on that paradigm; write "suggests".
- **"the VLMs tested here are not yet such models"** — scope to the open models.
- **"reached the level of the summary"** — Gemma-3-27B's position is 0.89 [0.83, 0.96], and no equivalence test was run.
- **"These studies … are text-only"** — so are your confirmatory results.

## 5. Likely reviewer questions and cheap answers

- **Is the API advantage due to reasoning, scale or readout?**
  - Report the Qwen3-VL-8B-Thinking result either way, with pre-written sentences. If it rises: "reasoning at the same scale removes the loss". If not: "the difference cannot be attributed to reasoning alone".
  - Split Claude's raw accuracy by visible reasoning (77% vs 23%) and flag it as descriptive.
  - Recompute open-model subset accuracy from per-copy greedy answers, exactly as for the API models. Averaging logits across copies removes label bias that per-copy answers keep.

- **Do the API models show the behavioural signature or only the accuracy?**
  - On the 960 choices per format, fit the same lagged-correctness regression with the previous-outcome regressor.
  - Report agreement with the Bayesian learner, the last-5 rule and outcome repetition.
  - Report accuracy by phase.

- **Where are the humans?** State plainly that there are no human data on this variant and that the reference is an ideal observer. Report stable and volatile phases separately, so the reader sees what is learned beyond "A is usually better".

- **Is outcome repetition an artifact of the quotas?** Report the previous-outcome weight and κ on the robustness set, where repetition scores .5. The data exist.

- **Does the model know who is better but fail to use it?** The judgment question ("which forecaster is currently more likely to be correct?") was collected under raw histories and never reported. Report it; it separates evaluation and integration from use.

- **Is this a binding failure or a position bias?** Split image-report accuracy by whether the better advisor's image is listed first.

- **Is φ ≈ 0 in Psych-101 meaningful?** Report payoff sensitivity and the perseveration weight of the model fits. If choices mostly copy the participant's last choice, say that φ is not identified.