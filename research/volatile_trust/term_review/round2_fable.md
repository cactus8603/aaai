## Canonical terms (recommended)

1. **history format** → one of the four VLM inputs (`\Raw{}`, `\RawCorrectness{}`, `\CorrectnessOnly{}`, `\Summary{}`). The paper's own noun is "history format" (the conventions' "input condition"). Keep "condition" for control, masking and hidden/stated-$h$ conditions, and "wording" for the original/reworded/compact prompt versions.
2. **`\Summary{}`** → all-trial counts given to the VLM; **`\Summary{}` variant** → last-$K$, change-point or reversal-aligned only. **Count rule** (all-trial / last-five / change-point / reversal-aligned) → chooses without a VLM. "BOCPD" → only the detection method and the BOCPD posterior-predictive rule.
3. **Bayesian volatility learner** → full form at first use in each paragraph, table or caption. **Computational reference** → any VLM-free learner or rule; **hidden-information diagnostic** → rules that use generator information.
4. **accuracy** (unqualified) → the model's or rule's percent correct; **advisor accuracy** → probability that an advisor is correct (as a proportion). **points** → percentage points only.
5. **choice probability** $y_t$ → probability for advisor A's report; **following probability** → probability for the **currently better advisor** (higher scheduled accuracy on that trial).
6. **episode pair** → episode plus counterbalanced copy; **held-out set** (48), **robustness set** (24, never "held-out"), **development episodes** (16); **advice-taking task**.
7. **outcome** → the revealed event; **outcome label** → the answer word stating it in a history line; **answer words** → the two response alternatives; **`\Summary{}`$-$`\Raw{}` gap**, which a format "recovers".
8. **hazard rate** $h$ (bandit) vs **assumed hazard** $h_{\mathrm{CP}}$ (BOCPD); **forgetting rate** $\phi$; **volatility** → only the advice-taking schedule and the learner's latent variable.

Six fixes below depend on something I could not verify from the source; each is marked "confirm".

## Issues, ranked by severity

### High

**1. Type:** polysemy ("held-out"). **Severity:** high.
- **Where:** App. B, Splits and seeds: `The robustness set has 24 held-out episodes (48 with copies; seed 20261112)`
- **Fix:** `The robustness set has 24 episodes (48 with copies; seed 20261112)`
- This breaks the held-out set (48) vs robustness set (24) convention.

**2. Type:** polysemy ("BOCPD" used for the change-point count rule, which is not the BOCPD posterior-predictive rule). **Severity:** high.
- **Where:** Fig. `fig:strategies` caption: `BOCPD: the change-point count rule; Label: outcome-label bias`
  **Fix:** `BOCPD: the change-point count rule (not the BOCPD posterior-predictive rule); Label: outcome-label bias`. Better still, relabel the figure axis "Change-point rule".
- **Where:** App. E: `the run since the last inferred change point (BOCPD);`
  **Fix:** `the run since the last change point inferred by BOCPD (change-point variant);`
- **Where:** App. E: `to the BOCPD run, or to the last 10 trials`
  **Fix:** `to the change-point run, or to the last 10 trials`
- **Where:** App. D: `the best BOCPD minus the best fixed window ranged from`
  **Fix:** `the best BOCPD posterior-predictive rule minus the best last-$K$ count rule ranged from`
- **Where:** App. D: `accuracy was 68.7\% for the best BOCPD ($h_{\mathrm{CP}}=1/8$)`
  **Fix:** `accuracy was 68.7\% for the best BOCPD posterior-predictive rule ($h_{\mathrm{CP}}=1/8$)`

**3. Type:** notation (one symbol for two different tests). **Severity:** high.
- **Where:** Results, "Following the count rule…": `(Gemma-3-27B: $-1.4$ points, $p_\text{Holm}=.053$; Table~\ref{tab:tests})`
- **Fix:** `(Gemma-3-27B: $-1.4$ points, Holm-corrected equivalence-test $p=.053$; Table~\ref{tab:tests})`
- In Table `tab:main`, $p_\text{Holm}$ is the primary Wilcoxon test; this value is the equivalence test, written $p_{\mathrm{TOST,Holm}}$ in Table `tab:tests`. The fix keeps "TOST" (defined only in App. C) out of the main text.

### Medium

**4. Type:** polysemy ("Count rule" as the all-trial rule only, beside "Last-5 rule"). **Severity:** medium.
- **Where:** Table `tab:apibeh` header: `\shortstack{Count\\rule} & \shortstack{Last-5\\rule}`
  **Fix:** `\shortstack{All-trial\\count rule} & \shortstack{Last-five\\count rule}`
- **Where:** Fig. `fig:strategies` caption: `Last-5 rule and Count rule: counts over the last five and all past trials`
  **Fix:** `Last-5 rule and Count rule: the last-five and all-trial count rules`

**5. Type:** synonymy (count rules named four ways). **Severity:** medium.
- **Where:** App. D: `majority counts over the last $K \in \{3,5,7,10,15,20,30\}$ trials, counts over all past trials, and counts since the last true reversal`
  **Fix:** `the last-$K$ count rule ($K \in \{3,5,7,10,15,20,30\}$), the all-trial count rule, and the count rule since the last true reversal`
- **Where:** App. D: `69.0\% for the best window ($K=3$), 63.2\% for counts over all trials and 73.9\% for counts since the last reversal`
  **Fix:** `69.0\% for the best last-$K$ count rule ($K=3$), 63.2\% for the all-trial count rule and 73.9\% for the count rule since the last true reversal`
- **Where:** Table `tab:normsim`: `Count since true reversal & 74.4 & 73.5`
  **Fix:** `Count rule (reversal-aligned) & 74.4 & 73.5`
- **Where:** Results: `whose choices also agreed weakly with recent-trial count rules`
  **Fix:** `whose choices also agreed weakly with the last-five count rule`
- **Where:** App. C: `the all-trial and recent-trial count rules, the change-point count rule`
  **Fix:** `the all-trial and last-five count rules, the change-point count rule`
- Confirm for the last two: Fig. `fig:strategies` lists only the last-five rule as a recent-trial strategy.

**6. Type:** condition-name (all-trial `\Summary{}` counted as a "variant"). **Severity:** medium.
- **Where:** Table `tab:tests` header: `Model & \Summary{} variant & Model $-$ count rule [90\% CI]`
  **Fix:** `Model & Count range & Model $-$ count rule [90\% CI]`
- **Where:** Table `tab:tests` caption: `with Holm correction over the three \Summary{} variants per model`
  **Fix:** `with Holm correction over \Summary{} (all trials) and its two variants per model`
- **Where:** App. E: `The variants are: all past trials (\Summary{}); the last`
  **Fix:** `The ranges are: all past trials (\Summary{} itself); and, as variants, the last`

**7. Type:** condition-name (macro plus "variants" where the text says plain `\Summary{}`). **Severity:** medium.
- **Where:** Table `tab:readout` caption: `Gemma-3-27B's failures concentrate in \Summary{} variants and chart histories`
- **Fix:** `Gemma-3-27B's failures concentrate in \Summary{} and correctness-chart histories`
- The adjacent text reports the failure "for `\Summary{}`". Confirm: add "and its variants" only if variants were pooled.

**8. Type:** polysemy ("main set"). **Severity:** medium.
- **Where:** App. I: `smaller than on the main set but positive`
- **Fix:** `smaller than on the held-out set but positive`
- App. B defines "main set" as development plus held-out episodes.

**9. Type:** polysemy ("points" = percentage points, payoff points, plot markers). **Severity:** medium.
- **Where:** App. H: `last observed values (per 10 points)` → **Fix:** `last observed values (per 10 payoff points)`
- **Where:** Table `tab:stale` caption: `log-odds per 10 points of last-observed value difference` → **Fix:** `log-odds per 10 payoff points of last-observed value difference`
- **Where:** Fig. `fig:teaser` caption: `Points show equal-weight mean accuracies` → **Fix:** `Markers show equal-weight mean accuracies`
- **Where:** Fig. `fig:masking_effects` caption: `Points show accuracy changes in percentage points` → **Fix:** `Markers show accuracy changes in percentage points`

**10. Type:** notation ("percentage points" not shortened after the Methods definition). **Severity:** medium.
- **Where:** Results: `gap was $+8.4$ to $+16.4$ percentage points and significant` → **Fix:** `gap was $+8.4$ to $+16.4$ points and significant`
- **Where:** Results: `by only 1.1--5.4 percentage points over \Raw{}` → **Fix:** `by only 1.1--5.4 points over \Raw{}`
- Two main-text occurrences. Five more are in appendix running text (App. C: three; App. D: two); shorten them too unless appendices are meant to stand alone. Captions can keep the full form.

**11. Type:** polysemy ("gap" unqualified, next to the description–history gap) and synonymy (removes/recovers). **Severity:** medium.
- **Where:** Abstract: `narrowed the gap in two open models` → **Fix:** `narrowed the \Summary{}$-$\Raw{} gap in two open models`
- **Where:** Results heading: `\CorrectnessOnly{} recovers most of the gap;` → **Fix:** `\CorrectnessOnly{} recovers most of the \Summary{}$-$\Raw{} gap;`
- **Where:** Discussion: `\CorrectnessOnly{} removed most of the gap, \RawCorrectness{} helped little` → **Fix:** `\CorrectnessOnly{} recovered most of the \Summary{}$-$\Raw{} gap, \RawCorrectness{} helped little`
- **Where:** Introduction: `\CorrectnessOnly{} removes most of the` → **Fix:** `\CorrectnessOnly{} recovers most of the`
- **Where:** App. C: `the fraction of the main-task gap recovered` → **Fix:** `the fraction of the \Summary{}$-$\Raw{} gap recovered`

**12. Type:** condition-name (named set in plain words) plus "gap". **Severity:** medium.
- **Where:** Limitations: `we therefore repeat the primary test on episodes with independent outcomes, where the gap was about half as large`
- **Fix:** `we therefore repeat the primary test on the robustness set (independent outcomes), where the \Summary{}$-$\Raw{} gap was about half as large`

**13. Type:** synonymy/polysemy (format, layout, wording, presentation). **Severity:** medium.
- **Where:** App. I heading: `\subsection{I. Robustness, Prompt Format and Masking}` → **Fix:** `\subsection{I. Robustness, Prompt Wording and Masking}`
- **Where:** Results: `stayed below the count rule in every layout (development episodes; Appendix~I)` → **Fix:** `stayed below the count rule in every wording (development episodes; Appendix~I)`
- **Where:** Results: `The size of the gap depends on presentation:` → **Fix:** `The size of the \Summary{}$-$\Raw{} gap depends on the prompt wording:`
- **Where:** App. H: `These exploratory formats are interpreted` → **Fix:** `These exploratory presentations are interpreted`
- App. I and Table `tab:wording` say "in every wording" and use a "Wording" column.

**14. Type:** polysemy ("history" used for the format; "representation"; "condition"). **Severity:** medium.
- **Where:** Results: `this history remained below the last-five count rule` → **Fix:** `\CorrectnessOnly{} remained below the last-five count rule`
- **Where:** Discussion: `history representation strongly affected` → **Fix:** `history format strongly affected`
- **Where:** Introduction: `we test which representations support good choices` → **Fix:** `we test which history formats support good choices`
- **Where:** Table `tab:readout` caption: `pooled over main-task histories and report types` → **Fix:** `pooled over history formats and report types`
- **Where:** Table `tab:apibeh` header: `Model & History & Acc.` → **Fix:** `Model & History format & Acc.`
- **Where:** App. I: `the natural-language layout used in all main conditions` → **Fix:** `the natural-language layout used in all four history formats`

**15. Type:** condition-name (macro used as a noun for a control history). **Severity:** medium.
- **Where:** App. C: `Qwen3-VL-8B with a long \Raw{} (53.3\%` → **Fix:** `Qwen3-VL-8B with a long easy history in \Raw{} (53.3\%`
- **Where:** App. G: `a short \Raw{}, a short \CorrectnessOnly{}, and a long \CorrectnessOnly{}` → **Fix:** `a short easy history in \Raw{}, and short and long easy histories in \CorrectnessOnly{}`
- **Where:** App. E: `at or below full \Raw{} (46.8--54.9\%)` → **Fix:** `at or below the full \Raw{} history (46.8--54.9\%)`

**16. Type:** synonymy (short form "Bayesian learner" at first use in a paragraph or table). **Severity:** medium. Six occurrences.
- **Where:** Introduction: `to a Bayesian learner, so the format effect` → **Fix:** `to a Bayesian volatility learner, so the history-format effect`
- **Where:** Table `tab:main`: `Bayesian learner & \multicolumn{4}{c}{66.8 / 62.9}` → **Fix:** `Bayesian volatility learner & \multicolumn{4}{c}{66.8 / 62.9}`
- **Where:** Results: `the Bayesian learner reached 62.9\% and the count rules` → **Fix:** `the Bayesian volatility learner reached 62.9\% and the count rules`
- **Where:** Discussion: `to the Bayesian learner, with no increase` → **Fix:** `to the Bayesian volatility learner, with no increase`
- **Where:** App. B: `so the Bayesian learner is a computational reference` → **Fix:** `so the Bayesian volatility learner is a computational reference`
- **Where:** App. J: `learner--quota disagreement trials` → **Fix:** `Bayesian-learner--quota disagreement trials`

**17. Type:** polysemy ("reference" vs "baseline" vs hidden-information rows). **Severity:** medium.
- **Where:** App. D: `together with diagnostic references to our episodes` → **Fix:** `together with hidden-information diagnostics, to our episodes`
- **Where:** Table `tab:normsim`: `Rule or reference (no VLM)` → **Fix:** `Rule (no VLM)`
- **Where:** Table `tab:normsim` caption: `are references, not available model strategies` → **Fix:** `are hidden-information diagnostics, not available model strategies`
- **Where:** Table `tab:variants` caption: `a reference that uses true reversal times` → **Fix:** `a hidden-information diagnostic that uses true reversal times`
- **Where:** App. D: `These rule baselines were computed` → **Fix:** `These rule accuracies were computed`

**18. Type:** synonymy (currently better advisor; following probability). **Severity:** medium.
- **Where:** Results: `the mean probability of following the currently better advisor stayed below .5` → **Fix:** `the mean following probability stayed below .5`
- **Where:** App. C: `the probability of following the currently more accurate advisor by phase, and learning curves aligned to reversals` → **Fix:** `the following probability by phase, and following-probability curves aligned to reversals`
- **Where:** App. I: `The models named the scheduled better advisor` → **Fix:** `The models named the currently better advisor`
- **Where:** Table `tab:judgpos` caption: `Names/follows better: the scheduled better advisor;` → **Fix:** `Names/follows better: the currently better advisor (higher scheduled accuracy on that trial);`
- **Where:** Results: `they followed the newly better advisor` → **Fix:** `they followed the currently better advisor`
- **Where:** Results (described-accuracy control, a different sense): `they followed the better advisor on only 59--84\% of items` → **Fix:** `they chose the advisor described as better on only 59--84\% of items`

**19. Type:** polysemy ("follow" used for choosing A and for agreeing with `\Summary{}`). **Severity:** medium.
- **Where:** App. J: `a logistic regression of following A on its correctness at lags 1--10` → **Fix:** `a logistic regression of choosing advisor A's report on A's correctness at lags 1--10`
- **Where:** App. I: `InternVL3.5-8B followed \Summary{} on only 72\% of non-tied trials` → **Fix:** `InternVL3.5-8B chose the advisor with more correct trials in \Summary{} on only 72\% of non-tied trials`

**20. Type:** synonymy/polysemy (outcome vs outcome label; answer labels vs answer words). **Severity:** medium.
- **Where:** Table `tab:masking_accuracy` caption: `Masking past outcomes improves accuracy;` → **Fix:** `Masking past outcome labels improves accuracy;`
- **Where:** same caption: `whether past outcomes and advisor reports are shown` → **Fix:** `whether past outcome labels and advisor reports are shown`
- **Where:** Fig. `fig:masking_effects` caption: `masking the reports, the outcome, or both against` → **Fix:** `masking the reports, the outcome label, or both against`
- **Where:** Results: `block contains each outcome five times` → **Fix:** `block contains each outcome label five times`
- **All occurrences:** `answer labels` → `answer words` (4: Fig. `fig:teaser` caption, Methods, App. A, App. G); `answer-label mapping` → `answer-word mapping` (1, App. B).

**21. Type:** definition ("candidate" logits never defined in the main text). **Severity:** medium.
- **Where:** Fig. `fig:gap` caption: `Open-model choices use candidate logits;`
- **Fix:** `Open-model choices use the logits of the two answer words (candidate logits);`

**22. Type:** polysemy ("counterbalanced copy" = the swapped twin, but "copies" = both members of a pair). **Severity:** medium.
- **Where:** Methods: `after averaging label-aligned logits over the counterbalanced copies; exact ties`
- **Fix:** `after averaging label-aligned logits over the two members of each episode pair (the episode and its counterbalanced copy; hereafter ``copies''); exact ties`
- Later uses such as "copies scored separately", "per-copy scoring" and "both counterbalanced copies" are then covered.

**23. Type:** synonymy/polysemy ("pair"). **Severity:** medium.
- **Where:** App. B: `Analyses average the label-aligned logits within each pair.` → **Fix:** `Analyses average the label-aligned logits within each episode pair.`
- **Where:** App. B: `alternates in pairs, and non-pet` → **Fix:** `alternates every two episodes, and non-pet`
- **Where:** App. G: `shows the original and selected pairs at 224 pixels` → **Fix:** `shows the original and selected icon pairs at 224 pixels`

**24. Type:** notation (unit of the bootstrap named differently). **Severity:** medium.
- **Where:** Table `tab:behrens` caption: `with a 95\% bootstrap CI over episodes`
- **Fix:** `with a 95\% bootstrap CI over episode pairs`
- Confirm that the resampling unit is the same; Table `tab:normsim` also says "episode resamples".

**25. Type:** synonymy (task name). **Severity:** medium.
- **All occurrences:** `advice task` → `advice-taking task` (3: Results bandit paragraph, Limitations, App. F); `advice-task comparisons` → `advice-taking-task comparisons` (1, Limitations).

**26. Type:** polysemy (volatility/hazard) and definition. **Severity:** medium.
- **Where:** Results: `this task varies volatility across games` → **Fix:** `this task varies the hazard rate across games`
- **Where:** App. F: `in which volatility differs across whole games` → **Fix:** `in which the hazard rate differs across whole games`
- **Where:** App. D: `for hazard rates between 1/8 and 1/30` → **Fix:** `for assumed hazards $h_{\mathrm{CP}}$ between 1/8 and 1/30`
- **Where:** App. H: `at the other hazards` → **Fix:** `at the other hazard rates`
- **Where:** Introduction (used before the Methods definition): `participants' estimated forgetting rose with hazard rate, whereas` → **Fix:** `participants' estimated forgetting rose with the hazard rate (the per-trial probability of a payoff change), whereas`
- **Where:** Abstract: `increased with change frequency` → **Fix:** `increased with the probability of change`

**27. Type:** synonymy (forgetting rate vs learning rate vs discounting). **Severity:** medium.
- **Where:** App. H: `report no learning rates by hazard rate` → **Fix:** `report no forgetting rates by hazard rate` (confirm that $\phi$ is meant)
- **Where:** Results: `the human discounting estimate did not increase with hazard rate` → **Fix:** `the participants' stale-value discounting (stale-minus-fresh weight; Appendix~H) did not increase with hazard rate`
- **Where:** Results: `even this learner's learning rate rises` → **Fix:** `even this learner's implied learning rate rises`

**28. Type:** synonymy (integration kernel). **Severity:** medium.
- **Where:** Results: `correctness kernel in Fig.~\ref{fig:kernel}` → **Fix:** `integration kernel in Fig.~\ref{fig:kernel}`
- **Where:** Fig. `fig:kernel` caption: `Curves show correctness integration kernels` → **Fix:** `Curves show integration kernels`
- **Where:** Results: `The kernel weights on past correctness` → **Fix:** `The integration-kernel weights on past correctness`
- **Where:** Table `tab:recovery` caption: `Strategy and kernel analyses` → **Fix:** `Strategy and integration-kernel analyses`

**29. Type:** polysemy ("readout" for the scoring scheme). **Severity:** medium.
- **Where:** App. J: `\paragraph{Sensitivity to counterbalancing the readout (exploratory).}` → **Fix:** `\paragraph{Sensitivity to per-copy scoring (exploratory).}`
- **Where:** App. J: `stale-value regression and readout sensitivity were specified` → **Fix:** `stale-value regression and per-copy scoring sensitivity were specified`

**30. Type:** definition/synonymy ("block quotas" used once, undefined; elsewhere "outcome quotas"). **Severity:** medium.
- **Where:** Methods: `without modeling our block quotas`
- **Fix:** `without modeling the fixed per-block counts described above`

**31. Type:** synonymy/definition ("labeled image" vs "composite image"; "panel" undefined in the main text). **Severity:** medium.
- **Where:** Results: `both reports were shown in one labeled image, although each panel was read correctly`
- **Fix:** `both reports were shown in one composite image (one panel per advisor, name above each panel), although each panel was read correctly`

**32. Type:** synonymy ("fixed models"; "summarizes"). **Severity:** medium.
- **Where:** Fig. `fig:curves_models` caption: `summarizes the same curves over the five fixed models`
- **Fix:** `averages the same curves over the five open models`

**33. Type:** polysemy ("correct trials" could be read as the model's). **Severity:** medium.
- **Where:** Abstract: `\Summary{} (correct-trial counts)` → **Fix:** `\Summary{} (each advisor's count of correct trials)`
- **Where:** Fig. `fig:teaser` caption: `supplies counts of correct trials (open diamond)` → **Fix:** `supplies each advisor's count of correct trials (open diamond)`

**34. Type:** polysemy ("described counts" uses the same count sentence as a `\Summary{}` variant, although `\Summary{}` "is not a description"). **Severity:** medium.
- **Where:** Methods: `described counts, short and long easy histories`
- **Fix:** `described counts (a stated count given in place of a history in a control item), short and long easy histories`
- Confirm the gloss.

**35. Type:** synonymy (integration rule / controller / count rule). **Severity:** medium.
- **Where:** Discussion: `and an integration rule outside the model, help models` → **Fix:** `and an integration rule outside the model (here, a count rule), help models`
- **Where:** App. E: `Before trial $t$, the controller runs` → **Fix:** `Before trial $t$, the change-point controller (which selects the trial range for the change-point \Summary{} variant and count rule) runs`

**36. Type:** synonymy (delta learner / delta rule / delta agent). **Severity:** medium.
- **Where:** App. F: `a constant-rate delta rule fitted to the same episode` → **Fix:** `a constant-rate delta learner fitted to the same episode`
- **Where:** App. F: `This defines the synthetic delta agent` → **Fix:** `This defines the synthetic delta learner`
- **Where:** Table `tab:recovery`: `Delta rule ($\alpha=0.3$, $\beta=8$)` → **Fix:** `Delta learner ($\alpha=0.3$, $\beta=8$)`
- **Where:** Table `tab:recovery`: `50\% delta &` → **Fix:** `50\% delta learner &`
- **Where:** Table `tab:recovery` caption: `For the delta rule the fitted kernel` → **Fix:** `For the delta learner the fitted integration kernel`
- **Where:** Table `tab:recovery` caption: `so the delta rule agrees best` → **Fix:** `so the delta learner agrees best`
- **Where:** App. F heading: `\paragraph{Constant-rate reference.}` → **Fix:** `\paragraph{Constant-rate delta learner.}`

### Low (grouped)

**37. "raw" as an ordinary adjective** (polysemy).
- Related Work: `contrasted raw and summarized histories` → `contrasted unsummarized and summarized histories`
- App. F: `high raw agreement` → `high uncorrected agreement`
- App. J: `raw candidate logits` → `unnormalized candidate logits`

**38. "format effect"** (synonymy). All occurrences: `format effect` → `history-format effect` (3: Introduction bullet, Results proprietary paragraph, App. J sensitivity paragraph).

**39. computation / step** (synonymy).
- Methods: `the result of step (i)` → `the result of computation (i)`
- Methods: `the result of steps (i) and (ii)` → `the result of computations (i) and (ii)`
- Methods: `(step (i) alone)` → `(computation (i) alone)`

**40. Model-group names** (synonymy).
- `open VLMs` → `open models` (3: Introduction bullet, Discussion ×2). Keep "open-weight VLMs" at first use in the Abstract and Methods.
- App. H: `the proprietary API models` → `the proprietary models`
- Results: `for these models, and accuracy with \Summary{}` → `for the two proprietary models, and accuracy with \Summary{}` (the referent is ambiguous after the open-model sentence).

**41. right / correct; source / advisor; option / machine** (synonymy).
- Introduction: `how often each advisor was right` → `how often each advisor was correct`
- Results: `had been right on every recent trial` → `had been correct on every recent trial`
- Discussion: `extracting source-relevant evidence` → `extracting evidence about advisor correctness`
- Results: `forgets the unchosen option faster` → `forgets the unchosen machine faster`

**42. trial / day / item outside quotations.**
- App. A: `day 1 reads` → `trial 1 reads`
- App. J: `On matched items with per-copy scoring` → `On the same trials with per-copy scoring`

**43. Polysemous small words.**
- App. I heading: `blocks use of its correctness statements` → `prevents use of its correctness statements` ("block" is the 10-trial unit).
- Table `tab:controls_full` caption: `Lower block:` → `Lower part:`
- App. F: `must be held out together` → `must be assigned to the same cross-validation fold`
- App. F heading: `\paragraph{Recovery.}` → `\paragraph{Parameter recovery.}`
- Table `tab:apibeh` caption and App. J: "genuine reversals" → "true reversals".
- Fig. `fig:curves` caption: `Recent-trial \Summary{} variants` — name the variants plotted (e.g. "Last-five and change-point