## Recommended canonical terms

- **Advisor accuracy** → an advisor’s probability or realized frequency of being correct; **prediction accuracy** → a model’s or baseline’s task score.
- **Episode** → a generated trial sequence; **episode pair** → an episode and its counterbalanced copy.
- **`\Raw{}`, `\RawCorrectness{}`, `\CorrectnessOnly{}`, `\Summary{}`** → VLM input conditions; distinguish **history format**, **layout**, and **output format**.
- **All-trial / last-five / change-point count rule** → non-VLM decision rules; identify the corresponding **`\Summary{}` variant** separately.
- **Bayesian volatility learner** → the hierarchical benchmark; **BOCPD posterior-predictive rule** and **change-point count rule** → distinct BOCPD-based baselines.
- **Choice probability `$y_t$`** → probability assigned to advisor A’s report; **following probability** → probability assigned to the currently more accurate advisor’s report.
- **Advisor-correctness observation** → whether an advisor was correct; **outcome label** → the displayed answer identifying the realized outcome.
- **Percentage points** → differences between percentage accuracies; use full model names and American spelling consistently.

The edits below concern authorial prose and labels. Literal prompt quotations, cover-story role names inside those quotations, checkpoint/API identifiers, and external dataset split names should remain unchanged.

## High-severity issues

### 1. `\Raw{}` incorrectly denotes unprocessed logits

**Type:** condition-name  
**Severity:** high  
**Where:** Appendix J, “Software and hardware”

**Quote**
```latex
Each run stores the rendered prompt hash, \Raw{} candidate logits, the probability mass on the two answer tokens, and a fingerprint of the configuration.
```

**Fix**
```latex
Each run stores the rendered prompt hash, raw candidate logits, the probability mass on the two answer tokens, and a fingerprint of the configuration.
```

Here the ordinary adjective *raw*, not the input condition, is meant.

### 2. The open-model readout and accuracy definition appear to cover proprietary models too

**Type:** polysemy  
**Severity:** high  
**Where:** Task and Methods, “Models and readout”

**Quote**
```latex
The readout is the choice probability $y_t$ for advisor A's report,
```

**Fix**
```latex
For the open models, the readout is the choice probability $y_t$ for advisor A's report,
```

**Where:** Task and Methods, “Measures and analysis”

**Quote**
```latex
Accuracy is the percentage of trials ($t\ge2$) on which the higher-logit answer is correct,
```

**Fix**
```latex
For the open models, prediction accuracy is the percentage of trials ($t\ge2$) on which the higher-logit answer is correct,
```

The proprietary-model score uses generated answers, not higher-logit answers.

### 3. `\Summary{}` is repeatedly treated as though it were a decision rule

**Type:** condition-name  
**Severity:** high  

| Where | Exact quote | Exact replacement |
|---|---|---|
| Results, proprietary models | `\Summary{} was numerically lower than \Raw{}, at the level of its rule.` | `prediction accuracy with \Summary{} was numerically lower than with \Raw{}, at the level of the all-trial count rule.` |
| Discussion, opening paragraph | `followed \Summary{} as faithfully as its rule` | `agreed with the all-trial count rule when given \Summary{}` |
| Appendix E, “Summary variants” | `On held-out episodes, accuracy followed the rule in every variant:` | `On held-out episodes, model prediction accuracy was close to that of the corresponding count rule in every \Summary{} variant:` |
| Appendix I, prompt wording | `InternVL3.5-8B followed \Summary{} on only 72\% of non-tied trials` | `InternVL3.5-8B agreed with the all-trial count rule on only 72\% of non-tied trials when given \Summary{}` |
| Appendix J, readout quality | `how well it would follow \Summary{} in the requested format` | `how closely it would follow the all-trial count rule when given \Summary{} and answering in the requested one-word output format` |

### 4. “Choice probability” changes its target from advisor A to the currently better advisor

**Type:** polysemy  
**Severity:** high  

The Methods definition of `$y_t$` is advisor-A-specific, whereas the reversal-aligned curves concern the currently better advisor.

**Where:** Task and Methods, “Measures and analysis”

**Quote**
```latex
exact ties receive half credit and chance is 50\%.
```

**Fix**
```latex
exact ties receive half credit and chance-level prediction accuracy is 50\%. Following probability denotes the choice probability assigned to the advisor with higher scheduled accuracy on the current trial.
```

**Where:** Figure `fig:curves` caption

**Quote**
```latex
Curves are mean choice probabilities over the five open models on held-out episodes,
```

**Fix**
```latex
Curves are mean following probabilities over the five open models on held-out episode pairs,
```

**Where:** Figure `fig:curves_models` caption

**Quote**
```latex
Each curve is a mean choice probability on held-out episodes;
```

**Fix**
```latex
Each curve is a mean following probability on held-out episode pairs;
```

**Where:** Appendix J, Table `tab:apibeh` caption

**Quote**
```latex
These windows match Fig.~\ref{fig:curves}, which plots choice probabilities instead.
```

**Fix**
```latex
These windows match Fig.~\ref{fig:curves}, which plots following probabilities rather than proportions of discrete choices.
```

### 5. “Outcome” denotes advisor correctness in the effective-learning-rate diagnostic

**Type:** polysemy  
**Severity:** high  
**Where:** Appendix F, “Effective learning rate”

**Quote**
```latex
The update of the model's choice probability in logit space after each outcome, regressed on the signed outcome separately for the stable and volatile phases, measures how strongly the model responds to new outcomes.
```

**Fix**
```latex
The update of the model's choice probability in logit space after each advisor-correctness observation, regressed on the signed advisor-correctness observation separately for the stable and volatile phases, measures how strongly the model responds to new advisor-correctness observations.
```

**Where:** Appendix F, “Can these episodes reveal volatility-dependent learning?”

**Quote**
```latex
volatile/stable ratio of the summed weight of the last three outcomes:
```

**Fix**
```latex
volatile/stable ratio of the summed integration-kernel weights on the last three advisor-correctness observations:
```

This distinguishes the diagnostic input from the task’s rain/dry, cat/dog, or other outcome labels.

### 6. “BOCPD” names both an inference algorithm and different decision rules

**Type:** polysemy  
**Severity:** high  

Figure `fig:strategies` identifies its BOCPD strategy as the change-point count rule, whereas Appendix D separately reports a posterior-predictive BOCPD baseline. Make those names explicit.

**Where:** Appendix C, statistical analysis

**Quote**
```latex
counts over all or recent trials, BOCPD and a label bias.
```

**Fix**
```latex
all-trial and recent-trial count rules, the change-point count rule and an outcome-label bias.
```

**Where:** Appendix F, strategy-agreement paragraph

**Quote**
```latex
even a constant-rate delta learner can agree most with BOCPD,
```

**Fix**
```latex
even a constant-rate delta learner can agree most with the change-point count rule,
```

**Where:** Table `tab:recovery`

**Quote**
```latex
Delta rule ($\alpha=0.3$, $\beta=8$) & BOCPD & 0.94
```

**Fix**
```latex
Delta rule ($\alpha=0.3$, $\beta=8$) & Change-point count rule & 0.94
```

**Where:** Table `tab:recovery` caption

**Quote**
```latex
the delta rule agrees best with BOCPD;
```

**Fix**
```latex
the delta rule agrees best with the change-point count rule;
```

**Where:** Figure `fig:strategies` caption

**Quote**
```latex
Last-5 rule and Count rule: counts over the last five and all past trials; BOCPD: change-point rule;
```

**Fix**
```latex
Last-5 rule and Count rule: the last-five and all-trial count rules; BOCPD: the change-point count rule;
```

**Where:** Table `tab:normsim`

**Quote**
```latex
BOCPD posterior observer & 67.7 & 67.5 \\
```

**Fix**
```latex
BOCPD posterior-predictive rule & 67.7 & 67.5 \\
```

**Where:** Table `tab:normsim` caption

**Quote**
```latex
Change-point count uses the inferred run's majority; the posterior observer uses BOCPD's predictive probability.
```

**Fix**
```latex
The change-point count rule uses the inferred run's majority; the BOCPD posterior-predictive rule uses BOCPD's predictive probability.
```

### 7. “Accuracy” is incorrectly made the agent that chooses

**Type:** polysemy  
**Severity:** high  
**Where:** Appendix D, Bayesian implementation

**Quote**
```latex
Accuracy follows A when $q_t>.5$ and B when $q_t<.5$, with half credit at ties.
```

**Fix**
```latex
The Bayesian volatility learner chooses A when $q_t>.5$ and B when $q_t<.5$; prediction accuracy assigns half credit at ties.
```

### 8. An integration-kernel caption names the bootstrap unit as episodes rather than episode pairs

**Type:** synonymy  
**Severity:** high  
**Where:** Figure `fig:kernel_models` caption

**Quote**
```latex
Integration kernels per model (shaded: 95\% bootstrap CI over episodes)
```

**Fix**
```latex
Integration kernels per model (shaded: 95\% bootstrap CI over episode pairs)
```

This aligns the caption’s unit terminology with Methods. The no-VLM simulations’ references to generated episodes need not be changed.

### 9. “Reversal-aligned reference” denotes both a VLM input and a no-VLM rule

**Type:** polysemy  
**Severity:** high  

| Where | Exact quote | Exact replacement |
|---|---|---|
| Appendix D, count rules | `The reversal-aligned reference uses reversals only up to $t-1$,` | `The reversal-aligned count rule uses reversals only up to $t-1$,` |
| Appendix E, variant list | `the trials since the last true reversal up to $t-1$ (reversal-aligned reference).` | `the trials since the last true reversal up to $t-1$ (reversal-aligned \Summary{} variant).` |
| Appendix E, variant results | `The same holds for the reversal-aligned reference (72.7--74.4\%; rule alone 74.4\%).` | `The same holds for the reversal-aligned \Summary{} variant (72.7--74.4\%; reversal-aligned count rule alone 74.4\%).` |
| Table `tab:variants` caption | `Since reversal uses true change times hidden from the models and is a reference.` | `Since reversal denotes the reversal-aligned \Summary{} variant, a reference input based on true reversal times hidden from the models.` |
| Table `tab:normsim` | `Since true reversal & 74.4 & 73.5 \\` | `Count rule (since true reversal) & 74.4 & 73.5 \\` |

### 10. “Held-out” can refer to either the main set or the separately held-out robustness set

**Type:** polysemy  
**Severity:** high  

**Where:** Appendix I, immediately after the robustness-set result

**Quote**
```latex
On held-out episodes the gap was positive in every non-pet cover story for every model
```

**Fix**
```latex
On the main held-out episode pairs the gap was positive in every non-pet cover story for every model
```

**Where:** Table `tab:normsim` header

**Quote**
```latex
Rule or reference (no VLM) & Held-out & Robustness \\
```

**Fix**
```latex
Rule or reference (no VLM) & Main held-out set & Robustness set \\
```

**Where:** Table `tab:normsim` caption

**Quote**
```latex
Rules on 48 held-out and 24 robustness episodes.
```

**Fix**
```latex
Rules on 48 main held-out episodes and 24 robustness-set episodes.
```

## Medium-severity issues

### 11. Accuracy differences use “points” rather than “percentage points”

**Type:** notation-spelling  
**Severity:** medium  

Apply the following find/replace pairs. Each occurs once except the explicitly marked pair, which occurs twice: **26 replacements in total**. Do not replace “points” in the bandit payoff descriptions or “grid points.”

| Where | Exact find | Exact replacement |
|---|---|---|
| Results, gap recovery | `$+8.8$ to $+13.9$ points` | `$+8.8$ to $+13.9$ percentage points` |
| Results, compact layout | `1--3 points` | `1--3 percentage points` |
| Results, masking | `2.7--6.4 points` | `2.7--6.4 percentage points` |
| Results, masking | `0.4 points` | `0.4 percentage points` |
| Results, masking | `2.1--5.5 points` | `2.1--5.5 percentage points` |
| Results, masking | `within 4 points` | `within 4 percentage points` |
| Results, count-rule equivalence | `$\pm$2 points` | `$\pm$2 percentage points` |
| Results, count-rule equivalence | `$-1.4$ points` | `$-1.4$ percentage points` |
| Results, variants | `4--6 points` | `4--6 percentage points` |
| Results, variants | `4--8 points` | `4--8 percentage points` |
| Results, proprietary models | `$-4.4$ points` | `$-4.4$ percentage points` |
| Results, image reports | `7--10 points` | `7--10 percentage points` |
| Discussion limitations; Appendix I robustness — **all occurrences, 2** | `$+3.6$ to $+8.3$ points` | `$+3.6$ to $+8.3$ percentage points` |
| Appendix I, cover stories | `$+8.0$ to $+17.9$ points` | `$+8.0$ to $+17.9$ percentage points` |
| Appendix I, development masking | `7.4 and 9.6 points` | `7.4 and 9.6 percentage points` |
| Appendix I, development masking | `$+1.1$ and $-0.2$ points` | `$+1.1$ and $-0.2$ percentage points` |
| Appendix I, development masking | `1.5--2.1 points` | `1.5--2.1 percentage points` |
| Appendix I, held-out masking | `$+2.7$ to $+6.4$ points` | `$+2.7$ to $+6.4$ percentage points` |
| Appendix I, held-out masking | `$-0.4$ and $+0.3$ points` | `$-0.4$ and $+0.3$ percentage points` |
| Appendix I, held-out masking | `$+2.9$ to $+6.8$ points` | `$+2.9$ to $+6.8$ percentage points` |
| Appendix I, held-out masking | `$+7.3$ to $+10.7$ points` | `$+7.3$ to $+10.7$ percentage points` |
| Appendix I, held-out masking | `$-4.0$ to $+0.8$ points` | `$-4.0$ to $+0.8$ percentage points` |
| Appendix I, reworded prompts | `$+12$ and $+15$ points` | `$+12$ and $+15$ percentage points` |
| Appendix I, compact prompts | `$+3.4$ points` | `$+3.4$ percentage points` |
| Appendix I, larger-model extensions | `$+12.0$ points` | `$+12.0$ percentage points` |

### 12. The hierarchical benchmark has several names

**Type:** synonymy  
**Severity:** medium  

**All occurrences in rendered text:**

| Exact find | Exact replacement | Count |
|---|---|---:|
| `Bayesian learner` | `Bayesian volatility learner` | 7 |
| `Bayesian reference learner` | `Bayesian volatility learner` | 2 |

The seven first-row occurrences are in the Introduction contribution, Table `tab:main`, the proprietary-model Results paragraph twice, the Discussion opening, Appendix B’s relation-to-human-study paragraph, and Table `tab:behrens` caption.

Also make these contextual replacements:

| Where | Exact quote | Exact replacement |
|---|---|---|
| Discussion, limitations | `in the Bayesian reference.` | `in the Bayesian volatility learner.` |
| Table `tab:behrens` | `Behrens-inspired Bayesian reference on our episodes` | `Behrens-inspired Bayesian volatility learner on our episodes` |
| Appendix D heading | `\subsection{D. Rule Simulations and Bayesian Reference}` | `\subsection{D. Rule Simulations and Bayesian Volatility Learner}` |
| Appendix D paragraph heading | `\paragraph{Bayesian reference implementation.}` | `\paragraph{Bayesian volatility learner implementation.}` |
| Appendix D implementation | `Our hierarchical learner follows the volatility-learning idea` | `Our Bayesian volatility learner follows the volatility-learning idea` |
| Appendix J, deviations | `Masking, prompt-wording checks and the Bayesian reference were added` | `Masking, prompt-wording checks and the Bayesian volatility learner were added` |

Generic *Bayesian observers* in Related Work and the bandit’s *normative learner* refer to other concepts and should not receive these replacements.

### 13. The task’s advisors are called reporters in authorial descriptions

**Type:** synonymy  
**Severity:** medium  

**Where:** Task and Methods and Appendix A task details  
**All occurrences of this exact phrase: 2**

**Find**
```latex
in which reporters say whether a hidden animal is a cat or a dog
```

**Replace**
```latex
in which advisors say whether a hidden animal is a cat or a dog
```

**Where:** Results, count-rule adaptation

**Quote**
```latex
adapts to a volatile source
```

**Fix**
```latex
adapts to an advisor whose accuracy changes
```

Keep *reporter* in the literal pet prompt and the cover-story role table. Likewise, generic *sources* in the discussion of tools, documents, and other agents need not become *advisors*.

### 14. The scored sample is variously called episodes, pairs, and episode pairs

**Type:** synonymy  
**Severity:** medium  

Use *episode* for the generated sequence and *episode pair* when naming the scored or aggregated unit.

| Where | Exact quote | Exact replacement |
|---|---|---|
| Abstract | `On 48 held-out episodes, four of five open-weight VLMs` | `On 48 held-out episode pairs, four of five open-weight VLMs` |
| Figure `fig:teaser` caption | `A pair comprises an episode and its counterbalanced copy,` | `An episode pair comprises an episode and its counterbalanced copy,` |
| Figure `fig:kernel` caption | `on held-out test episodes with text reports.` | `on held-out episode pairs with text reports.` |
| Figure `fig:strategies` caption | `Cohen's $\kappa$ on held-out episodes,` | `Cohen's $\kappa$ on held-out episode pairs,` |
| Table `tab:mm` caption | `main-task accuracy uses held-out episodes with \CorrectnessOnly{}.` | `main-task accuracy uses held-out episode pairs with \CorrectnessOnly{}.` |
| Table `tab:variants` caption | `\Summary{} variants and truncated \Raw{} on held-out episodes (accuracy in percent);` | `\Summary{} variants and truncated \Raw{} on held-out episode pairs (accuracy in percent);` |
| Appendix J, readout sensitivity | `All 48 held-out pairs and trials $t\ge2$ were retained` | `All 48 held-out episode pairs and trials $t\ge2$ were retained` |

The reversal-curve captions are covered in issue 4; the bootstrap-unit correction is covered in issue 8. References to individual prompt episodes, generated schedules, and no-VLM simulations can remain *episodes*.

### 15. “Test” and “held-out” alternate as dataset names

**Type:** synonymy  
**Severity:** medium  

| Where | Exact quote | Exact replacement |
|---|---|---|
| Figure `fig:teaser` caption | `48 held-out test episode pairs` | `48 held-out episode pairs` |
| Task and Methods | `48 separate episodes form the held-out test set.` | `48 separate episodes form the main held-out set.` |
| Figure `fig:gap` caption | `on the same test trials.` | `on the same held-out trials.` |
| Table `tab:main` | `Full test set: 48 episode pairs; copies combined before scoring` | `Full held-out set: 48 episode pairs; copies combined before scoring` |

The remaining *held-out test episodes* occurrence is included in issue 14. Keep *test split* for the Oxford-IIIT Pet dataset.

### 16. Answer labels and the answer-label mapping are described as though they were interchangeable

**Type:** synonymy  
**Severity:** medium  

Appendix B’s *answer-label mapping reversed* is the clearest terminology.

| Where | Exact quote | Exact replacement |
|---|---|---|
| Figure `fig:teaser` caption | `with advisor names and answer labels swapped.` | `with advisor names swapped and the answer-label mapping reversed.` |
| Task and Methods | `advisor names and answer labels are swapped while outcomes are identical` | `advisor names are swapped and the answer-label mapping is reversed while the underlying outcome sequence is unchanged` |
| Appendix A, task details | `advisor names, answer labels and the order in which the instructions introduce the advisors are swapped while outcomes and images are identical` | `advisor names are swapped, the answer-label mapping is reversed, and the advisor-introduction order is reversed, while the underlying outcome sequence and images are unchanged` |

### 17. `\Summary{}` is written as plain text in headings naming the condition

**Type:** condition-name  
**Severity:** medium  

**Where:** Appendix E subsection heading

**Quote**
```latex
\subsection{E. Change-Point Controller and Summary Variants}
```

**Fix**
```latex
\subsection{E. Change-Point Controller and \Summary{} Variants}
```

**Where:** Appendix E paragraph heading

**Quote**
```latex
\paragraph{Summary variants.}
```

**Fix**
```latex
\paragraph{\Summary{} variants.}
```

Plain *summary* in quoted prompts and *history summaries* in Related Work are ordinary concepts, not missing macros.

### 18. Variant names sometimes become bare “change point” or “window”

**Type:** condition-name  
**Severity:** medium  
**Where:** Results, count-rule-following paragraph

**Quote**
```latex
equivalent to the rule in five of five models for the change point and three of five for the window
```

**Fix**
```latex
equivalent to the corresponding count rule in five of five models for the change-point \Summary{} variant and three of five for the last-five \Summary{} variant
```

**Quote**
```latex
two (change point) or three (last five) trials after a reversal
```

**Fix**
```latex
two trials for the change-point \Summary{} variant or three trials for the last-five \Summary{} variant after a reversal
```

**Where:** Appendix E, variant results

**Quote**
```latex
last 5 trials 66.7--69.1\% (rule alone 69.2\%), change point 66.6--67.7\% (67.8\%), last 10 trials 60.5--61.6\% (61.2\%), last 20 trials 58.1--58.6\% (57.9\%).
```

**Fix**
```latex
the last-five \Summary{} variant 66.7--69.1\% (count rule alone 69.2\%), the change-point \Summary{} variant 66.6--67.7\% (67.8\%), the last-10 \Summary{} variant 60.5--61.6\% (61.2\%), and the last-20 \Summary{} variant 58.1--58.6\% (57.9\%).
```

### 19. Unqualified “rule” obscures the all-trial versus variant-specific count rule

**Type:** synonymy  
**Severity:** medium  

| Where | Exact quote | Exact replacement |
|---|---|---|
| Results, count-rule equivalence | `four of five were equivalent to the rule alone within` | `four of five were equivalent to the all-trial count rule alone within` |
| Table `tab:tests` header | `Model $-$ rule [90\% CI]` | `Model accuracy $-$ count-rule accuracy [90\% CI]` |
| Table `tab:recovery` | `Last-five count rule & Last 5 & 1.00` | `Last-five count rule & Last-five count rule & 1.00` |
| Table `tab:recovery` caption | `the last-5 agent has a flat kernel up to lag 5.` | `the last-five count-rule agent has a flat integration kernel up to lag 5.` |

**Where:** Table `tab:apibeh`, baseline row

**Quote**
```latex
accuracy: repeat 45.8; WSLS 62.3; count 58.5; last 5 67.9; learner 62.9; outcome quota (hidden) 70.2
```

**Fix**
```latex
prediction accuracy: outcome repetition 45.8; WSLS 62.3; all-trial count rule 58.5; last-five count rule 67.9; Bayesian volatility learner 62.9; outcome-quota rule (hidden) 70.2
```

### 20. “Accuracy” in the control conditions is not the main-task prediction-accuracy measure

**Type:** polysemy  
**Severity:** medium  

| Where | Exact quote | Exact replacement |
|---|---|---|
| Results, controls | `every model named the correct advisor on a single shown trial` | `every model correctly identified which advisor's report matched the outcome on a single shown trial` |
| Appendix C, controls | `Image reports lowered accuracy in almost every control condition,` | `Image reports lowered control-item accuracy in almost every control condition,` |
| Appendix G, opening | `With described accuracy, accuracy in four models was lowest` | `In the described-accuracy control, control-item accuracy in four models was lowest` |
| Appendix G, results and limits | `accuracy in the described-accuracy condition, pooled over these two image-report cover stories is` | `control-item accuracy in the described-accuracy condition, pooled over these two image-report cover stories is` |
| Appendix G, results and limits | `the 80\% described-accuracy criterion.` | `the 80\% control-item accuracy criterion for the described-accuracy condition.` |

### 21. Scheduled advisor accuracy and realized advisor accuracy need explicit labels at their point of contrast

**Type:** polysemy  
**Severity:** medium  

**Where:** Task and Methods

**Quote**
```latex
Within each 10-trial block A's realized accuracy equals the scheduled value, and each outcome label appears five times.
```

**Fix**
```latex
A block is a consecutive 10-trial segment within a phase. Within each block, A's realized advisor accuracy equals its scheduled advisor accuracy, and each outcome label appears five times.
```

**Where:** Table `tab:normsim`

**Quote**
```latex
Known current accuracy & 80.0 & 78.8 \\
```

**Fix**
```latex
Known current advisor accuracy & 80.0 & 78.8 \\
```

**Where:** Appendix D, count rules

**Quote**
```latex
Knowing the current accuracy is likewise not a strict ceiling
```

**Fix**
```latex
Knowing the current advisor accuracy is likewise not a strict ceiling
```

### 22. “Proportion” is used where the reported quantity is a percentage

**Type:** notation-spelling  
**Severity:** medium  

**Where:** Table `tab:behrens` caption

**Quote**
```latex
Accuracy of the Bayesian learner: proportion of trials ($t\ge 2$)
```

**Fix**
```latex
Prediction accuracy of the Bayesian volatility learner: percentage of trials ($t\ge 2$)
```

This replacement subsumes the benchmark-name replacement in issue 12 at this location.

**Where:** Table `tab:tests` caption

**Quote**
```latex
Agreement is the proportion of non-tied trials with the same choice as the rule;
```

**Fix**
```latex
Choice agreement is the percentage of non-tied trials with the same choice as the corresponding count rule;
```

**Where:** Appendix D, opening

**Quote**
```latex
Both score a rule by the proportion of trials ($t \ge 2$) on which the advisor it favours is correct; ties score 0.5.
```

**Fix**
```latex
Both report a rule's prediction accuracy as the percentage of trials ($t \ge 2$) on which its chosen advisor is correct; ties receive half credit (0.5).
```

### 23. Percentage choice agreement and Cohen’s `$\kappa$` are both called “agreement”

**Type:** polysemy  
**Severity:** medium  

**Where:** Task and Methods

**Quote**
```latex
compute agreement (Cohen's $\kappa$) with simple strategies.
```

**Fix**
```latex
compute chance-corrected strategy agreement (Cohen's $\kappa$) with simple strategies.
```

**Where:** Table `tab:tests` header

**Quote**
```latex
$p_{\mathrm{TOST,Holm}}$ & Choice agreement \\
```

**Fix**
```latex
$p_{\mathrm{TOST,Holm}}$ & Choice agreement (\%) \\
```

**Where:** Appendix F

**Quote**
```latex
Cohen's $\kappa$ adjusts agreement for chance.
```

**Fix**
```latex
Cohen's $\kappa$ adjusts observed agreement for the agreement expected from the two sets of marginal choice frequencies.
```

This also distinguishes chance-expected agreement from the task’s fixed 50% chance-level accuracy.

### 24. “Near chance” is operationally defined only in the appendix

**Type:** definition  
**Severity:** medium  
**Where:** Results, outcome-repetition paragraph

**Quote**
```latex
\Raw{} accuracy was near chance (50.7--52.0\%) in four models.
```

**Fix**
```latex
\Raw{} accuracy was near chance (defined as within 3 percentage points of 50\%; observed accuracy 50.7--52.0\%) in four models.
```

**Where:** Appendix C

**Quote**
```latex
\Raw{} accuracy is called near chance only within $50 \pm 3$\%.
```

**Fix**
```latex
\Raw{} accuracy is called near chance only within 3 percentage points of 50\%.
```

### 25. The outcome-quota strategy is used in the main text before it is defined there

**Type:** definition  
**Severity:** medium  
**Where:** Results, proprietary models

**Quote**
```latex
or with the outcome quota (.07--.10), a rule that would score 70.2\% on these trials;
```

**Fix**
```latex
or with the outcome-quota rule (.07--.10), which predicts the label with more occurrences remaining in the current block's quota and would score 70.2\% on these trials;
```

**Quote**
```latex
where the quota and the learner disagreed they followed the learner
```

**Fix**
```latex
where the outcome-quota rule and the Bayesian volatility learner disagreed they followed the Bayesian volatility learner
```

### 26. Advisor A’s identity is defined only in Appendix B

**Type:** definition  
**Severity:** medium  
**Where:** Task and Methods, advice-taking task

**Quote**
```latex
Advisor A is correct with probability 0.8 in a stable phase of 40 trials;
```

**Fix**
```latex
Advisor A, defined as the advisor favored on trial~1, is correct with probability 0.8 in a stable phase of 40 trials;
```

### 27. `$o_t$` is used before its binary coding is defined

**Type:** definition  
**Severity:** medium  
**Where:** Appendix D, Bayesian implementation

**Quote**
```latex
The predictive mean $q_t=E[p_t\mid o_{1:t-1}]$ determines the choice before observing correctness $o_t$;
```

**Fix**
```latex
Let $o_t=1$ if advisor A was correct on trial $t$ and $o_t=0$ otherwise. The predictive mean $q_t=E[p_t\mid o_{1:t-1}]$ determines the choice before observing advisor correctness $o_t$;
```

The later Appendix E and F definitions are compatible restatements, not conflicting definitions.

### 28. `$p$` is used in `$10p$` before being identified as scheduled advisor accuracy

**Type:** definition  
**Severity:** medium  
**Where:** Appendix B, stratification

**Quote**
```latex
Within each block, A is correct on exactly $10p$ trials ($8$ or $2$) in random order,
```

**Fix**
```latex
Within each block, let $p$ denote A's scheduled advisor accuracy. A is correct on exactly $10p$ trials ($8$ or $2$) in random order,
```

### 29. `$k$` denotes both a trial lag and meta-volatility

**Type:** notation-spelling  
**Severity:** medium  
**Where:** Appendix D, Bayesian implementation

Use `$\eta$` for meta-volatility, leaving `$k$` for integration-kernel lag.

| Exact quote | Exact replacement |
|---|---|
| `a static, unknown meta-volatility $k\in\{.05,.2,.5,1\}$` | `a static, unknown meta-volatility $\eta\in\{.05,.2,.5,1\}$` |
| `$\exp[-(v'-v)^2/(2k^2)]$` | `$\exp[-(v'-v)^2/(2\eta^2)]$` |
| `integrating over $v$ and $k$.` | `integrating over $v$ and $\eta$.` |

### 30. `$h$` denotes both the bandit’s environmental hazard and the BOCPD controller’s assumed hazard

**Type:** notation-spelling  
**Severity:** medium  

Reserve `$h$` for the bandit and use `$h_{\mathrm{CP}}$` for BOCPD.

**Where:** Appendix D, design simulation

**Quote**
```latex
compares BOCPD (posterior predictive, $h \in \{1/8, 1/15, 1/30\}$),
```

**Fix**
```latex
compares the BOCPD posterior-predictive rule (assumed change-point hazard $h_{\mathrm{CP}} \in \{1/8, 1/15, 1/30\}$),
```

Additional exact replacements, one occurrence each:

| Where | Exact find | Exact replacement |
|---|---|---|
| Appendix D | `$h=1/8$` | `$h_{\mathrm{CP}}=1/8$` |
| Appendix D | `$h=1/15$` | `$h_{\mathrm{CP}}=1/15$` |
| Appendix E | `a constant hazard rate $h = 1/15$.` | `a constant assumed change-point hazard $h_{\mathrm{CP}} = 1/15$.` |

### 31. “Volatility” names both reversal frequency and the Bayesian learner’s transition scale

**Type:** polysemy  
**Severity:** medium  
**Where:** Appendix D, grid definition

**Quote**
```latex
20 equally spaced log-volatility values $v\in[\log(.02),\log(3)]$,
```

**Fix**
```latex
20 equally spaced log-volatility values $v\in[\log(.02),\log(3)]$ (the log transition scale of advisor accuracy, distinct from a reversal hazard),
```

### 32. “Block” names both trial segments and groups of episodes

**Type:** polysemy  
**Severity:** medium  

| Where | Exact quote | Exact replacement |
|---|---|---|
| Appendix A, task details | `assigned in blocks so that cover story is crossed with phase order.` | `assigned in groups of episodes so that cover story is crossed with phase order.` |
| Appendix B, counterbalancing | `non-pet cover stories rotate in blocks of four episodes` | `non-pet cover stories rotate in groups of four episodes` |

The trial-block definition is supplied in issue 21. *Lower block* in a table caption is plainly a table region and need not be treated as a temporal block.

### 33. Schedule lengths and “dwell” lack consistent units or terminology

**Type:** notation-spelling  
**Severity:** medium  

**Where:** Appendix D, design simulation

**Quote**
```latex
phase lengths 40/40, 60/60 and 80/40; dwell 10 or 15 trials
```

**Fix**
```latex
phase lengths of 40/40, 60/60 and 80/40 trials; dwell times of 10 or 15 trials
```

**Quote**
```latex
For our setting (0.8; 40/40; dwell 10),
```

**Fix**
```latex
For our setting (advisor accuracy 0.8; phase lengths 40/40 trials; dwell time 10 trials),
```

**Where:** Appendix B, robustness-set generation

**Quote**
```latex
volatile-phase dwell times are drawn uniformly from 6--16 trials.
```

**Fix**
```latex
volatile-phase dwell times (numbers of trials between reversals) are drawn uniformly from 6--16 trials.
```

### 34. “Format” denotes history input, visual presentation, and output compliance

**Type:** polysemy  
**Severity:** medium  

| Where | Exact quote | Exact replacement |
|---|---|---|
| Appendix J, readout quality | `full-vocabulary probability mass on the two answer tokens and greedy format following.` | `full-vocabulary probability mass on the two answer tokens and greedy one-word output-format compliance.` |
| Appendix J, readout quality | `For four models nearly all mass fell on the answer words in every format.` | `For four models nearly all mass fell on the answer words in every history format.` |
| Table `tab:readout` | `& \multicolumn{3}{c}{Candidate mass} & Format \\` | `& \multicolumn{3}{c}{Candidate mass} & Output-format \\` |
| Table `tab:readout` | `Model & Mean & Min & Share $<0.5$ & following \\` | `Model & Mean & Min & Share $<0.5$ & compliance \\` |
| Table `tab:readout` caption | `format following is the proportion of greedy outputs equal to an answer word.` | `one-word output-format compliance is the proportion of greedy outputs equal to an answer word.` |
| Appendix H heading | `\paragraph{Image and chart formats.}` | `\paragraph{Image and chart payoff presentations.}` |
| Table `tab:apibeh` header | `Model & History & Acc. & Repeat` | `Model & History format & Acc. & Repeat` |

### 35. “Judgment readout” names a different question, not an extraction method

**Type:** polysemy  
**Severity:** medium  
**Where:** Appendix A, after Example 1

**Quote**
```latex
In the judgment readout the current reports and question are replaced by
```

**Fix**
```latex
For the advisor-judgment question, the current reports and prediction question are replaced by
```

**Where:** Appendix J

**Quote**
```latex
\paragraph{Readout.} Models use bfloat16 with a float32 output projection.
```

**Fix**
```latex
\paragraph{Open-model readout.} The open models use bfloat16 with a float32 output projection.
```

### 36. The bandit readout describes a probability “between” two letters without identifying what the probabilities attach to

**Type:** definition  
**Severity:** medium  
**Where:** Appendix H, replay

**Quote**
```latex
The readout is the choice probability between the two letters from the first-token logits, as in the main task.
```

**Fix**
```latex
The readout consists of the choice probabilities assigned to the two machine-answer letters, obtained by normalizing their first-token logits as in the main task.
```

### 37. “Model-free” denotes both a reinforcement-learning model class and a regression that avoids fitting that class

**Type:** polysemy  
**Severity:** medium  
**Where:** Appendix H paragraph heading

**Quote**
```latex
\paragraph{Model-free check of forgetting (exploratory).}
```

**Fix**
```latex
\paragraph{Regression check of forgetting (exploratory).}
```

Keep *myopic model-free learner* for the fitted learning model.

### 38. The bandit’s stated-hazard condition is also called “told”

**Type:** synonymy  
**Severity:** medium  

| Where | Exact quote | Exact replacement |
|---|---|---|
| Appendix H, replay | `the original instructions state it (told).` | `the original instructions state it (stated).` |
| Appendix H, fits | `with hidden and told conditions shown in Fig.~\ref{fig:human}.` | `with hidden- and stated-hazard conditions shown in Fig.~\ref{fig:human}.` |
| Table `tab:stale` header | `Agent (hazard hidden) & $h=0.1$` | `Agent (hazard hidden for models) & $h=0.1$` |
| Table `tab:stale` row label | `Participants (told) &` | `Participants (hazard stated) &` |

Ordinary prose such as “participants were told `$h$`” can remain unchanged.

### 39. A bandit game is called a model episode

**Type:** synonymy  
**Severity:** medium  
**Where:** Appendix H, calibration

**Quote**
```latex
20 sampled choice sequences per model episode
```

**Fix**
```latex
20 sampled choice sequences per model game
```

### 40. “Kernel regression” can be read as a different ML method

**Type:** polysemy  
**Severity:** medium  

**Where:** Appendix C and Table `tab:recovery` caption  
**All occurrences: 2**

**Find**
```latex
kernel regression
```

**Replace**
```latex
integration-kernel logistic regression
```

**Where:** Results, outcome repetition

**Quote**
```latex
The 15-lag kernel describes local sensitivity to recent trials
```

**Fix**
```latex
The 15-lag correctness integration kernel describes local sensitivity to recent trials
```

**Quote**
```latex
correctness kernel in Fig.~\ref{fig:kernel}
```

**Fix**
```latex
correctness integration kernel in Fig.~\ref{fig:kernel}
```

### 41. “Weight on the previous outcome” can mean an outcome-value coefficient rather than the outcome-repetition regressor

**Type:** synonymy  
**Severity:** medium  

**Where:** Results, Appendix I, Table `tab:wording` caption, and Appendix J  
**All occurrences of this exact phrase: 9**

**Find**
```latex
weight on the previous outcome
```

**Replace**
```latex
weight on the outcome-repetition regressor
```

**Where:** Appendix I, held-out masking

**Quote**
```latex
repetition weights (Table~\ref{tab:mask})
```

**Fix**
```latex
outcome-repetition coefficients (Table~\ref{tab:mask})
```

### 42. “Copy” is ambiguous between outcome repetition and copying an advisor

**Type:** synonymy  
**Severity:** medium  
**Where:** Table `tab:recovery`

**Quote**
```latex
50\% copy, 50\% delta
```

**Fix**
```latex
50\% outcome repetition, 50\% delta
```

For the same table, standardize the strategy label:

**All occurrences: 3**

**Find**
```latex
Repeat last outcome
```

**Replace**
```latex
Repeat previous outcome
```

### 43. The described-accuracy control has a second name in its table

**Type:** synonymy  
**Severity:** medium  
**Where:** Table `tab:controls_full`  
**All occurrences: 4**

**Find**
```latex
Described \%
```

**Replace**
```latex
Described accuracy
```

Also clarify the abbreviated control-task labels:

| Exact quote | Exact replacement |
|---|---|
| `Single trial & Text &` | `Single-trial correctness check & Text &` |
| `Composite read & Image &` | `Composite-panel reading & Image &` |
| `Chart read & Chart &` | `Correctness-chart reading & Chart &` |

### 44. The recovery formula uses an undefined accuracy abbreviation

**Type:** definition  
**Severity:** medium  
**Where:** Appendix C, statistical analysis

**Quote**
```latex
the position of \CorrectnessOnly{} between \Raw{} and \Summary{},
```

**Fix**
```latex
the position of \CorrectnessOnly{} between \Raw{} and \Summary{}, with $\mathrm{acc}$ denoting prediction accuracy,
```

This defines the notation immediately before the first displayed-in-text recovery ratio.

### 45. The same `$w$` labels generating parameters and fitted coefficients

**Type:** notation-spelling  
**Severity:** medium  
**Where:** Table `tab:recovery` caption

**Quote**
```latex
Best strategy: highest Cohen's $\kappa$; Repeat $w$ and Lag-1 $w$:
```

**Fix**
```latex
Best strategy: highest Cohen's $\kappa$. In synthetic-agent labels, $w$ is the generating log-odds weight; in the column headings it is a fitted coefficient. Repeat $w$ and Lag-1 $w$:
```

### 46. Trial/day equivalence is explained only after the main-text prompt examples

**Type:** definition  
**Severity:** medium  
**Where:** Task and Methods, history formats

**Quote**
```latex
Before trial $t$, the model sees trials $1,\dots,t-1$ in one of four formats (Fig.~\ref{fig:teaser}; full prompts in Appendix~A).
```

**Fix**
```latex
Before trial $t$, the model sees trials $1,\dots,t-1$ in one of four history formats (Fig.~\ref{fig:teaser}; full prompts in Appendix~A). Prompt labels follow the cover story; for example, weather-task trials are called days.
```

Do not change *Day* to *Trial* inside the verbatim weather prompts.

## Low-severity cosmetic issues

### 47. Model-name spelling and abbreviation conventions vary

**Type:** notation-spelling  
**Severity:** low  

| Where | Exact quote | Exact replacement |
|---|---|---|
| Task and Methods | `Qwen3-VL 8B and 32B` | `Qwen3-VL-8B and Qwen3-VL-32B` |
| Task and Methods | `InternVL3.5 8B and 38B` | `InternVL3.5-8B and InternVL3.5-38B` |
| Table `tab:controls_full` caption | `Qwen3: Qwen3-VL; IVL: InternVL3.5.` | `Qwen3: Qwen3-VL; IVL: InternVL3.5; Gemma: Gemma-3.` |
| Results, proprietary models | `77\% of Fable's \Raw{} answers` | `77\% of Claude Fable 5.2's \Raw{} answers` |
| Appendix J, image preprocessing | `for Qwen models` | `for Qwen3-VL models` |
| Appendix J, image preprocessing | `For Gemma-3,` | `For Gemma-3-27B,` |
| Table `tab:readout` caption | `Gemma's failures` | `Gemma-3-27B's failures` |

**Where:** Task and Methods, first model list

**Quote**
```latex
Gemma-3-27B \citep{gemmateam2025gemma3}.
```

**Fix**
```latex
Gemma-3-27B \citep{gemmateam2025gemma3}. B in model names denotes billions of parameters.
```

The existing definition of *B* is otherwise confined to Appendix J.

### 48. British and American spelling are mixed

**Type:** notation-spelling  
**Severity:** low  

Apply these as **case-sensitive whole-word replacements**, not substring replacements. Counts refer to rendered prose and captions.

| Where | Find | Replace | Count |
|---|---|---|---:|
| Introduction | `analysed` | `analyzed` | 1 |
| Task and Methods | `modelling` | `modeling` | 1 |
| Figures `fig:kernel`, `fig:curves` | `colours` | `colors` | 2 |
| Figure `fig:strategies` | `colour` | `color` | 2 |
| Figure `fig:strategies` | `centred` | `centered` | 1 |
| Results, image reports | `labelled` | `labeled` | 1 |
| Discussion; Appendix J deviations | `behavioural` | `behavioral` | 2 |
| Conclusion | `behaviour` | `behavior` | 1 |
| Appendix J caption and paragraph heading | `Behaviour` | `Behavior` | 2 |
| Appendix B caption; Appendix D opening | `favours` | `favors` | 2 |
| Appendix G | `grey` | `gray` | 2 |
| Appendix G | `Coloured` | `Colored` | 1 |
| Appendix J deviations | `unanalysed` | `unanalyzed` | 1 |

The Appendix D sentence containing one occurrence of *favours* is already replaced in issue 22.

### 49. Minor terminology and abbreviation cleanup

**Type:** notation-spelling  
**Severity:** low  

| Where | Exact quote | Exact replacement | Count |
|---|---|---|---:|
| Appendix E, variant list | `the last inferred change-point` | `the last inferred change point` | 1 |
| Appendix F, recovery | `logit noise s.d.\ 0.3` | `logit-noise standard deviation 0.3` | 1 |
| Appendix H, fits | `re-analysis` | `reanalysis` | 2 |
| Appendix I, prompt wording | `We re-ran the four text history formats` | `We reran the four text history formats` | 1 |

BOCPD, TOST, WSLS, `$\kappa$`, `$\phi$`, and `$y_t$` otherwise have compatible definitions before their relevant technical uses; the source does not require additional terminology corrections merely because those definitions are repeated in an abstract, caption, or appendix.