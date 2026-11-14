### Recommended canonical terms
- **Advisor accuracy** → an advisor’s scheduled or realized correctness; **prediction accuracy** → accuracy of model or rule choices; **control-item accuracy** → performance on controls.
- **Episode pair** → an episode and its counterbalanced copy; **held-out set / robustness set / development episodes** → the three evaluation resources.
- **\Raw{}, \RawCorrectness{}, \CorrectnessOnly{}, \Summary{}** → VLM input conditions; **\Summary{} variant** → a count-based input using a specified alternative trial range.
- **All-trial / last-five / change-point / reversal-aligned count rule** → choices made without a VLM; **BOCPD** → detection method, not a choice rule.
- **Bayesian volatility learner** → the specific computational baseline; **Bayesian learner / learner** → subsequent references within the same paragraph.
- **Choice probability \(y_t\)** → probability for advisor A; **following probability** → probability for the currently better advisor; **discrete-choice following rate** → percentage of choices following that advisor.
- **Outcome label** → the displayed feedback word; **answer words** → permitted model outputs; **advisor correctness** → whether an advisor’s report matched the outcome.
- **Integration kernel** → lagged correctness weights; **\Summary{}$-$\Raw{} gap** → prediction-accuracy difference; **points** → percentage points.

## Main text
Issues are ranked by severity within this section. Literal prompt quotations are not terminology to standardize.

### High severity

1. **Type:** polysemy  
   **Severity:** high  
   **Where:** Results, “Following the count rule…”—equivalence of *accuracy* is described as equivalence of models or rules, immediately beside choice-agreement results.

   **Quote:** `four of five were equivalent to the all-trial count rule alone within $\pm$2 points`  
   **Fix:** `four of five had prediction accuracy equivalent to that of the all-trial count rule within $\pm$2 points`

   **Quote:** `equivalent to the corresponding count rule in five of five models for the change-point variant and three of five for the last-five variant`  
   **Fix:** `prediction accuracy equivalent to that of the corresponding count rule in five of five models for the change-point variant and three of five for the last-five variant`

2. **Type:** condition-name  
   **Severity:** high  
   **Where:** Results, “Following the count rule…”—an input variant is described as something the model follows, conflating the input with its corresponding count rule.

   **Quote:** `\Summary{} variants covering the last five trials or the run since a change point inferred by BOCPD were followed as faithfully`  
   **Fix:** `Given the last-five and change-point \Summary{} variants, models showed similarly high choice agreement with the corresponding count rules`

   The paragraph’s closing reference to “the rule” should also identify which rule:

   **Quote:** `set by the rule outside the model.`  
   **Fix:** `set by the corresponding count rule outside the model.`

### Medium severity

3. **Type:** synonymy  
   **Severity:** medium  
   **Where:** Abstract and Results—evaluation results alternate between “episodes,” “episode pairs,” and the held-out set. Retain “episodes” for generation descriptions; use the analytical unit or set name for results.

   - **Abstract quote:** `On 48 held-out episodes,`  
     **Fix:** `On 48 held-out episode pairs (each episode and its counterbalanced copy),`
   - **Results, “\CorrectnessOnly{} recovers…” quote:** `On held-out episodes, accuracy was`  
     **Fix:** `On the held-out set, accuracy was`
   - **Results, “Under \Raw{}…” quote:** `44.5\% on held-out episodes`  
     **Fix:** `44.5\% on the held-out set`
   - **Results, masking quote:** `On held-out episodes, all predictions`  
     **Fix:** `On the held-out set, all predictions`
   - **Results, computational reference quote:** `run on the same held-out episodes`  
     **Fix:** `run on the same held-out set`
   - **Results, image reports quote:** `On held-out episodes much of the cost`  
     **Fix:** `On the held-out set, much of the cost`

4. **Type:** definition  
   **Severity:** medium  
   **Where:** Teaser caption and Methods, “Advice-taking task”—“answer words swapped” is less precise than Appendix B’s “answer-word mapping reversed”; “outcomes are identical” does not distinguish underlying outcomes from their labels.

   **Quote:** `An episode pair comprises an episode and its counterbalanced copy, with advisor names and answer words swapped.`  
   **Fix:** `An episode pair comprises an episode and its counterbalanced copy, with advisor names swapped and the answer-word mapping reversed.`

   **Quote:** `Every episode has a counterbalanced copy in which advisor names and answer words are swapped while outcomes are identical; averaging over the two cancels name and label biases.`  
   **Fix:** `Every episode has a counterbalanced copy in which advisor names are swapped and the answer-word mapping is reversed while the underlying outcome sequence is unchanged; averaging over the two copies cancels advisor-name and answer-word biases.`

5. **Type:** polysemy  
   **Severity:** medium  
   **Where:** Abstract, Introduction, and Methods—bare “accuracy” initially names both advisor reliability and prediction performance. Establish the distinction explicitly, then allow “accuracy” as the stated shorthand for prediction accuracy.

   - **Abstract quote:** `\CorrectnessOnly{} raised accuracy by 8--16 percentage points`  
     **Fix:** `\CorrectnessOnly{} raised prediction accuracy by 8--16 percentage points`
   - **Introduction quote:** `learn a changing, non-stationary accuracy by in-context learning`  
     **Fix:** `learn changing, non-stationary advisor accuracy by in-context learning`
   - **Methods, “Measures and analysis” quote:** `For the open models, accuracy is the percentage of trials`  
     **Fix:** `For the open models, prediction accuracy (hereafter accuracy) is the percentage of trials`

6. **Type:** polysemy  
   **Severity:** medium  
   **Where:** Results, control and image-report paragraphs—“correct advisor” can mean either the advisor correct on an observed trial or the currently better advisor; image-control accuracy is subsequently called simply “Accuracy.”

   **Quote:** `every model named the correct advisor on a single shown trial`  
   **Fix:** `every model identified which advisor was correct on a single shown trial`

   **Quote:** `Accuracy was mostly higher when the name was printed on the image`  
   **Fix:** `Control-item accuracy was mostly higher when the name was printed on the image`

   **Quote:** `Models could not read single cells of the correctness chart reliably (61--71\%), so chart histories are not interpreted.`  
   **Fix:** `Models could not read single cells of the correctness chart reliably (cell-reading accuracy: 61--71\%), so chart histories are not interpreted.`

7. **Type:** definition  
   **Severity:** medium  
   **Where:** Methods, “Measures and analysis”—the definition of *following probability* is present, but the shorthand *currently better advisor*, later used in the main text, is explicitly defined only in an appendix caption.

   **Quote:** `The \emph{following probability} is the choice probability for the advisor with the higher scheduled accuracy on that trial.`  
   **Fix:** `The \emph{currently better advisor} is the advisor with the higher scheduled accuracy on that trial; the \emph{following probability} is the choice probability for that advisor.`

8. **Type:** definition  
   **Severity:** medium  
   **Where:** Methods, “Measures and analysis”—*choice agreement* is used in the main Results but defined only in Appendix C. Its distinction from Cohen’s \(\kappa\) should be available in the main text.

   **Quote:** `and compute chance-corrected agreement (Cohen's $\kappa$) with simple strategies.`  
   **Fix:** `and compute chance-corrected agreement (Cohen's $\kappa$) with simple strategies. Separately, \emph{choice agreement} is the percentage of non-tied trials on which the model and the corresponding count rule make the same choice.`

9. **Type:** notation-spelling  
   **Severity:** medium  
   **Where:** Results, proprietary reasoning models—the post-reversal result uses discrete-choice percentages, whereas the surrounding comparisons use following probabilities. The current repeated “trials” wording also obscures that distinction.

   **Quote:** `after a reversal they followed the currently better advisor on 6--9\% of trials in the first three trials of the reversed block, rising to 94--99\% in trials 8--10.`  
   **Fix:** `after a reversal, the percentage of discrete choices following the currently better advisor was 6--9\% in the first three trials of the reversed block, rising to 94--99\% in trials 8--10.`

10. **Type:** polysemy  
    **Severity:** medium  
    **Where:** Abstract, Introduction, Results, and Discussion—“learner” sometimes denotes the computational agent and sometimes stands for its accuracy. These substitutions also remove the awkward “accuracy to a learner” construction.

    - **Abstract quote:** `had similar observed \Raw{} accuracy to a Bayesian volatility learner (63\%)`  
      **Fix:** `had observed \Raw{} prediction accuracy similar to that of a Bayesian volatility learner (63\%)`
    - **Introduction contribution quote:** `have similar observed \Raw{} accuracy to a Bayesian volatility learner`  
      **Fix:** `have observed \Raw{} prediction accuracy similar to that of a Bayesian volatility learner`
    - **Results, computational reference quote:** `reached the learner (66.6--69.1\%; Appendix~E)`  
      **Fix:** `reached the learner's prediction accuracy (66.6--69.1\%; Appendix~E)`
    - **Discussion quote:** `had similar observed \Raw{} accuracy to the Bayesian volatility learner`  
      **Fix:** `had observed \Raw{} prediction accuracy similar to that of the Bayesian volatility learner`

11. **Type:** polysemy  
    **Severity:** medium  
    **Where:** Results, “Humans and open models in the changing bandit”—the unqualified normative learner is a different agent from the advice-taking task’s Bayesian volatility learner.

    **Quote:** `a normative learner forgets the unchosen machine faster`  
    **Fix:** `a normative bandit learner forgets the unchosen machine faster`

12. **Type:** polysemy  
    **Severity:** medium  
    **Where:** Abstract, teaser, Methods, Results, and Discussion—use *outcome label* when referring to the visible history field, not the underlying event.

    - **Abstract quote:** `\Raw{} (reports and outcomes)`  
      **Fix:** `\Raw{} (reports and outcome labels)`
    - **Teaser caption quote:** `\Raw{} lists reports and outcomes;`  
      **Fix:** `\Raw{} lists reports and outcome labels;`
    - **Methods, “History formats” quote:** `\Raw{} lists both reports and the outcome`  
      **Fix:** `\Raw{} lists both reports and the outcome label`
    - **Results, “\CorrectnessOnly{} recovers…” quote:** `The reports and outcomes therefore do more than add work:`  
      **Fix:** `The reports and outcome labels therefore do more than add work:`
    - **Discussion quote:** `the reports and outcomes in \Raw{}`  
      **Fix:** `the reports and outcome labels in \Raw{}`

    **Where:** Main masking table, column-group headings.

    **Quote:** `\multicolumn{2}{c}{Outcome shown} & \multicolumn{2}{c}{Outcome masked}`  
    **Fix:** `\multicolumn{2}{c}{Outcome label shown} & \multicolumn{2}{c}{Outcome label masked}`

13. **Type:** polysemy  
    **Severity:** medium  
    **Where:** Integration-kernel figure caption and Results, “Under \Raw{}…”—“past correctness” does not identify whose correctness supplies the regressor.

    **All occurrences in the main text: 2.**

    **Find:** `past correctness`  
    **Replace:** `past advisor correctness`

    The two source contexts are `decaying weights on past correctness` and `integration-kernel weights on past correctness were near zero`.

14. **Type:** definition  
    **Severity:** medium  
    **Where:** Methods, “Models and readout”—“B” is subsequently used independently in “the two 8B models” and “up to 38B,” but its meaning is defined only in Appendix J.

    **Quote:** `and Gemma-3-27B \citep{gemmateam2025gemma3}.`  
    **Fix:** `and Gemma-3-27B \citep{gemmateam2025gemma3}. In model names, B denotes billions of parameters.`

### Low-severity consistency items

15. **Type:** synonymy  
    **Severity:** low  
    **Where:** Main-text caption and Results—shortened technical names are understandable but depart from the chosen canonical names.

    - **Integration-kernel caption quote:** `Per-model kernels and CIs`  
      **Fix:** `Per-model integration kernels and CIs`
    - **Results, “Under \Raw{}…” quote:** `Under \CorrectnessOnly{}, the kernel instead peaked`  
      **Fix:** `Under \CorrectnessOnly{}, the integration kernel instead peaked`
    - **Results, proprietary models quote:** `where the quota rule and the Bayesian learner disagreed`  
      **Fix:** `where the outcome-quota rule and the Bayesian learner disagreed`

16. **Type:** notation-spelling  
    **Severity:** low  
    **Where:** Teaser caption and Introduction opening paragraph—American-style consistency.

    **All occurrences: 2.**

    **Find:** `afterwards`  
    **Replace:** `afterward`

## Appendix issues
High and medium severity only; ranked within this section.

### High severity

17. **Type:** condition-name  
    **Severity:** high  
    **Where:** Appendix I, “Prompt wording”—\Summary{} is used as though it were the decision rule.

    **Quote:** `InternVL3.5-8B followed \Summary{} on only 72\% of non-tied trials (98\% with the original wording).`  
    **Fix:** `InternVL3.5-8B showed only 72\% choice agreement with the all-trial count rule when given \Summary{} (98\% with the original wording).`

18. **Type:** polysemy  
    **Severity:** high  
    **Where:** Appendix F, strategy figure caption—“BOCPD” labels the change-point count rule, contrary to the convention reserving it for the detection method.

    **Quote:** `Last-5 rule and Count rule: the last-five and all-trial count rules; BOCPD: the change-point count rule (not the BOCPD posterior-predictive rule);`  
    **Fix:** `Last-five count rule and All-trial count rule: count rules over the last five trials and all past trials, respectively; Change-point count rule: the count rule over the run inferred by BOCPD;`

    The corresponding artwork labels identified by this caption must also be changed: `Last-5 rule` → `Last-five count rule`; `Count rule` → `All-trial count rule`; `BOCPD` → `Change-point count rule`. The supplied LaTeX does not contain the artwork’s editable labels.

### Medium severity

19. **Type:** definition  
    **Severity:** medium  
    **Where:** Appendices A and B, counterbalancing descriptions—apply the same outcome/mapping distinction as in main-text issue 4.

    **Appendix A quote:** `In the counterbalanced copy, advisor names, answer words and the order in which the instructions introduce the advisors are swapped while outcomes and images are identical;`  
    **Fix:** `In the counterbalanced copy, advisor names are swapped, the answer-word mapping and advisor-introduction order are reversed, and the underlying outcome sequence and images are unchanged;`

    **Appendix B quote:** `outcomes and images are identical.`  
    **Fix:** `the underlying outcome sequence and images are unchanged.`

20. **Type:** synonymy  
    **Severity:** medium  
    **Where:** Appendix D, design simulation and rule comparison—several names depart from the canonical rule names.

    **All occurrences of this phrase in Appendix D: 2.**

    **Find:** `count rule since the last true reversal`  
    **Replace:** `reversal-aligned count rule`

    **Quote:** `We therefore do not claim that a change-point rule is optimal`  
    **Fix:** `We therefore do not claim that the change-point count rule is optimal`

    **Table quote:** `Outcome quota & 65.5 & -- \\`  
    **Fix:** `Outcome-quota rule & 65.5 & -- \\`

21. **Type:** synonymy  
    **Severity:** medium  
    **Where:** Appendix E, variants table—“Since reversal” is another name for the reversal-aligned variant and also resembles the post-reversal timing heading in Appendix J.

    **Quote:** `Model & Last 5 & Last 10 & Last 20 & Change-point & Since reversal & Last 10 & Change-point \\`  
    **Fix:** `Model & Last 5 & Last 10 & Last 20 & Change-point & Reversal-aligned & Last 10 & Change-point \\`

    **Caption quote:** `Since reversal is the reversal-aligned \Summary{} variant, a hidden-information diagnostic`  
    **Fix:** `The reversal-aligned \Summary{} variant is a hidden-information diagnostic`

22. **Type:** polysemy  
    **Severity:** medium  
    **Where:** Appendix E, “What the controller is”—“rule” alternately means the trial-range selector and the rule that chooses an advisor.

    **Quote:** `The controller is a training-free rule outside the model;`  
    **Fix:** `The change-point controller is a training-free trial-range selector outside the model;`

    **Quote:** `but we use it as an integration rule, not as a model of the VLM or of the brain.`  
    **Fix:** `but we use it to select the trial range for correctness counts, not as a model of the VLM or of the brain.`

23. **Type:** polysemy  
    **Severity:** medium  
    **Where:** Appendix D, Bayesian learner implementation—unqualified “accuracy” denotes the latent advisor property rather than prediction performance.

    **Quote:** `using discrete Gaussian transition weights in logit accuracy; the original formulation uses a beta transition in accuracy.`  
    **Fix:** `using discrete Gaussian transition weights in logit advisor accuracy; the original formulation uses a beta transition in advisor accuracy.`

    **Quote:** `The grid has 60 equally spaced accuracy values`  
    **Fix:** `The grid has 60 equally spaced advisor-accuracy values`

24. **Type:** polysemy  
    **Severity:** medium  
    **Where:** Appendix G, “Results and limits”—“described-accuracy criterion” can be mistaken for a criterion on the stated advisor accuracy rather than on model performance.

    **Quote:** `below the 80\% described-accuracy criterion.`  
    **Fix:** `below the 80\% control-item accuracy criterion for the described-accuracy control.`

25. **Type:** notation-spelling  
    **Severity:** medium  
    **Where:** Appendix F, synthetic-agent table and caption—the same \(w\) denotes generating weights and fitted coefficients. The caption explains the distinction, but distinct symbols would remove the overload.

    **All occurrences of this expression in the table: 2.**

    **Find:** `$w=3$`  
    **Replace:** `$w_{\mathrm{gen}}=3$`

    **Header quote:** `Repeat $w$ & Lag-1 $w$`  
    **Fix:** `Repeat $\hat{w}$ & Lag-1 $\hat{w}$`

    **Caption quote:** `In agent labels $w$ is the generating weight; in column headings it is the fitted coefficient. Repeat $w$ and Lag-1 $w$:`  
    **Fix:** `In agent labels $w_{\mathrm{gen}}$ is the generating weight; in column headings $\hat{w}$ is the fitted coefficient. Repeat $\hat{w}$ and Lag-1 $\hat{w}$:`

26. **Type:** notation-spelling  
    **Severity:** medium  
    **Where:** Appendix H, calibration paragraph—label the standard error in percentage points, distinguishing it from a relative percentage.

    **Quote:** `Monte Carlo standard error 1.7\%`  
    **Fix:** `Monte Carlo standard error 1.7 percentage points`

27. **Type:** polysemy  
    **Severity:** medium  
    **Where:** Appendix H, image and chart payoff presentations—bare “points” denotes payoffs here, whereas the main-text convention defines “points” as percentage points.

    **All occurrences of this phrase: 2.**

    **Find:** `which gave more points`  
    **Replace:** `which gave a higher payoff`

    The already qualified expression `payoff points` elsewhere can remain.

28. **Type:** polysemy  
    **Severity:** medium  
    **Where:** Appendix I, masking table—same visible-field terminology as main-text issue 12.

    **Quote:** `Model & \RawCorrectness{} & Reports masked & Outcome masked & Both masked & \CorrectnessOnly{} \\`  
    **Fix:** `Model & \RawCorrectness{} & Reports masked & Outcome label masked & Both masked & \CorrectnessOnly{} \\`

29. **Type:** definition  
    **Severity:** medium  
    **Where:** Appendix J, “Behavior of the proprietary models”—the paragraph’s first reference to the baseline uses a shortened compound rather than its required full name.

    **Quote:** `Bayesian-learner--quota disagreement trials`  
    **Fix:** `trials on which the Bayesian volatility learner and the outcome-quota rule disagree`

30. **Type:** polysemy  
    **Severity:** medium  
    **Where:** Appendix J, “Behavior of the proprietary models”—same advisor/model correctness ambiguity as main-text issue 13.

    **All occurrences in this paragraph: 2.**

    **Find:** `past correctness`  
    **Replace:** `past advisor correctness`

31. **Type:** notation-spelling  
    **Severity:** medium  
    **Where:** Appendix C, control-results table header—“Gemma-27B” drops the model-generation identifier; unlike Qwen3 and IVL, this shorthand is not defined in the caption.

    **Quote:** `Gemma-27B & IVL-38B`  
    **Fix:** `Gemma-3-27B & IVL-38B`