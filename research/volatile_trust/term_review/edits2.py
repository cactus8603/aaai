# Round 2 (Claude Fable 5.2) edits; numbers refer to Fable's issue list.
E = [
# 1
(r"The robustness set has 24 held-out episodes (48 with copies;", r"The robustness set has 24 episodes (48 with copies;", None),
# 2
(r"BOCPD: the change-point count rule;", r"BOCPD: the change-point count rule (not the BOCPD posterior-predictive rule);", None),
(r"the run since the last inferred change point (BOCPD);", r"the run since the last change point inferred by BOCPD (change-point variant);", None),
(r"to the BOCPD run, or to the last 10 trials", r"to the change-point run, or to the last 10 trials", None),
(r"the best BOCPD minus the best fixed window ranged from", r"the best BOCPD posterior-predictive rule minus the best last-$K$ count rule ranged from", None),
(r"accuracy was 68.7\% for the best BOCPD ($h_{\mathrm{CP}}=1/8$)", r"accuracy was 68.7\% for the best BOCPD posterior-predictive rule ($h_{\mathrm{CP}}=1/8$)", None),
# 3
(r"(Gemma-3-27B: $-1.4$ points, $p_\text{Holm}=.053$;", r"(Gemma-3-27B: $-1.4$ points, Holm-corrected equivalence-test $p=.053$;", None),
# 4
(r"\shortstack{Count\\rule} & \shortstack{Last-5\\rule}", r"\shortstack{All-trial\\count rule} & \shortstack{Last-five\\count rule}", None),
(r"Last-5 rule and Count rule: counts over the last five and all past trials", r"Last-5 rule and Count rule: the last-five and all-trial count rules", None),
# 5
(r"majority counts over the last $K \in \{3,5,7,10,15,20,30\}$ trials, counts over all past trials, and counts since the last true reversal",
 r"the last-$K$ count rule ($K \in \{3,5,7,10,15,20,30\}$), the all-trial count rule, and the count rule since the last true reversal", None),
(r"69.0\% for the best window ($K=3$), 63.2\% for counts over all trials and 73.9\% for counts since the last reversal",
 r"69.0\% for the best last-$K$ count rule ($K=3$), 63.2\% for the all-trial count rule and 73.9\% for the count rule since the last true reversal", None),
(r"Count since true reversal & 74.4", r"Count rule (reversal-aligned) & 74.4", None),
(r"whose choices also agreed weakly with recent-trial count rules", r"whose choices also agreed weakly with the last-five count rule", None),
(r"the all-trial and recent-trial count rules, the change-point count rule", r"the all-trial and last-five count rules, the change-point count rule", None),
# 6
(r"Model & \Summary{} variant & Model $-$ count rule [90\% CI]", r"Model & Count range & Model $-$ count rule [90\% CI]", None),
(r"with Holm correction over the three \Summary{} variants per model", r"with Holm correction over \Summary{} (all trials) and its two variants per model", None),
(r"The variants are: all past trials (\Summary{}); the last", r"The ranges are: all past trials (\Summary{} itself); and, as variants, the last", None),
# 7
(r"Gemma-3-27B's failures concentrate in \Summary{} variants and chart histories", r"Gemma-3-27B's failures concentrate in \Summary{}, its variants and correctness-chart histories", None),
# 8
(r"smaller than on the main set but positive", r"smaller than on the held-out set but positive", None),
# 9
(r"last observed values (per 10 points)", r"last observed values (per 10 payoff points)", None),
(r"log-odds per 10 points of last-observed value difference", r"log-odds per 10 payoff points of last-observed value difference", None),
(r"Points show equal-weight mean accuracies", r"Markers show equal-weight mean accuracies", None),
(r"Points show accuracy changes in percentage points", r"Markers show accuracy changes in percentage points", None),
# 10
(r"gap was $+8.4$ to $+16.4$ percentage points and significant", r"gap was $+8.4$ to $+16.4$ points and significant", None),
(r"by only 1.1--5.4 percentage points over \Raw{}", r"by only 1.1--5.4 points over \Raw{}", None),
# 11
(r"narrowed the gap in two open models", r"narrowed the \Summary{}$-$\Raw{} gap in two open models", None),
(r"\CorrectnessOnly{} recovers most of the gap;", r"\CorrectnessOnly{} recovers most of the \Summary{}$-$\Raw{} gap;", None),
(r"\CorrectnessOnly{} removed most of the gap, \RawCorrectness{} helped little", r"\CorrectnessOnly{} recovered most of the \Summary{}$-$\Raw{} gap, \RawCorrectness{} helped little", None),
(r"\CorrectnessOnly{} removes most of the", r"\CorrectnessOnly{} recovers most of the", None),
(r"the fraction of the main-task gap recovered", r"the fraction of the \Summary{}$-$\Raw{} gap recovered", None),
# 12
(r"we therefore repeat the primary test on episodes with independent outcomes, where the gap was about half as large",
 r"we therefore repeat the primary test on the robustness set (independent outcomes), where the \Summary{}$-$\Raw{} gap was about half as large", None),
# 13
(r"\subsection{I. Robustness, Prompt Format and Masking}", r"\subsection{I. Robustness, Prompt Wording and Masking}", None),
(r"stayed below the count rule in every layout (development episodes; Appendix~I)", r"stayed below the count rule in every wording (development episodes; Appendix~I)", None),
(r"The size of the gap depends on presentation:", r"The size of the \Summary{}$-$\Raw{} gap depends on the prompt wording:", None),
(r"These exploratory formats are interpreted", r"These exploratory presentations are interpreted", None),
# 14
(r"this history remained below the last-five count rule", r"\CorrectnessOnly{} remained below the last-five count rule", None),
(r"history representation strongly affected", r"history format strongly affected", None),
(r"we test which representations support good choices", r"we test which history formats support good choices", None),
(r"pooled over main-task histories and report types", r"pooled over history formats and report types", None),
(r"Model & History & Acc.", r"Model & History format & Acc.", None),
(r"the natural-language layout used in all main conditions", r"the natural-language layout used in all four history formats", None),
# 15
(r"Qwen3-VL-8B with a long \Raw{} (53.3\%", r"Qwen3-VL-8B with a long easy history in \Raw{} (53.3\%", None),
(r"a short \Raw{}, a short \CorrectnessOnly{}, and a long \CorrectnessOnly{}", r"a short easy history in \Raw{}, and short and long easy histories in \CorrectnessOnly{}", None),
(r"at or below full \Raw{} (46.8--54.9\%)", r"at or below the full \Raw{} history (46.8--54.9\%)", None),
# 16
(r"to a Bayesian learner, so the format effect", r"to a Bayesian volatility learner, so the history-format effect", None),
(r"Bayesian learner & \multicolumn{4}{c}{66.8 / 62.9}", r"Bayesian volatility learner & \multicolumn{4}{c}{66.8 / 62.9}", None),
(r"the Bayesian learner reached 62.9\% and the count rules", r"the Bayesian volatility learner reached 62.9\% and the count rules", None),
(r"to the Bayesian learner, with no increase", r"to the Bayesian volatility learner, with no increase", None),
(r"so the Bayesian learner is a computational reference", r"so the Bayesian volatility learner is a computational reference", None),
(r"learner--quota disagreement trials", r"Bayesian-learner--quota disagreement trials", None),
# 17
(r"together with diagnostic references to our episodes", r"together with hidden-information diagnostics, to our episodes", None),
(r"Rule or reference (no VLM)", r"Rule (no VLM)", None),
(r"are references, not available model strategies", r"are hidden-information diagnostics, not available model strategies", None),
(r"a reference that uses true reversal times", r"a hidden-information diagnostic that uses true reversal times", None),
(r"These rule baselines were computed", r"These rule accuracies were computed", None),
# 18
(r"the mean probability of following the currently better advisor stayed below .5", r"the mean following probability stayed below .5", None),
(r"the probability of following the currently more accurate advisor by phase, and learning curves aligned to reversals", r"the following probability by phase, and following-probability curves aligned to reversals", None),
(r"The models named the scheduled better advisor", r"The models named the currently better advisor", None),
(r"Names/follows better: the scheduled better advisor;", r"Names/follows better: the currently better advisor (higher scheduled accuracy on that trial);", None),
(r"they followed the newly better advisor", r"they followed the currently better advisor", None),
(r"they followed the better advisor on only 59--84\% of items", r"they chose the advisor described as better on only 59--84\% of items", None),
# 19
(r"a logistic regression of following A on its correctness at lags 1--10", r"a logistic regression of choosing advisor A's report on A's correctness at lags 1--10", None),
# 20
(r"Masking past outcomes improves accuracy;", r"Masking past outcome labels improves accuracy;", None),
(r"whether past outcomes and advisor reports are shown", r"whether past outcome labels and advisor reports are shown", None),
(r"masking the reports, the outcome, or both against", r"masking the reports, the outcome label, or both against", None),
(r"block contains each outcome five times", r"block contains each outcome label five times", None),
(r"answer labels", r"answer words", 4),
(r"answer-label mapping", r"answer-word mapping", 1),
# 21
(r"Open-model choices use candidate logits;", r"Open-model choices use the logits of the two answer words (candidate logits);", None),
# 22
(r"after averaging label-aligned logits over the counterbalanced copies; exact ties", r"after averaging label-aligned logits over the two copies of each episode pair (the episode and its counterbalanced copy); exact ties", None),
# 23
(r"Analyses average the label-aligned logits within each pair.", r"Analyses average the label-aligned logits within each episode pair.", None),
(r"alternates in pairs, and non-pet", r"alternates every two episodes, and non-pet", None),
(r"shows the original and selected pairs at 224 pixels", r"shows the original and selected icon pairs at 224 pixels", None),
# 25
(r"advice-task comparisons", r"advice-taking-task comparisons", 1),
(r"advice task", r"advice-taking task", 3),
# 26
(r"this task varies volatility across games", r"this task varies the hazard rate across games", None),
(r"in which volatility differs across whole games", r"in which the hazard rate differs across whole games", None),
(r"for hazard rates between 1/8 and 1/30", r"for assumed hazards $h_{\mathrm{CP}}$ between 1/8 and 1/30", None),
(r"at the other hazards", r"at the other hazard rates", None),
(r"participants' estimated forgetting rose with hazard rate, whereas", r"participants' estimated forgetting rose with the hazard rate (the per-trial probability of a payoff change), whereas", None),
(r"increased with change frequency", r"increased with the probability of change", None),
# 27
(r"report no learning rates by hazard rate", r"report no fitted learning or forgetting rates by hazard rate", None),
(r"the human discounting estimate did not increase with hazard rate", r"the participants' stale-value discounting did not increase with hazard rate", None),
(r"even this learner's learning rate rises", r"even this learner's implied learning rate rises", None),
# 28
(r"correctness kernel in Fig.~\ref{fig:kernel}", r"integration kernel in Fig.~\ref{fig:kernel}", None),
(r"Curves show correctness integration kernels", r"Curves show integration kernels", None),
(r"The kernel weights on past correctness", r"The integration-kernel weights on past correctness", None),
(r"Strategy and kernel analyses", r"Strategy and integration-kernel analyses", None),
# 29
(r"\paragraph{Sensitivity to counterbalancing the readout (exploratory).}", r"\paragraph{Sensitivity to per-copy scoring (exploratory).}", None),
(r"stale-value regression and readout sensitivity were specified", r"stale-value regression and per-copy scoring sensitivity were specified", None),
# 30
(r"without modeling our block quotas", r"without modeling the fixed per-block counts described above", None),
# 31
(r"both reports were shown in one labeled image, although each panel was read correctly", r"both reports were shown in one composite image (one labeled panel per advisor), although each panel was read correctly", None),
# 32
(r"summarizes the same curves over the five fixed models", r"averages the same curves over the five open models", None),
# 33
(r"\Summary{} (correct-trial counts)", r"\Summary{} (each advisor's count of correct trials)", None),
(r"supplies counts of correct trials (open diamond)", r"supplies each advisor's count of correct trials (open diamond)", None),
# 34
(r"described counts, short and long easy histories", r"described counts (counts of correct trials stated without the trial history), short and long easy histories", None),
# 35
(r"and an integration rule outside the model, help models", r"and an integration rule outside the model (here, a count rule), help models", None),
(r"Before trial $t$, the controller runs", r"Before trial $t$, the change-point controller (which selects the trial range for the change-point \Summary{} variant and count rule) runs", None),
# 36
(r"a constant-rate delta rule fitted to the same episode", r"a constant-rate delta learner fitted to the same episode", None),
(r"This defines the synthetic delta agent", r"This defines the synthetic delta learner", None),
(r"Delta rule ($\alpha=0.3$, $\beta=8$)", r"Delta learner ($\alpha=0.3$, $\beta=8$)", None),
(r"50\% delta &", r"50\% delta learner &", None),
(r"For the delta rule the fitted kernel", r"For the delta learner the fitted integration kernel", None),
(r"so the delta rule agrees best", r"so the delta learner agrees best", None),
(r"\paragraph{Constant-rate reference.}", r"\paragraph{Constant-rate delta learner.}", None),
# 37
(r"high raw agreement", r"high uncorrected agreement", None),
(r"rendered prompt hash, raw candidate logits", r"rendered prompt hash, unnormalized candidate logits", None),
# 38
(r"boundary on the format effect", r"boundary on the history-format effect", None),
(r"This supports the format effect under", r"This supports the history-format effect under", None),
# 39
(r"the result of step (i)", r"the result of computation (i)", None),
(r"the result of steps (i) and (ii)", r"the result of computations (i) and (ii)", None),
(r"(step (i) alone)", r"(computation (i) alone)", None),
# 40
(r"the proprietary API models", r"the proprietary models", None),
(r"for these models, and accuracy with \Summary{}", r"for the two proprietary models, and accuracy with \Summary{}", None),
# 41
(r"how often each advisor was right", r"how often each advisor was correct", None),
(r"had been right on every recent trial", r"had been correct on every recent trial", None),
(r"extracting source-relevant evidence", r"extracting evidence about advisor correctness", None),
(r"forgets the unchosen option faster", r"forgets the unchosen machine faster", None),
# 42
(r"day 1 reads", r"trial 1 reads", None),
(r"On matched items with per-copy scoring", r"On the same trials with per-copy scoring", None),
# 43
(r"blocks use of its correctness statements", r"prevents use of its correctness statements", None),
(r"Lower block:", r"Lower part:", None),
(r"must be held out together", r"must be assigned to the same cross-validation fold", None),
(r"\paragraph{Recovery.}", r"\paragraph{Parameter recovery.}", None),
(r"genuine reversals", r"true reversals", None),
(r"Recent-trial \Summary{} variants recover", r"Last-five and change-point \Summary{} variants recover", None),
]
