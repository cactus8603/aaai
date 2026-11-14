# Round 3 (GPT-6-Astra) edits; numbers refer to Astra's round-3 list.
E = [
# 1
(r"four of five were equivalent to the all-trial count rule alone within $\pm$2 points", r"four of five had accuracy equivalent to that of the all-trial count rule within $\pm$2 points", None),
(r"equivalent to the corresponding count rule in five of five models for the change-point variant", r"accuracy equivalent to that of the corresponding count rule in five of five models for the change-point variant", None),
# 2
(r"\Summary{} variants covering the last five trials or the run since a change point inferred by BOCPD were followed as faithfully", r"Given the last-five and change-point \Summary{} variants, models agreed as closely with the corresponding count rules", None),
(r"set by the rule outside the model.", r"set by the count rule outside the model.", None),
# 3
(r"On 48 held-out episodes,", r"On 48 held-out episode pairs,", None),
(r"On held-out episodes, accuracy was", r"On the held-out set, accuracy was", None),
(r"44.5\% on held-out episodes", r"44.5\% on the held-out set", None),
(r"On held-out episodes, all predictions", r"On the held-out set, all predictions", None),
(r"run on the same held-out episodes", r"run on the same held-out set", None),
(r"On held-out episodes much of the cost", r"On the held-out set, much of the cost", None),
# 4
(r"with advisor names and answer words swapped.", r"with advisor names swapped and the answer-word mapping reversed.", None),
(r"in which advisor names and answer words are swapped while outcomes are identical; averaging over the two cancels name and label biases.",
 r"in which advisor names are swapped and the answer-word mapping is reversed while the outcome sequence is unchanged; averaging over the two copies cancels advisor-name and answer-word biases.", None),
# 5
(r"\CorrectnessOnly{} raised accuracy by 8--16 percentage points", r"\CorrectnessOnly{} raised prediction accuracy by 8--16 percentage points", None),
(r"learn a changing, non-stationary accuracy by in-context learning", r"learn changing, non-stationary advisor accuracy by in-context learning", None),
(r"For the open models, accuracy is the percentage of trials", r"For the open models, prediction accuracy (hereafter accuracy) is the percentage of trials", None),
# 6
(r"every model named the correct advisor on a single shown trial", r"every model identified which advisor was correct on a single shown trial", None),
(r"Accuracy was mostly higher when the name was printed on the image", r"Control-item accuracy was mostly higher when the name was printed on the image", None),
(r"single cells of the correctness chart reliably (61--71\%)", r"single cells of the correctness chart reliably (cell-reading accuracy 61--71\%)", None),
# 7
(r"The \emph{following probability} is the choice probability for the advisor with the higher scheduled accuracy on that trial.",
 r"The \emph{currently better advisor} has the higher scheduled accuracy on that trial; the \emph{following probability} is the choice probability for that advisor.", None),
# 8
(r"compute chance-corrected agreement (Cohen's $\kappa$) with simple strategies.",
 r"compute chance-corrected agreement (Cohen's $\kappa$) with simple strategies; \emph{choice agreement} is the percentage of non-tied trials on which the model and a count rule choose the same advisor.", None),
# 9
(r"after a reversal they followed the currently better advisor on 6--9\% of trials in the first three trials of the reversed block, rising to 94--99\% in trials 8--10.",
 r"after a reversal, 6--9\% of their choices followed the currently better advisor in the first three trials of the reversed block, rising to 94--99\% in trials 8--10.", None),
# 10
(r"had similar observed \Raw{} accuracy to a Bayesian volatility learner (63\%)", r"had observed \Raw{} accuracy similar to that of a Bayesian volatility learner (63\%)", None),
(r"have similar observed \Raw{} accuracy to a Bayesian volatility learner", r"have observed \Raw{} accuracy similar to that of a Bayesian volatility learner", None),
(r"reached the learner (66.6--69.1\%; Appendix~E)", r"reached the learner's accuracy (66.6--69.1\%; Appendix~E)", None),
(r"had similar observed \Raw{} accuracy to the Bayesian volatility learner", r"had observed \Raw{} accuracy similar to that of the Bayesian volatility learner", None),
# 11
(r"a normative learner forgets the unchosen machine faster", r"a normative bandit learner forgets the unchosen machine faster", None),
# 12
(r"\Raw{} (reports and outcomes)", r"\Raw{} (reports and outcome labels)", None),
(r"\Raw{} lists reports and outcomes;", r"\Raw{} lists reports and outcome labels;", None),
(r"\Raw{} lists both reports and the outcome", r"\Raw{} lists both reports and the outcome label", None),
(r"The reports and outcomes therefore do more than add work:", r"The reports and outcome labels therefore do more than add work:", None),
(r"the reports and outcomes in \Raw{}", r"the reports and outcome labels in \Raw{}", None),
(r"\multicolumn{2}{c}{Outcome shown} & \multicolumn{2}{c}{Outcome masked}", r"\multicolumn{2}{c}{Outcome label shown} & \multicolumn{2}{c}{Outcome label masked}", None),
# 13 (main text and App. J)
(r"past correctness", r"past advisor correctness", 4),
# 14
(r"and Gemma-3-27B \citep{gemmateam2025gemma3}.", r"and Gemma-3-27B \citep{gemmateam2025gemma3}; B denotes billions of parameters.", None),
# 15
(r"Per-model kernels and CIs", r"Per-model integration kernels and CIs", None),
(r"Under \CorrectnessOnly{}, the kernel instead peaked", r"Under \CorrectnessOnly{}, the integration kernel instead peaked", None),
(r"where the quota rule and the Bayesian learner disagreed", r"where the outcome-quota rule and the Bayesian learner disagreed", None),
# 16
(r"afterwards", r"afterward", 2),
# 17
(r"InternVL3.5-8B followed \Summary{} on only 72\% of non-tied trials (98\% with the original wording).", r"InternVL3.5-8B showed only 72\% choice agreement with the all-trial count rule when given \Summary{} (98\% with the original wording).", None),
# 18 (artwork labels unchanged; caption maps them)
(r"Last-5 rule and Count rule: the last-five and all-trial count rules; BOCPD: the change-point count rule (not the BOCPD posterior-predictive rule);",
 r"Last-5 rule, Count rule and BOCPD: the last-five, all-trial and change-point count rules (the change point is inferred by BOCPD);", None),
# 19
(r"In the counterbalanced copy, advisor names, answer words and the order in which the instructions introduce the advisors are swapped while outcomes and images are identical;",
 r"In the counterbalanced copy, advisor names are swapped, the answer-word mapping and the advisor-introduction order are reversed, and the outcome sequence and images are unchanged;", None),
(r"outcomes and images are identical.", r"the outcome sequence and images are unchanged.", None),
# 20
(r"count rule since the last true reversal", r"reversal-aligned count rule", 2),
(r"We therefore do not claim that a change-point rule is optimal", r"We therefore do not claim that the change-point count rule is optimal", None),
(r"Outcome quota & 65.5 & -- \\", r"Outcome-quota rule & 65.5 & -- \\", None),
# 21
(r"& Change-point & Since reversal & Last 10 & Change-point \\", r"& Change-point & Reversal-aligned & Last 10 & Change-point \\", None),
(r"Since reversal is the reversal-aligned \Summary{} variant, a hidden-information diagnostic", r"The reversal-aligned \Summary{} variant is a hidden-information diagnostic", None),
# 22
(r"The controller is a training-free rule outside the model;", r"The change-point controller is a training-free trial-range selector outside the model;", None),
(r"but we use it as an integration rule, not as a model of the VLM or of the brain.", r"but we use it to select the trial range for correctness counts, not as a model of the VLM or of the brain.", None),
# 23
(r"transition weights in logit accuracy; the original formulation uses a beta transition in accuracy.", r"transition weights in logit advisor accuracy; the original formulation uses a beta transition in advisor accuracy.", None),
(r"The grid has 60 equally spaced accuracy values", r"The grid has 60 equally spaced advisor-accuracy values", None),
# 24
(r"below the 80\% described-accuracy criterion.", r"below the 80\% control-item accuracy criterion for the described-accuracy control.", None),
# 25
(r"$w=3$", r"$w_{\mathrm{gen}}=3$", 2),
(r"Repeat $w$ & Lag-1 $w$", r"Repeat $\hat{w}$ & Lag-1 $\hat{w}$", None),
(r"In agent labels $w$ is the generating weight; in column headings it is the fitted coefficient. Repeat $w$ and Lag-1 $w$:",
 r"In agent labels $w_{\mathrm{gen}}$ is the generating weight; in column headings $\hat{w}$ is the fitted coefficient. Repeat $\hat{w}$ and Lag-1 $\hat{w}$:", None),
# 26
(r"Monte Carlo standard error 1.7\%", r"Monte Carlo standard error 1.7 percentage points", None),
# 27
(r"which gave more points", r"which gave a higher payoff", 2),
# 28
(r"& Reports masked & Outcome masked & Both masked &", r"& Reports masked & Outcome label masked & Both masked &", None),
# 29
(r"Bayesian-learner--quota disagreement trials", r"trials on which the Bayesian volatility learner and the outcome-quota rule disagree", None),
# 31
(r"Gemma-27B & IVL-38B", r"Gemma-3-27B & IVL-38B", None),
]
