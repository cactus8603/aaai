# Round 1 (Astra) edits accepted after checking; numbers refer to Astra's issue list.
E = [
# 1
(r"the rendered prompt hash, \Raw{} candidate logits", r"the rendered prompt hash, raw candidate logits", None),
# 2
(r"The readout is the choice probability $y_t$ for advisor A's report,", r"For the open models, the readout is the choice probability $y_t$ for advisor A's report,", None),
(r"Accuracy is the percentage of trials ($t\ge2$) on which the higher-logit answer is correct,", r"For the open models, accuracy is the percentage of trials ($t\ge2$) on which the higher-logit answer is correct,", None),
# 3
(r"\Summary{} was numerically lower than \Raw{}, at the level of its rule.", r"accuracy with \Summary{} was numerically lower than with \Raw{}, at the level of the all-trial count rule.", None),
(r"followed \Summary{} as faithfully as its rule", r"agreed with the all-trial count rule when given \Summary{}", None),
(r"On held-out episodes, accuracy followed the rule in every variant:", r"On held-out episodes, model accuracy was close to that of the corresponding count rule for every \Summary{} variant:", None),
(r"how well it would follow \Summary{} in the requested format", r"how closely it would follow the all-trial count rule when given \Summary{} and asked for a one-word answer", None),
# 4 following probability
(r"exact ties receive half credit and chance is 50\%.", r"exact ties receive half credit and chance is 50\%. The \emph{following probability} is the choice probability for the advisor with the higher scheduled accuracy on that trial.", None),
(r"Curves are mean choice probabilities over the five open models on held-out episodes,", r"Curves are mean following probabilities over the five open models on held-out episode pairs,", None),
(r"Each curve is a mean choice probability on held-out episodes;", r"Each curve is a mean following probability on held-out episode pairs;", None),
(r"which plots choice probabilities instead.", r"which plots following probabilities rather than proportions of discrete choices.", None),
# 5
(r"after each outcome, regressed on the signed outcome separately for the stable and volatile phases, measures how strongly the model responds to new outcomes.",
 r"after each trial, regressed on the signed correctness of advisor A on that trial separately for the stable and volatile phases, measures how strongly the model responds to new correctness observations.", None),
(r"summed weight of the last three outcomes:", r"summed integration-kernel weights on the last three correctness observations:", None),
# 6
(r"counts over all or recent trials, BOCPD and a label bias.", r"the all-trial and recent-trial count rules, the change-point count rule and an outcome-label bias.", None),
(r"can agree most with BOCPD,", r"can agree most with the change-point count rule,", None),
(r"Delta rule ($\alpha=0.3$, $\beta=8$) & BOCPD &", r"Delta rule ($\alpha=0.3$, $\beta=8$) & Change-point count rule &", None),
(r"the delta rule agrees best with BOCPD;", r"the delta rule agrees best with the change-point count rule;", None),
(r"BOCPD: change-point rule;", r"BOCPD: the change-point count rule;", None),
(r"BOCPD posterior observer &", r"BOCPD posterior-predictive rule &", None),
(r"Change-point count uses the inferred run's majority; the posterior observer uses BOCPD's predictive probability.",
 r"The change-point count rule uses the inferred run's majority; the BOCPD posterior-predictive rule uses BOCPD's predictive probability.", None),
# 7
(r"Accuracy follows A when $q_t>.5$ and B when $q_t<.5$, with half credit at ties.", r"The learner chooses A when $q_t>.5$ and B when $q_t<.5$; ties receive half credit.", None),
# 8
(r"(shaded: 95\% bootstrap CI over episodes)", r"(shaded: 95\% bootstrap CI over episode pairs)", None),
# 9
(r"The reversal-aligned reference uses reversals only up to $t-1$,", r"The reversal-aligned count rule uses reversals only up to $t-1$,", None),
(r"up to $t-1$ (reversal-aligned reference).", r"up to $t-1$ (reversal-aligned \Summary{} variant).", None),
(r"The same holds for the reversal-aligned reference (72.7--74.4\%; rule alone 74.4\%).", r"The same holds for the reversal-aligned \Summary{} variant (72.7--74.4\%; count rule alone 74.4\%).", None),
(r"Since reversal uses true change times hidden from the models and is a reference.", r"Since reversal is the reversal-aligned \Summary{} variant, a reference that uses true reversal times hidden from the models.", None),
(r"Since true reversal & 74.4", r"Count since true reversal & 74.4", None),
# 10
(r"On held-out episodes the gap was positive in every non-pet cover story", r"On the 48 held-out episode pairs the gap was positive in every non-pet cover story", None),
(r"Rules on 48 held-out and 24 robustness episodes.", r"Rules on the 48 held-out episodes and the 24 robustness-set episodes.", None),
# 11 (define "points" once instead of 26 replacements)
(r"These CIs are pointwise:", r"Differences between accuracies are in percentage points (``points''). These CIs are pointwise:", None),
# 12
(r"Bayesian reference learner", r"Bayesian volatility learner", 2),
(r"in the Bayesian reference.", r"in the Bayesian volatility learner.", None),
(r"Behrens-inspired Bayesian reference on our episodes", r"Behrens-inspired Bayesian volatility learner on our episodes", None),
(r"\subsection{D. Rule Simulations and Bayesian Reference}", r"\subsection{D. Rule Simulations and Bayesian Volatility Learner}", None),
(r"\paragraph{Bayesian reference implementation.}", r"\paragraph{Bayesian volatility learner implementation.}", None),
(r"Our hierarchical learner follows the volatility-learning idea", r"Our Bayesian volatility learner follows the volatility-learning idea", None),
(r"prompt-wording checks and the Bayesian reference were added", r"prompt-wording checks and the Bayesian volatility learner were added", None),
# 13
(r"in which reporters say whether a hidden animal is a cat or a dog", r"in which advisors say whether a hidden animal is a cat or a dog", 2),
(r"adapts to a volatile source", r"adapts to an advisor whose accuracy changes", None),
# 14 / 15
(r"A pair comprises an episode and its counterbalanced copy,", r"An episode pair comprises an episode and its counterbalanced copy,", None),
(r"48 held-out test episode pairs", r"48 held-out episode pairs", None),
(r"on held-out test episodes with text reports.", r"on held-out episode pairs with text reports.", None),
(r"Cohen's $\kappa$ on held-out episodes,", r"Cohen's $\kappa$ on held-out episode pairs,", None),
(r"All 48 held-out pairs and trials", r"All 48 held-out episode pairs and trials", None),
(r"48 separate episodes form the held-out test set.", r"48 separate episodes form the held-out set.", None),
(r"on the same test trials.", r"on the same held-out trials.", None),
(r"Full test set: 48 episode pairs;", r"Full held-out set: 48 episode pairs;", None),
# 17
(r"\subsection{E. Change-Point Controller and Summary Variants}", r"\subsection{E. Change-Point Controller and \Summary{} Variants}", None),
(r"\paragraph{Summary variants.}", r"\paragraph{\Summary{} variants.}", None),
# 18
(r"equivalent to the rule in five of five models for the change point and three of five for the window", r"equivalent to the corresponding count rule in five of five models for the change-point variant and three of five for the last-five variant", None),
(r"exceeded .5 two (change point) or three (last five) trials after a reversal", r"exceeded .5 two (change-point variant) or three (last-five variant) trials after a reversal", None),
(r"last 5 trials 66.7--69.1\% (rule alone 69.2\%), change point 66.6--67.7\% (67.8\%), last 10 trials 60.5--61.6\% (61.2\%), last 20 trials 58.1--58.6\% (57.9\%).",
 r"last five trials 66.7--69.1\% (count rule alone 69.2\%), change point 66.6--67.7\% (67.8\%), last 10 trials 60.5--61.6\% (61.2\%), last 20 trials 58.1--58.6\% (57.9\%).", None),
# 19
(r"four of five were equivalent to the rule alone within", r"four of five were equivalent to the all-trial count rule alone within", None),
(r"Model $-$ rule [90\% CI]", r"Model $-$ count rule [90\% CI]", None),
(r"Last-five count rule & Last 5 & 1.00", r"Last-five count rule & Last-five count rule & 1.00", None),
(r"the last-5 agent has a flat kernel up to lag 5.", r"the last-five count-rule agent has a flat integration kernel up to lag 5.", None),
(r"accuracy: repeat 45.8; WSLS 62.3; count 58.5; last 5 67.9; learner 62.9; outcome quota (hidden) 70.2",
 r"accuracy: repeat previous outcome 45.8; WSLS 62.3; all-trial count rule 58.5; last-five count rule 67.9; Bayesian volatility learner 62.9; outcome-quota rule (hidden) 70.2", None),
# 20
(r"Image reports lowered accuracy in almost every control condition,", r"Image reports lowered control-item accuracy in almost every control condition,", None),
(r"With described accuracy, accuracy in four models was lowest", r"In the described-accuracy control, control-item accuracy in four models was lowest", None),
# 21
(r"Within each 10-trial block A's realized accuracy equals the scheduled value, and each outcome label appears five times.",
 r"Within each 10-trial block, A's realized accuracy equals its scheduled accuracy, and each outcome label appears five times.", None),
(r"Known current accuracy &", r"Known current advisor accuracy &", None),
(r"Knowing the current accuracy is likewise", r"Knowing the current advisor accuracy is likewise", None),
# 22
(r"Accuracy of the Bayesian learner: proportion of trials ($t\ge 2$)", r"Accuracy of the Bayesian volatility learner: percentage of trials ($t\ge 2$)", None),
(r"Agreement is the proportion of non-tied trials with the same choice as the rule;", r"Choice agreement is the percentage of non-tied trials with the same choice as the corresponding count rule;", None),
(r"Both score a rule by the proportion of trials ($t \ge 2$) on which the advisor it favours is correct; ties score 0.5.",
 r"Both score a rule by the percentage of trials ($t \ge 2$) on which the advisor it chooses is correct; ties receive half credit.", None),
# 23
(r"compute agreement (Cohen's $\kappa$) with simple strategies.", r"compute chance-corrected agreement (Cohen's $\kappa$) with simple strategies.", None),
(r"$p_{\mathrm{TOST,Holm}}$ & Choice agreement \\", r"$p_{\mathrm{TOST,Holm}}$ & Choice agreement (\%) \\", None),
# 24
(r"\Raw{} accuracy was near chance (50.7--52.0\%) in four models.", r"\Raw{} accuracy was near chance (within 3 points of 50\%; 50.7--52.0\%) in four models.", None),
(r"\Raw{} accuracy is called near chance only within $50 \pm 3$\%.", r"\Raw{} accuracy is called near chance only within 3 percentage points of 50\%.", None),
# 25
(r"or with the outcome quota (.07--.10), a rule that would score 70.2\% on these trials;", r"or with the outcome-quota rule (.07--.10), which predicts the label with more occurrences left in the current block and would score 70.2\% on these trials;", None),
(r"where the quota and the learner disagreed they followed the learner", r"where the quota rule and the Bayesian learner disagreed they followed the learner", None),
# 26
(r"Advisor A is correct with probability 0.8 in a stable phase", r"Advisor A, the advisor with the higher scheduled accuracy on trial~1, is correct with probability 0.8 in a stable phase", None),
(r"Advisor A is the advisor favored on trial~1.", r"Advisor A is the advisor with the higher scheduled accuracy on trial~1.", None),
# 27
(r"The predictive mean $q_t=E[p_t\mid o_{1:t-1}]$ determines the choice before observing correctness $o_t$;",
 r"Let $o_t=1$ if advisor A was correct on trial $t$ and $o_t=0$ otherwise. The predictive mean $q_t=E[p_t\mid o_{1:t-1}]$ determines the choice before $o_t$ is observed;", None),
# 28
(r"Within each block, A is correct on exactly $10p$ trials", r"Within each block with scheduled accuracy $p$, A is correct on exactly $10p$ trials", None),
# 29
(r"meta-volatility $k\in\{.05,.2,.5,1\}$", r"meta-volatility $\eta\in\{.05,.2,.5,1\}$", None),
(r"$\exp[-(v'-v)^2/(2k^2)]$", r"$\exp[-(v'-v)^2/(2\eta^2)]$", None),
(r"integrating over $v$ and $k$.", r"integrating over $v$ and $\eta$.", None),
# 30
(r"compares BOCPD (posterior predictive, $h \in \{1/8, 1/15, 1/30\}$),", r"compares the BOCPD posterior-predictive rule (assumed hazard $h_{\mathrm{CP}} \in \{1/8, 1/15, 1/30\}$),", None),
(r"$h=1/8$", r"$h_{\mathrm{CP}}=1/8$", None),
(r"$h=1/15$", r"$h_{\mathrm{CP}}=1/15$", None),
(r"a constant hazard rate $h = 1/15$.", r"a constant assumed hazard $h_{\mathrm{CP}} = 1/15$.", None),
# 31
(r"log-volatility values $v\in[\log(.02),\log(3)]$,", r"log-volatility values $v\in[\log(.02),\log(3)]$ (the log step size of changes in advisor accuracy),", None),
# 32
(r"assigned in blocks so that cover story is crossed with phase order.", r"assigned in groups of episodes so that cover story is crossed with phase order.", None),
(r"non-pet cover stories rotate in blocks of four episodes", r"non-pet cover stories rotate in groups of four episodes", None),
# 33
(r"phase lengths 40/40, 60/60 and 80/40; dwell 10 or 15 trials", r"phase lengths of 40/40, 60/60 and 80/40 trials; dwell times of 10 or 15 trials", None),
(r"For our setting (0.8; 40/40; dwell 10),", r"For our setting (advisor accuracy 0.8; phases of 40/40 trials; dwell time 10 trials),", None),
(r"volatile-phase dwell times are drawn uniformly from 6--16 trials.", r"volatile-phase dwell times (trials between reversals) are drawn uniformly from 6--16 trials.", None),
# 34
(r"full-vocabulary probability mass on the two answer tokens and greedy format following.", r"full-vocabulary probability mass on the two answer tokens and greedy one-word answer compliance.", None),
(r"nearly all mass fell on the answer words in every format.", r"nearly all mass fell on the answer words in every history format.", None),
(r"& \multicolumn{3}{c}{Candidate mass} & Format \\", r"& \multicolumn{3}{c}{Candidate mass} & One-word \\", None),
(r"Model & Mean & Min & Share $<0.5$ & following \\", r"Model & Mean & Min & Share $<0.5$ & answers \\", None),
(r"format following is the proportion of greedy outputs equal to an answer word.", r"one-word answers is the proportion of greedy outputs equal to an answer word.", None),
(r"\paragraph{Image and chart formats.}", r"\paragraph{Image and chart payoff presentations.}", None),
# 35
(r"In the judgment readout the current reports and question are replaced by", r"For the judgment question, the current reports and prediction question are replaced by", None),
(r"\paragraph{Readout.} Models use bfloat16", r"\paragraph{Open-model readout.} The open models use bfloat16", None),
# 36
(r"The readout is the choice probability between the two letters from the first-token logits, as in the main task.", r"The readout is the choice probability for each machine's answer letter, from the first-token logits normalized over the two letters, as in the main task.", None),
# 37
(r"\paragraph{Model-free check of forgetting (exploratory).}", r"\paragraph{Regression check of forgetting (exploratory).}", None),
# 38
(r"the original instructions state it (told).", r"the original instructions state it (stated).", None),
(r"with hidden and told conditions shown in Fig.~\ref{fig:human}.", r"with hidden- and stated-hazard conditions shown in Fig.~\ref{fig:human}.", None),
(r"Participants (told) &", r"Participants (hazard stated) &", None),
# 39
(r"20 sampled choice sequences per model episode", r"20 sampled choice sequences per model game", None),
# 40
(r"kernel regression", r"integration-kernel regression", 2),
(r"The 15-lag kernel describes local sensitivity", r"The 15-lag integration kernel describes local sensitivity", None),
# 41
(r"weight on the previous outcome", r"weight on repeating the previous outcome", 9),
(r"repetition weights (Table~\ref{tab:mask})", r"weights on repeating the previous outcome (Table~\ref{tab:mask})", None),
# 42
(r"50\% copy, 50\% delta", r"50\% repeat previous outcome, 50\% delta", None),
(r"Repeat last outcome", r"Repeat previous outcome", 3),
# 43
(r"Described \%", r"Described accuracy", 4),
(r"Single trial & Text &", r"Single-trial check & Text &", None),
(r"Composite read & Image &", r"Composite-panel reading & Image &", None),
(r"Chart read & Chart &", r"Correctness-chart reading & Chart &", None),
# 44
(r"the position of \CorrectnessOnly{} between \Raw{} and \Summary{},", r"the position of \CorrectnessOnly{} between \Raw{} and \Summary{} (acc: accuracy),", None),
# 45
(r"Best strategy: highest Cohen's $\kappa$; Repeat $w$ and Lag-1 $w$:", r"Best strategy: highest Cohen's $\kappa$. In agent labels $w$ is the generating weight; in column headings it is the fitted coefficient. Repeat $w$ and Lag-1 $w$:", None),
# 46
(r"in one of four formats (Fig.~\ref{fig:teaser}; full prompts in Appendix~A).", r"in one of four history formats (Fig.~\ref{fig:teaser}; full prompts in Appendix~A); prompts name trials after the cover story (e.g. days).", None),
# 47
(r"Qwen3-VL 8B and 32B", r"Qwen3-VL-8B and Qwen3-VL-32B", None),
(r"InternVL3.5 8B and 38B", r"InternVL3.5-8B and InternVL3.5-38B", None),
(r"77\% of Fable's \Raw{} answers", r"77\% of Claude Fable 5.2's \Raw{} answers", None),
(r"for Qwen models", r"for Qwen3-VL models", None),
(r"For Gemma-3,", r"For Gemma-3-27B,", None),
(r"Gemma's failures", r"Gemma-3-27B's failures", None),
# 48 American spelling
(r"analysed with the same", r"analyzed with the same", None),
(r"without modelling our block quotas", r"without modeling our block quotas", None),
(r"colours", r"colors", 2),
(r"labelled image", r"labeled image", None),
(r"behavioural", r"behavioral", 2),
(r"behaviour expected of people", r"behavior expected of people", None),
(r"Behaviour", r"Behavior", 2),
(r"unanalysed", r"unanalyzed", None),
(r"Coloured", r"Colored", None),
# 49
(r"the last inferred change-point", r"the last inferred change point", None),
(r"logit noise s.d.\ 0.3", r"logit-noise standard deviation 0.3", None),
(r"We re-ran the four text history formats", r"We reran the four text history formats", None),
]
