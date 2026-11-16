# 2026-11-16 review of the version with lcy edits (discussion/review_2026-11-16_lcy_edits.md) + Table 10 fixes.
# Numbers refer to items in that review file.
import re
from pathlib import Path

_t = Path(__file__).resolve().parents[3].joinpath('paper', 'main.tex').read_text()

def _between(a, b):
    i = _t.index(a); j = _t.index(b, i)
    return _t[i:j]

# 4: Discussion paragraph; rebuild its opening and move the BOLD sentence next to the disclaimer
_disc_old = _between(r"Across five open VLMs answering directly, four of five were no better than chance with \Raw{}, \CorrectnessOnly{}",
                     r" Our manipulations change what the history states")
_bold = _between(r"In the human study, who was correct is the observation from which a prediction error about the advisor is computed; this",
                 r" \lcy{Despite explicit correctness")
_tail = _between(r" What the models learned themselves was limited.", r" Our manipulations change what the history states")
_disc_new = (r"Across five open VLMs answering directly, four of five were no better than chance with \Raw{}, and \CorrectnessOnly{} recovered most of the \Summary{}$-$\Raw{} gap. "
             r"\lcy{\RawCorrectness{} states the same correctness yet was less accurate than \CorrectnessOnly{}, and masking showed that the visible outcome label elicits repetition of the previous outcome. "
             r"This extends the question raised by the description--history gap \citep{ge2026dhgap}: in this task, visible outcome labels promote a competing outcome-repetition response.} "
             r"This resembles Stroop interference in people, in which reading \lcy{an incongruent color word} interferes with naming its ink color \citep{macleod1991half}: the correctness information was present, and masking the competing outcome label reduced the repetition."
             + _tail + " " + _bold)

E = [
# 1
(r"so the formats also ask which step of this human-derived computation a model can perform.",
 r"so the formats vary which result of this human-derived computation is supplied to the model.", None),
# 2
(r"The layout check points the same way: in the two 8B models, the compact record lowered the weight on repeating the previous outcome (Table~\ref{tab:wording}) and raised \Raw{} accuracy, although the outcome label still ended each line, so its position alone does not explain the repetition.",
 r"The compact record adds a qualification: in the two 8B models it lowered the weight on repeating the previous outcome (Table~\ref{tab:wording}) and raised \Raw{} accuracy although the outcome label still ended each line, so the format of the record also modulates the repetition.", None),
# 3
(r"The size of the gap depends on the layout: a compact one-line record per trial shrank it",
 r"The size of the gap depends on the prompt format: a compact one-line record per trial, which changes wording and layout together, shrank it", None),
(r" \lcy{This compact prompt changes both wording and layout, so the comparison does not isolate a layout effect.}", "", None),
(r"\Raw{} accuracy stayed below the count rule in every wording (development episodes; Appendix~I).",
 r"\Raw{} accuracy stayed below the count rule in every prompt format (development episodes; Appendix~I).", None),
(r"the size of the \Summary{}$-$\Raw{} gap depends on the history's layout (Appendix~I).",
 r"the size of the \Summary{}$-$\Raw{} gap depends on the prompt format (Appendix~I).", None),
# 4
(_disc_old, _disc_new, None),
(r"Our manipulations change what the history states, not internal stages,",
 r"Our manipulations change whether this observation is stated in the history, not internal stages,", None),
# 5, 6, 7 (abstract)
(r"\RawCorrectness{}, \CorrectnessOnly{} (who was correct) or \Summary{} (counts of correct trials).",
 r"\RawCorrectness{} (\Raw{} plus who was correct), \CorrectnessOnly{} (who was correct) or \Summary{} (counts of correct trials).", None),
(r"whereas adding correctness to \Raw{} helped little.", r"whereas \RawCorrectness{} helped little.", None),
(r"two proprietary reasoning models learned from \Raw{} about as well as a Bayesian volatility learner, and a compact layout narrowed the gap in two 8B models.",
 r"two proprietary reasoning models, tested on a subset, learned from \Raw{} about as well as a Bayesian volatility learner, and a compact one-line record narrowed the \Summary{}$-$\Raw{} gap in two 8B models.", None),
# 8 + 13: move robustness result into the gap paragraph; keep a short pointer in the Raw paragraph
(r" \lcy{On the robustness set, with independently sampled correctness and outcome labels and irregular reversals, \Raw{} accuracy was near chance in four models, while the \Summary{}$-$\Raw{} gap remained positive in all five ($+3.6$ to $+8.3$ percentage points; Table~\ref{tab:robust}). The direction of this comparison therefore persists beyond the main generator, although its magnitude is smaller.}",
 r" \lcy{On the robustness set, which has no outcome quotas, \Raw{} accuracy was instead near chance in four models.}", None),
(r"the gap remained $+8.8$ to $+13.9$ percentage points (Appendix~J)}.",
 r"the gap remained $+8.8$ to $+13.9$ percentage points (Appendix~J). On the robustness set, with independently sampled correctness and outcome labels and irregular reversals, the gap was smaller but remained positive in all five models ($+3.6$ to $+8.3$ percentage points; Table~\ref{tab:robust}), so it does not depend on the outcome quotas}.", None),
# 9
(r"A robustness set has 24 episodes with independent outcomes and irregular reversals.",
 r"A robustness set has 24 episodes with independently sampled correctness and outcome labels (no outcome quotas) and irregular reversals.", None),
(r"Accuracy (\%) on 24 robustness episode pairs", r"Accuracy (\%) on 24 robustness-set episode pairs", None),
(r"robustness set (independent outcomes; reported in Results)", r"robustness set (no outcome quotas; reported in Results)", None),
# 10
(r"\paragraph{\lcy{Truncating \Raw{} does not recover the benefit of \Summary{}.}}",
 r"\paragraph{\lcy{Truncating \Raw{} does not recover the benefit of recent counts (exploratory).}}", None),
(r"versus 66.6--67.7\% for \Summary{} on the same 48 held-out episode pairs",
 r"versus 66.6--67.7\% for the change-point \Summary{} variant on the same 48 held-out episode pairs", None),
# 11
(r"computational modeling approach to evaluate models.}", r"computational modeling approach to evaluate VLMs.}", None),
# 12 + 14
(r"We ask how models without parameter updates use trial-by-trial feedback", r"We ask how models use trial-by-trial feedback", None),
(r" For VLM-based agents, some of these messages arrive as images, such as screenshots or photographs, so we present the current reports both as text and as images.", "", None),
(r"and the model's choices do not change later feedback.",
 r"and the model's choices do not change later feedback. For VLM-based agents, some of these messages arrive as images, such as screenshots or photographs, so we present the current reports both as text and as images.", None),
# 15
(r"\lcy{PhotoGuider fine-tunes a VLM to generate photographic explanations conditioned on measured composition features \citep{lee2026photoguider}.}",
 r"\lcy{PhotoGuider fine-tunes a VLM to generate photographic explanations conditioned on measured composition features, showing that explicit intermediate features can shape VLM outputs \citep{lee2026photoguider}.}", None),
# 17
(r"(62.9\% and 62.8\%), the same as the Bayesian volatility learner on these trials (62.9\%)",
 r"(62.9\% and 62.8\%), matching the Bayesian volatility learner on these trials (62.9\%)", None),
# 18
(r"In the interpretable advice-taking-task comparisons, images appear only as current reports.",
 r"In the advice-taking task, images appear only as current reports; correctness-chart histories were not legible enough to interpret.", None),
# Table 10 (tab:normsim): float placement, set names, text claim about window ordering, row grouping
(r"""\begin{table}[tbp]\centering\small
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{lcc}
    \toprule
    Rule (no VLM) & Held-out & Robustness \\""",
 r"""\begin{table}[t]\centering\small
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{lcc}
    \toprule
    Rule (no VLM) & Held-out set & Robustness set \\""", None),
(r"""    \midrule
    Count rule (reversal-aligned) & 74.4 & 73.5 \\""",
 r"""    \midrule
    \multicolumn{3}{l}{\textit{Hidden-information diagnostics}} \\
    Count rule (reversal-aligned) & 74.4 & 73.5 \\""", None),
(r"Rows below the divider use hidden generator information and are hidden-information diagnostics, not available model strategies. The robustness set has no outcome quota.",
 r"Rows below the divider use generator information hidden from the models (true reversal times, scheduled accuracies or the per-block outcome quotas) and are not available model strategies. The robustness set has no outcome quotas, so the outcome-quota rule does not apply (--).", None),
(r"the all-trial count rule lags after reversals, and the ordering of the longer windows differs across sets.",
 r"the all-trial count rule lags after reversals and scores above the 10- and 20-trial count rules on the held-out set but below them on the robustness set.", None),
]
