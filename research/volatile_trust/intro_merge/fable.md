## A. Merge check

| # | Edit | Verdict |
|---|---|---|
| 1 | Behrens sentence (Intro ¶1) | **Fix needed.** Your "each advisor" undoes the co-author's correction. As I recall it, Behrens et al. (2008) used a single confederate advising on a two-option reward task, which is why "two advisors disagree" and "each advisor" were deleted. Use "the advisor". Changing "were" to "are consistent" is right. |
| 2 | "We adapt this human-neuroscience paradigm…" | Faithful. Expanding VLMs here at first use is correct. |
| 3 | Prior-work sentence (four 2026 citations) | Faithful and verbatim. I cannot verify these papers, so please check each characterization. Keep the sentence in the Introduction, since these keys do not appear in Related Work. |
| 4 | "Conversational LLMs can choose differently…" | Faithful. Your "differently … than when" is better grammar than the co-author's "versus"; I restored their "specified/presented". |
| 5 | Related Work, greedy choice | **Not faithful.** "…, as well as premature greedy action selection" makes the greedy finding an object of "Fits of reinforcement-learning models … have found" again. The co-author split it out as "Behavioral analyses" because it is not a model-fitting result. Keep it as a separate sentence. |
| 6 | Related Work, Diaconescu | Faithful and accurate as far as I know (arbitration as relative precision of the two systems' predictions). |

Two unpaired additions in Related Work:
- **Coin flips and discounted filtering:** the coin-flip sentence matches my recollection of that paper. I cannot verify the discounted-filtering one.
- **PhotoGuider:** this sentence is tangential and is the first candidate to cut if space is tight.

## B. Review of your shortened Introduction

Your draft is already about 410 words, so length is not the problem. A few cuts removed needed content.

**Dropped or weakened**
- **Framing:** the phrase "context-dependent learning" is gone, though it is the abstract's theme.
- **Question:** "how models learn" presupposes learning. The abstract asks "whether", and four of five models were at chance with \Raw{}.
- **What is manipulated:** the bullets use \CorrectnessOnly{}, \Summary{} and \Raw{}, but the text never says what they are. "Visible outcome label" has no referent unless \Raw{} is described.
- **Masking claim:** "a masking test predicted before held-out evaluation" is ambiguous and slightly changes the claim. I restored the original wording.
- **\RawCorrectness{} result:** "helped little" is in the abstract and is the logical step that motivates masking. It costs four words.

**Stronger than the abstract**
- The proprietary-model bullet lost both "observed" and the abstract's "tested on a subset".
- "Five VLMs receive…" is followed by a bullet about two other models. I removed the count from that sentence.
- Please check that "recovers most of the gap" holds per model. If it only holds for the mean, say "on average".

**Redundancy**
- "Requires deriving advisor correctness…" and computation (i) say the same thing, so I merged them.
- The Ge and Krishnamurthy sentence overlaps Related Work. I kept it because the co-author edited it and it sets up the gap.

**Open gap I cannot fill**
- The Introduction motivates image reports, but neither the abstract nor the contributions state an image result. Add a half-clause from your actual results, or reviewers will ask.
- Methods should also state that your task uses two advisors with conflicting reports, unlike the original.

## C. Final recommended text

```latex
People learn whom to trust from feedback. When advice can be checked against the outcome, people track how trustworthy the advisor is, and their choices are consistent with a Bayesian learner that updates faster when the advisor's accuracy is volatile \citep{behrens2008associative}; advisor reputations form quickly from a track record \citep{yaniv2000advice,bonaccio2006advice}. We adapt this human-neuroscience paradigm and its computational modeling approach to evaluate vision-language models (VLMs).

Agents built on large language models (LLMs) and VLMs face the same problem when they combine tool outputs, retrieved documents or messages from other agents whose reliability can change; for VLM-based agents, some arrive as images. Prior work has tested source weighting under fixed reliability \citep{ashkinaze2026information} and test-time reliability inversion \citep{zhou2026epistemic}, studied trust formation and recovery after scripted peer failures \citep{chen2026trust}, and proposed memories that track reliability outside the model \citep{feng2026sigmamem}. We ask whether models learn changing advisor accuracy from trial-by-trial feedback across repeated reversals---context-dependent learning without parameter updates---and how history format affects their choices.

The format of past evidence matters: conversational LLMs can choose differently when prospects are specified explicitly than when they are presented as histories of outcomes \citep{ge2026dhgap}, and summarized histories can improve exploration in bandits \citep{krishnamurthy2024explore}. Learning whom to trust involves three computations: (i) judging which advisor was correct by comparing conflicting reports with the observed outcome, which those tasks do not require; (ii) integrating these judgments across trials; and (iii) using the estimate. In the human study, (i) yields the observation from which a prediction error about the advisor is computed \citep{behrens2008associative}. We present the same history in formats that state these results to different degrees (Fig.~\ref{fig:teaser}): \Raw{} lists reports and outcome labels, \RawCorrectness{} appends the result of (i), \CorrectnessOnly{} keeps only that result, and \Summary{} gives each advisor's count of correct trials. Formats also change length and salience, so we do not assume separable internal stages. Our contributions are:
\begin{itemize}
  \item[$\bullet$] An advice-taking evaluation adapted from human neuroscience, with current reports as text or as images.
  \item[$\bullet$] For five open-weight VLMs answering directly, \CorrectnessOnly{} recovered most of the \Summary{}$-$\Raw{} gap whereas \RawCorrectness{} helped little, and a masking test, whose outcome we predicted before running it on held-out data, showed that the visible outcome label drives repetition of the previous outcome.
  \item[$\bullet$] A boundary condition: two proprietary reasoning models, tested on a subset, had \Raw{} accuracy similar to that of a Bayesian volatility learner, so the history-format effect depends on the system.
  \item[$\bullet$] A descriptive human--model comparison in a Psych-101 changing bandit: participants' estimated forgetting rose with the hazard rate (the per-trial probability of a payoff change), whereas the open models' estimates changed little.
\end{itemize}
```

Related Work, Cognitive evaluation (the greedy finding stays separate):

```latex
Fits of reinforcement-learning models to in-context behavior in bandits have found asymmetric or biased updating and rigid choice \citep{schubert2024incontext,hayes2024biased,wang2026rigidity}. Behavioral analyses have also found premature greedy action selection in bandit tasks \citep{schmied2025greedy}, and reversal-learning schedules have been compared across LLMs \citep{wang2026comparative,li2025reflectionbench}.
```

Related Work, Volatility (your merge, unchanged):

```latex
Normative accounts of learning under volatility include hierarchical Bayesian and change-point models \citep{behrens2007learning,nassar2010approximately,nassar2012rational,mathys2011bayesian,piray2021model,piray2024computational,soltani2019adaptive}, and arbitration between social and individual learning has been modeled using the relative precision of their predictions \citep{diaconescu2020neural}.
```

**Word count:** about 425 by hand count (prose plus bullets, citations excluded), slightly above your 420 target. If "hazard rate" is defined in the Psych-101 section, dropping the gloss in the last bullet brings it to about 417.

**Changes from your draft**
- **"each advisor" → "the advisor":** matches the co-author's singular and the original single-confederate design.
- **"how models learn" → "whether models learn":** matches the abstract and the at-chance \Raw{} result.
- **Restored "context-dependent learning":** this is the framing the abstract opens with.
- **Merged the gap sentence into computation (i):** removes a duplicate statement and keeps the contrast with prior tasks.
- **Named the formats in the text:** defines the macros before the bullets use them and makes "outcome label" interpretable. It replaces "the result of (i), or of (i) and (ii), to different degrees".
- **Removed "Five VLMs receive…":** avoids a mismatch with the bullet about two additional models.
- **Second bullet:** added "whereas \RawCorrectness{} helped little", restored the exact wording of the prediction claim, and used "open-weight" as in the abstract.
- **Third bullet:** added "tested on a subset" to match the abstract's hedge.
- **Tense:** results in the bullets are now in past tense, consistent with the abstract and the last bullet.
- **Image angle:** shortened the clause in the second paragraph and moved "current reports as text or as images" into the first bullet.
- **Not restored:** "the model's choices do not change later feedback" (leave to Methods), the recency citation (in Related Work), the accuracy-description condition, and the one-line-record boundary (add to the third bullet if you want full parity with the abstract).