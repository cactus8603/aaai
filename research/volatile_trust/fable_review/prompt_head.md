You are reviewing a draft paper for the NeuroAI workshop at AAAI-27 (8-page archival PMLR track; workshop papers are judged on relevance to neuroscience/cognitive science and AI, soundness, and clarity). The full LaTeX source follows after this brief. Cells marked \tbd and \todo are pending runs; ignore missing numbers, but do judge whether the claims are supported by the numbers that are present.

New results NOT yet in the draft (use them when judging the framing):
- Two proprietary reasoning models were queried through their APIs (generated answers, reasoning effort "low") on a held-out subset: 24 episodes x 2 counterbalanced copies, every 4th trial, text reports, three history formats (raw, correctness-only, all-trial summary). Same-subset accuracy:
  - Claude Fable 5.2 (complete, 2,880 calls): raw .629, correctness-only .623, summary .585. It shows visible reasoning in 77% of raw answers.
  - GPT-6-Astra (partial, ~560 calls per format so far): raw .626, correctness-only .624, summary .606.
  - The five open models without reasoning on the same subset: raw .446-.529, correctness-only .571-.615, summary .565-.598.
  - Reference rules on the same subset: count over all past trials .585, last-5 count .679; Bayesian (Behrens-style) volatility learner .629.
  So the API reasoning models reach the Bayesian learner from the raw history, whereas the open models do not.
- A reasoning checkpoint of an open model (Qwen3-VL-8B-Thinking) is queued with raw, correctness-only and summary histories; results unknown.

Constraints from the authors: no experiment codes in the paper (descriptive condition names only); every term must be defined at first use; do not propose new large experiments (the deadline is in 7 days; cheap re-analyses are fine).

Please give:
1. An overall verdict as a reviewer (main strengths, main weaknesses, likely score), in at most 150 words.
2. The single framing you recommend for title, abstract and introduction, given the API result. Say what the central claim should be and what must no longer be claimed.
3. A prioritized list (most important first, at most 15 items) of concrete revisions to claims, wording and narrative. For each: the location (section/paragraph or a quoted phrase), the problem, and a concrete replacement text where applicable.
4. Statements in the draft that overreach or are not supported by the reported numbers (quote them).
5. What a NeuroAI reviewer will most likely ask that the paper does not answer, and how to address it with text or a cheap re-analysis.

Be specific and critical. Write in English.

=== LaTeX source ===
