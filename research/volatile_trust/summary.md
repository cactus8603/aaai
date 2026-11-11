# Volatile Trust: literature and novelty summary (2026-11-11)
Full details, sources and BibTeX are in `literature.json`. The topic scout's report is in `topic_scout.md`.

## Verdict
- The full combination appears novel: VLMs, the Behrens-2008 schedule, per-phase learning rates, image vs text reports, judgment vs choice, and now a surprise-gated controller. No single component is new.
- Partial pre-emption: Wang, Ward & Zhang 2026 (arXiv 2604.04182, IPMU 2026) put LLMs on a non-social, text-only reversal task with fixed vs random schedules and fitted gain/loss learning rates. They do not test stable vs volatile learning rates.
- Also cite 2512.18489 (LLMs as discounted Bayesian filters with a fixed discount).

**Minimal claim:** we adapt the Behrens et al. (2008) advisor-fidelity schedule to frozen open VLMs. We test whether their choice-implied learning rate is higher in the volatile phase. We then test whether a training-free change-point controller, by analogy to LC-NE signalling of unexpected uncertainty, improves adaptation. Say "to our knowledge, within a bounded search".

**Do not claim to be first in:**
- LLM reversal learning (2604.04182, Reflection-Bench, CogBench)
- cognitive-model fits of in-context learning (CogBench, Schubert 2024, Hayes 2024)
- trust under changing reliability (ECL, Chen 2606.14923, Sigma-Mem)
- estimation-vs-use gaps (Learn2Discern, the knowing-doing gap)

Also make no brain-mechanism or human-alignment claim.

## Closest prior work
| # | Paper | Difference from ours |
|---|---|---|
| 1 | 2604.04182 | Non-social reward reversal; gain/loss learning rates only; API models, text only |
| 2 | ECL 2601.21742 | One reliability flip at test time; no fits; RL training |
| 3 | Chen 2606.14923 | Scripted failures; trust measured as verification; no fits |
| 4 | Schubert ICML 2024 | Stationary bandits; asymmetric learning rates |
| 5 | CogBench ICML 2024 | Learning rate fitted only in a stationary task |
| 6–8 | Wang 2603.07717; Hayes 2405.11422; Reflection-Bench ICML 2025 | Stationary or a single reversal; no social element |
| 9 | Sigma-Mem 2607.27958 | External memory module |
| 10–11 | Learn2Discern; LLMs are Greedy Agents | Precedents for the estimation-vs-use gap |
| 12 | 2512.18489 | Fixed-discount Bayesian filter |
| 14 | IsoBench COLM 2024 | Precedent for the image-vs-text gap |

## Behrens 2008 design (verified in the Supplementary Information)
- 120 trials, 24 subjects.
- Advice schedule:
  - trials 1–30: 75% correct
  - trials 31–70: alternates between 80% correct and 80% incorrect every 10 trials
  - trials 71–120: 85% incorrect
  - 11 subjects got the inverted schedule
- Our volatile phase matches theirs exactly. Our stable phase differs, so we say "adapted from".
- Behrens 2007 RL2 (separate stable and volatile learning rates) is the same as our M2.

## Venue
- Deadline: Nov 20, 23:59 AoE.
- Format: short papers of 4 pages, non-archival, AAAI-27 two-column, double-blind. Author kit: https://aaai.org/authorkit27/
- Non-archival status allows later submission to AAAI or ICLR.
- **Every author needs an active OpenReview profile.** Profiles registered with a public email such as Gmail can sit in moderation for up to 2 weeks, so use an institutional email.
