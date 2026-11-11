# Volatile Trust: Do VLMs Adapt Source Trust to Changing Reliability In-Context?

> **注意：本文件是早期設計草稿。正式的實驗規格以 `EXPERIMENTS.md` 為準。**

Target venue: NeuroAI 2.0 workshop at AAAI-27 (https://neuroai-multimodal-workshop.github.io/).
The deadline is 2026-11-20. We plan a 4-page short paper (non-archival). Review is double-blind and uses the AAAI-27 author kit.

## Question
Two reporters give conflicting reports about a hidden binary value. After each trial, the true value is revealed.
Reporter A's fidelity, P(A correct), changes over the episode in a stable phase and a volatile phase, following Behrens et al. 2008 (Nature), "Associative learning of social value".
We ask three things:
1. Do frozen VLMs track changes in reliability in-context, as measured by reversal adaptation and perseveration?
2. Is their effective learning rate higher in the volatile phase than in the stable phase? This is the Behrens signature.
3. Does tracking differ when the current report is an image rather than text (multimodal factor)? Does the model's explicit reliability judgment diverge from the reliability its choices imply (estimation versus use)?

## Task (per episode)
- Two reporters with neutral aliases. Each episode samples the aliases from a pool, and their order is counterbalanced.
- Every trial is a conflict trial: A and B always report different labels. The hidden truth equals A's report with probability p_A(t). Equivalently, B's fidelity is 1 - p_A(t). This is the same structure as the one-advisor Behrens design.
- T = 80 trials.
  - Stable phase: 40 trials at p_A = 0.8.
  - Volatile phase: 40 trials. p_A alternates between 0.2 and 0.8 every 10 trials, starting from the opposite of the preceding stable value. The starting value is counterbalanced.
- Orders: SV (stable first) and VS (volatile first), half the episodes each. Which alias is favored first is also counterbalanced.
- Realized accuracy is stratified: each 10-trial window has exactly round(10 * p) A-correct trials, in shuffled positions. The truth label is balanced within windows.
- Domains:
  - color: labels red and blue. Images are solid squares.
  - pet: labels cat and dog. Images are Oxford-IIIT Pet letterbox images.
- History is always textual: past reports and the revealed truth.
- The current trial's two reports appear as text or as images. This is the format factor.

## Readouts (frozen models, single-token candidates, raw logits)
- choice: at trial t, the model sees the history of trials 1..t-1 plus the current reports and predicts the hidden value. We record P(label = A's report), normalized over the two labels.
- judgment (text only): at trial t, the model sees history 1..t-1 and is asked "Which reporter is currently more reliable?". We record P(A) normalized over the two aliases.
- Every trial t = 1..80 is queried.

## Models
These are cached and locally pinned:
- Qwen2.5-VL-3B
- Qwen3-VL-4B
- InternVL3.5-4B

We plan to add, if time permits:
- Qwen2.5-VL-7B
- Qwen3-VL-8B
- InternVL3.5-8B

We use RTX 3090 GPUs (bf16) with batch size 1. The backend is `research/source_trust/model_backends.py` (FrozenVLM).

## Splits
- Development: 16 episodes. Used for pipeline checks and to freeze the analysis.
- Held-out: 48 episodes. Run once, after the analysis is frozen.

## Analyses
- Learning curves aligned to reversals, P(follow currently-better reporter) by trial since reversal.
- Cognitive models fit to per-trial soft choice probabilities (cross-entropy). See `cogmodels.py`.
  - M1: delta-rule fidelity estimate q with a single alpha, plus softmax beta and bias.
  - M2: separate alpha for the stable and volatile phases. This is the key test.
  - M3: asymmetric alpha+ and alpha- for confirming versus disconfirming outcomes.
  - M4: a Bayesian ideal observer, either a change-point/HMM model with an inferred hazard rate or the Behrens-style volatile learner, as the normative reference.
  - Baselines: win-stay/lose-shift, count-all (Beta), and last-k window.
- Parameter recovery on synthetic agents.
- The estimation-versus-use gap: judgment P(A) against choice P(follow A), compared trial by trial.

## Non-claims
- No claim of brain-like mechanism, neural homology, or human alignment.
- Any comparison with humans is qualitative, against published Behrens results.

## Intervention arm (added 2026-11-11, following the topic_scout.md recommendation: merge rather than switch)
- **Controller** (`controller.py`):
  - Runs Bayesian online change-point detection (Adams & MacKay 2007, Beta-Bernoulli, hazard 1/15) on the observed feedback stream.
  - It is training-free and black-box.
  - It is the normative analogue of Nassar et al. (2010, 2012): change-point probability sets the learning rate, an LC-NE "unexpected uncertainty" signal (Yu & Dayan 2005).
- **History modes** (`run_volatile.py --history-mode`):
  - `full`: the full history (diagnosis baseline).
  - `window:K`: a fixed recency window, K in {5, 10, 20}.
  - `cp_hard`: show only the trials since the inferred change point.
  - `cp_soft`: keep the full history and insert a change marker.
- **Claim we are testing:** the controller raises adaptation after reversals and the volatile-phase learning rate without hurting the stable phase. We compare it against the best fixed window, which is oracle-tuned.
- **Practical transfer (if time permits):** a non-stationary Oxford-Pet in-context stream.
