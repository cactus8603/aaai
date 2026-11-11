# Topic scout: alternatives to the Volatile Trust lead
Written 2026-11-11, 9 days before the 2026-11-20 deadline. No GPUs were used and no other files were changed.

## 0. Venue facts (checked 2026-11-11 against https://neuroai-multimodal-workshop.github.io/)
- **Theme:** context-dependent learning, adaptive intelligence, and continual learning. The workshop is held with AAAI-27 in Montreal, Feb 22-23, 2027.
- **Listed topics:** continual learning, catastrophic forgetting, context-dependent adaptation, multimodal representation learning, brain-inspired architectures, neural data analysis, neuromorphic computing, and biologically grounded evaluation of AI.
- **Listed biological principles:** cortical microcircuits, E/I balance, sparse/modular representations, dendritic nonlinearities, context-sensitive plasticity. Open benchmarks bridging neuroscience and AI are also invited.
- **Formats:** short papers of 4 pages are non-archival; long papers of 8 pages are archival. Submission is via OpenReview, double-blind, using the AAAI-27 kit.
- **Speakers:** Blake Richards, Doina Precup, Guillaume Bellec, Guillaume Lajoie, Ida Momennejad, Markus Marks, Paul Cisek, Priya Panda.
  - Precup, Momennejad and Richards work on RL, cognitive maps and adaptive behaviour, so they are a natural audience for learning-rate and volatility work.
  - Cisek works on context and affordance-based decisions. Lajoie and Bellec work on learning rules and plasticity.
- **1st edition (PMLR v308, 19 papers):** mostly brain-inspired architectures and neural-data or alignment papers. Examples:
  - CATFormer (continual spiking transformer)
  - Hierarchical predictive processing for uncertainty
  - fMRI-aligned fine-tuning
  - EEG-steered attention
  - Visual-cortex digital twins

  None of them studies in-context adaptation of pretrained VLMs, so the lead's niche is open at this venue. The bar is moderate: several accepted papers had small or near-chance experiments (see `research/source_trust/workshop_benchmark_audit.json`).

## 1. Novelty threats to the lead found in this search (must be cited)
| Work | What it does | Gap the lead still fills |
|---|---|---|
| Wang, Ward & Zhang, *Comparative reversal learning reveals rigid adaptation in LLMs under non-stationary uncertainty*, arXiv 2604.04182 (Apr 2026, IPMU 2026) | Text LLMs on a probabilistic reversal task, comparing a fixed schedule with a random (more volatile) one. Hierarchical RL fits find perseveration and weak loss learning. | They compare whole conditions, not a within-episode stable vs volatile learning rate (the Behrens signature). There are no VLMs, no image format, no source reliability, no estimation-vs-use analysis, and no intervention. **This is the closest prior work.** |
| Zhang, Yang & Wang, *LLMs as Discounted Bayesian Filters*, arXiv 2512.18489 (Dec 2025) | LLM belief updates fit an exponential-forgetting filter with a fixed discount factor. Uses shifting dice and GP parameters. Proposes prompt recalibration. | A fixed discount amounts to a fixed learning rate. Whether the rate adapts to volatility is exactly the lead's question. |
| Schubert et al., *In-context learning agents are asymmetric belief updaters*, arXiv 2402.03969 | Asymmetric alpha+ and alpha- in in-context bandits. | Relevant to the lead's M3 model; it does not address volatility. |
| Rahn et al., *Entropic activation steering*, arXiv 2406.00244 | Steers exploration of in-context bandit agents. | Shows that intervening on in-context RL is accepted practice, but it does not target learning rate or volatility. |
| Zeng & Wang, *Abstract representational geometry supports inference in LLMs*, arXiv 2606.23345 (Jun 2026) | Contextual reversal task (Bernardi et al. 2020 hippocampus) with humans vs LLMs. Finds hippocampal-like geometry and uses regularisation interventions. | Covers the latent-context and geometry angle, so we should not pivot there. |

Conclusion: the lead's core is still novel. Its weakness for this CFP is criterion (b): as designed it is a diagnosis only. That is the gap the merge option below fixes.

---

## 2. Candidates

### Candidate A (MERGE, recommended): add a volatility-gated "neuromodulatory" context controller to the lead
**Claim:** frozen VLMs use a nearly fixed, sticky in-context learning rate. A training-free controller that sets how much history the model attends to recovers the Behrens volatility signature and improves accuracy after a change, both on the advisor task and on a practical non-stationary multimodal in-context stream. The controller is modelled on locus coeruleus / norepinephrine (LC-NE) signalling of unexpected uncertainty and is driven by the model's own surprise.

**Neuroscience anchor:**
- Behrens et al. 2007 (Nat Neurosci) and 2008 (Nature): ACC tracks volatility, and learning rate rises when the environment is volatile.
- Yu & Dayan 2005 (Neuron): acetylcholine signals expected uncertainty and norepinephrine signals unexpected uncertainty.
- Aston-Jones & Cohen 2005: adaptive gain theory of LC-NE.
- Nassar et al. 2010 (J Neurosci) and 2012 (Nat Neurosci): a reduced Bayesian delta rule in which change-point probability sets the learning rate, with pupil-linked arousal.
- Liakoni et al. 2021 (Neural Comput, arXiv 1907.02936): Bayes-factor surprise sets the learning rate.

All of these sit under the CFP's "context-sensitive plasticity" heading.

**Experiment (reuses the lead's code, episodes and readouts):**
1. **Surprise signal.** Use the model's own per-trial prediction error, e.g. -log P(revealed truth) from the choice readout. This makes the method black-box and needs no access to the model internals.
2. **Controller ("NE gain").** A 2-parameter reduced-Bayesian change-point filter in the style of Nassar 2010. It runs on the surprise stream, outputs a change-point probability Ω_t and run length r_t, and sets the effective history shown to the VLM:
   - Arm A1 (hard gate): show the last ceil(r_t) trials. This is like an event boundary / hippocampal segmentation.
   - Arm A2 (soft gate): annotate history with recency salience, such as a "[before the change]" marker or collapsing old trials into a summary count.
   - Arm A3 (white-box, optional, only if time): add an attention-logit bias, slope λ·Ω_t times trial age, from the query to history-trial tokens. This is an ALiBi-like gain modulation through a 4-D float mask in eager attention on Qwen2.5-VL-3B.
3. **Baselines:**
   - full history (the lead)
   - fixed last-k windows, with k from 5 to 40; the best fixed k is an oracle-tuned baseline
   - a prompt hint such as "recent trials matter more"
   - an exponential-discount summary
   - the M4 Bayesian ideal observer as an upper bound
4. **Metrics:**
   - fitted alpha_stable vs alpha_volatile (the Behrens signature; does the controller create it?)
   - perseveration after a reversal
   - accuracy over 10-trial windows
   - area under the recovery curve
   - the estimation-vs-use gap
5. **Practical transfer task (makes the result useful for AI systems):** a non-stationary multimodal in-context classification stream on Oxford-IIIT Pet. Either an annotator's reliability switches (noisy-label stream), or the label mapping reverses (cat→"A"/dog→"B" swaps every 10-30 trials). Both arms use image queries. Report the gain over full-context ICL and over the best fixed window, and also show that the controller does not hurt a stationary control stream.

**Compute and time:**
- The lead's per-model dev run already exists in `runs/dev_*`.
- Held-out set: 48 episodes × 80 trials × (2 domains × 2 formats), about 15k forward passes per arm per model. That is about 1 GPU-hour per arm for 3-4B models at batch 1 on a 3090.
- Totals: 4 arms × 3 models ≈ 12 GPU-hours, plus Pet transfer ≈ 6 GPU-hours. This fits in 1 day on 4 GPUs.
- Implementation: about 1.5 days for the controller and transfer task (A1 and A2 are prompt-side only), plus 0.5 day for A3 if attempted.

**Closest prior work and the difference:**
- 2512.18489: prompt recalibration of a fixed discount. Here the controller is adaptive and driven by surprise and volatility, as in neuroscience.
- 2604.04182: diagnosis only, text only.
- Changepoint-aware world models, arXiv 2609.18950: trims an RL replay buffer at change points. It acts on weights and replay, not in-context; the mechanism is analogous and should be cited.
- Surprise-gated robot episodic memory, arXiv 2606.03787: gates memory writes, not forgetting or learning rate.
- Searches found no paper that sets in-context history or attention gain in a VLM from surprise (see §4).

**Risk:**
- (i) Some models may already show a learning-rate difference between stable and volatile phases, which would weaken the "fixed learning rate" story. The intervention result still holds.
- (ii) A tuned fixed window may match the controller on the advisor task. The Pet transfer task and the stationary control test robustness, since a controller adapts where a fixed window cannot be tuned.
- (iii) Reviewers may call A1 "just truncation". Frame it explicitly as a normative change-point model in the Nassar style, applied to the context window, and add A3 if feasible.

**Fit: 9/10.** It covers context-dependent adaptation, a specific paradigm, a specific neuromodulatory computation, a multimodal model, and a practical intervention.

---

### Candidate B: change-point vs oddball, testing whether VLMs know when surprise should not drive learning
**Claim:** normative learners raise their learning rate after surprise when the environment has change points, but lower it after surprise when surprises are one-off oddballs. We test whether frozen VLMs show this sign flip (Nassar et al. 2019 eLife), and whether a context-gated controller restores it.

**Neuroscience anchor:**
- Nassar, Bruckner & Frank 2019 (eLife 8:e46975): the P300 predicts more learning in the change-point context and less learning in the oddball context. Data are on Dryad (doi:10.5061/dryad.570pf8n) and code is on GitHub (learning-memory-and-decision-lab/NassarBrucknerFrank_eLife_2019). This gives real human behavioural and EEG data.
- d'Acremont & Bossaerts 2016 (leptokurtic vs Gaussian noise).
- Yu & Dayan 2005.

**Experiment:**
- Task: the helicopter / predictive-inference task. The outcome is a position from 0-300.
  - Change-point condition: Gaussian noise around a mean that jumps with hazard h.
  - Oddball condition: a drifting mean with occasional uniform outliers.
- Formats: text numbers vs an image (a rendered line with a bag marker). This is the multimodal factor.
- Readout: greedy constrained 3-digit generation. Compute the per-trial learning rate = update / prediction error, and regress it on surprise × condition, exactly as in the human analysis.
- Comparisons: humans (Dryad behaviour), the reduced Bayesian observer, and a fixed-alpha model.
- Optional neural link: correlate a model-internal surprise proxy (NLL, or hidden-state change at the outcome token) with update size. Test whether its coupling flips sign across contexts, as the P300 does in humans.
- Intervention: the Candidate A controller, extended with an oddball branch (Nassar 2019 normative model).

**Compute and time:**
- 2 conditions × 24 sequences × 200 trials × 2 formats ≈ 19k generations per model, about 2-3 GPU-hours per 4B model.
- Writing the task and analysis takes about 2 days. Parsing the Dryad data takes about 1 day.

**Closest prior work and the difference:**
- 2512.18489: change in parameters only, no oddball condition, fixed discount.
- bioRxiv 2025.03.13.642273, *Neurocomputational underpinnings of suboptimal beliefs in RNN agents*: trained RNNs on the helicopter task, not pretrained (V)LMs in-context.
- Searches found no pretrained LLM or VLM study of change-point vs oddball.

**Risk:**
- Small VLMs may give poor numeric regression from images. Mitigate by running text first and using images as a secondary factor.
- A whole new task pipeline with 9 days left is tight.
- It overlaps conceptually with the lead, which is a strength if merged and redundant if run separately.

**Fit: 8/10.** It has a sharper normative falsifier than the lead and real human data, but no continual-learning or fine-tuning angle.

---

### Candidate C: Flesch-style context-dependent continual fine-tuning of a VLM with XdG / Hebbian context-gated LoRA
**Claim:** when two rules that depend on a context cue are learned in sequence on the same images, standard LoRA on a VLM forgets the first rule. This matches the network side of Flesch et al. 2018, where humans benefit from blocked training and networks do not. Context-dependent gating of LoRA units restores a blocked-training advantage like the human one, and produces the orthogonal task geometry seen in human and macaque prefrontal cortex. The gating is either XdG-style random masks or gates learned with Hebbian context.

**Neuroscience anchor:**
- Flesch et al. 2018 (PNAS), *Comparing continual task learning in minds and machines*.
- Flesch et al. 2022 (Neuron), *Orthogonal representations for robust context-dependent task performance in brains and neural networks*, which uses human fMRI and the Mante et al. 2013 macaque data.
- Flesch et al. 2023 (PLoS CB, arXiv 2203.11560), Hebbian context gating with decaying task signals.
- Masse et al. 2018 (PNAS), XdG.
- These map onto the CFP's "sparse/modular representations" and "context-sensitive plasticity".

**Experiment:**
- Stimuli:
  - Synthetic stimuli with two continuous features, e.g. hue × orientation or tree leafiness × branchiness.
  - A natural variant: Oxford Pet images with an overlaid continuous colour-frame feature, where context selects which feature defines "plant / don't plant".
- Curricula: blocked (A then B) vs interleaved.
- Model: Qwen2.5-VL-3B with LoRA on the LLM blocks.
- Methods:
  - LoRA
  - LoRA + 1% replay
  - O-LoRA
  - XdG-LoRA, a fixed random 50-80% mask over LoRA rank and MLP units per context
  - Hebbian-gated LoRA
- Metrics:
  - forgetting of rule A
  - accuracy on rule B
  - the Flesch "factorised vs linear" choice-pattern model fits
  - RSA of hidden states against the orthogonal-plane model RDM
  - retention of general VQA ability (MMBench or ScienceQA-img subset)

**Compute:**
- About 1-1.5 GPU-hours per run: 3B model, roughly 8k images, 2 epochs.
- 5 methods × 2 curricula × 3 seeds ≈ 30 runs ≈ 40 GPU-hours, about 1.5 days on 6 GPUs.
- About 2 days of engineering.

**Closest prior work and the difference:**
- NeuroLoRA (arXiv 2603.12378, Mar 2026): a neuromodulation-inspired context gate in MoE-LoRA, text LLMs only.
- GainLoRA (2025), JumpLoRA (arXiv 2604.16171), and expander random masks (ICLR 2026 submission).
- MoE-Adapters for CLIP continual learning (arXiv 2403.11549).
- ProgLoRA and BranchLoRA (ACL 2025) for multimodal continual instruction tuning.
- Russin et al. (arXiv 2402.08674): blocking advantage from in-context learning in LLMs, interleaving advantage from in-weight learning.
- The difference: a VLM, the Flesch paradigm with its geometry readout, and a direct mapping to human curriculum data. The gated-LoRA method itself is **not** novel.

**Risk:**
- The gated-adapter continual-learning space is crowded, so reviewers may see "yet another gated LoRA".
- The synthetic task may be too easy, giving no forgetting or a ceiling.
- It is the furthest from existing code, so it is the highest schedule risk.

**Fit: 7.5/10.** It hits the continual-learning and brain-inspired-architecture keywords directly, but novelty is the weakest of the three.

---

## 3. Considered and rejected
- **Instruction or context effects on fMRI alignment (NSD/THINGS):** already done in *Correlating instruction-tuning (in multimodal models) with vision-language processing (in the brain)* (ICLR 2025, arXiv 2505.20029) and *Task-conditioned probing of instruction-tuned MLLMs* (arXiv 2506.08277). NSD also needs a data-use agreement. The novelty margin is low and the practical value weak.
- **WCST / set shifting in VLMs:** Hao, Alexandre & Yu (arXiv 2505.22112) and *Set-shifting behavioral test for harnessed agents* (arXiv 2607.13396).
- **Latent-cause / contextual-reversal inference in LMs:** Zeng & Wang (arXiv 2606.23345, Jun 2026) covers behaviour, geometry and interventions.
- **Mante-Sussillo context-dependent integration inside VLMs:** no prior VLM work found, but there is no practical arm and the interpretability pipeline is costly for 9 days.

## 4. Searches run (2026-11-11, web search over arXiv/OpenReview/ACL)
- "LLM in-context learning rate volatility reversal learning Behrens": found 2604.04182, 2402.03969, 2607.03660 (effective context length, not volatility).
- "volatility / unexpected uncertainty LLM in-context adaptive learning rate change-point": nothing direct (2603.10299 is about financial volatility).
- "LLM belief updating surprise changepoint vs outlier": found 2512.18489, 2503.04722, 2610.29331, 2605.06915. None has an oddball condition.
- "LLM helicopter task change-point oddball Nassar": only RNN agents (bioRxiv 2025.03.13.642273).
- "surprise-gated context forgetting / prompt reset, non-stationary LLM": found 2609.18950 (RL replay), 2606.03787 (robot memory), 2608.02464 (monitoring). None operates in-context.
- "surprise-modulated attention temperature / neuromodulation transformer inference-time": found 2511.01292 (optimal temperature under shift, theory), 2511.06818 (learned focal temperature), 2305.03232 (neuromodulation gated transformer, trained). None is driven by surprise at inference.
- "activation steering in-context RL learning rate": found 2406.00244, 2509.24923, 2604.05859.
- "context-dependent gating / neuromodulation LoRA continual (multimodal)": found 2603.12378, 2604.18075, 2604.16171, 2607.23837, 2403.11549, ProgLoRA / BranchLoRA (ACL 2025), 2511.01831, 2511.15164.
- "blocked vs interleaved Flesch transformer fine-tuning": found 2402.08674 and 2203.11560. There is no VLM version.
- "latent cause / task-set switching LLM": found 2606.23345.
- "WCST VLM": found 2505.22112 and 2607.13396.
- "instruction context brain alignment NSD/THINGS": found 2505.20029 and 2506.08277.

Caveat: some hits were judged from the abstract only. Before writing related work, read the full texts of 2604.04182 and 2512.18489.

## 5. Ranked recommendation
1. **MERGE: the lead plus Candidate A (do this).**
   - Keep the lead's frozen analysis and held-out run unchanged.
   - Add the surprise-driven change-point controller arms A1 and A2, plus the Pet non-stationary transfer stream. The title becomes something like "Volatile trust: VLMs under-adapt their in-context learning rate, and a neuromodulation-inspired change-point controller fixes it".
   - This turns the lead from a diagnosis into diagnosis plus intervention, which meets criterion (b). The extra compute is under 20 GPU-hours.
   - Pre-register the controller hyperparameters on the dev split before the held-out run.
2. **Optional add-on, days 5-7 only if A is done: a text-only change-point vs oddball sub-experiment from Candidate B** (one figure). It tests whether the controller needs to know the context type, and it brings in human Dryad data. Drop it if anything slips.
3. **Do not switch to C.** It is the best keyword match but has the weakest novelty and the highest schedule risk with 9 days left. Keep it as a follow-up for a long or archival paper.

**Suggested schedule:**
- Day 1-2: controller and Pet stream, dev-tuned.
- Day 3: freeze; held-out runs for the lead and A arms on 3 models.
- Day 4-5: fits and figures.
- Day 6-7: Candidate B text sub-experiment, or the 7-8B models.
- Day 8-9: writing.
