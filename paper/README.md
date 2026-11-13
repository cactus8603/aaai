# Paper: Learning Whom to Trust from Feedback

This is a **long paper (8 pages plus references and supplementary material, archival in PMLR)** for the **NeuroAI 2.0 workshop at AAAI-27**.
- Workshop site: https://neuroai-multimodal-workshop.github.io/
- Deadline: **2026-11-20**
- Submission portal: OpenReview
- Review is double-blind.

## Layout
| Path | Contents |
|---|---|
| `main.tex` | Paper source, built from `AnonymousSubmission2027.tex` |
| `aaai2027.sty`, `aaai2027.bst` | Unmodified copies from the kit. **Never edit them.** |
| `refs.bib` | Verified BibTeX entries (from `../research/volatile_trust/literature.json` plus later additions) |
| `figures/` | Figures, pre-rendered as PDF, PNG or JPG. Crop them outside LaTeX. |
| `tables/` | Reserved for table fragments; current tables are inline in `main.tex` |
| `authorkit_original/` | The official AAAI-27 kit (`AuthorKit27.zip`, TemplateVersion 2027.1), downloaded from https://aaai.org/authorkit27/ and unzipped. Treat it as read-only reference material. |

## Compile
**Quick way:** `bash paper/build.sh` (from anywhere) compiles `main.pdf` with pdfLaTeX + BibTeX and prints the checks: the page on which References start (main text must fit in 8 pages), overfull boxes, undefined references, fonts that are not embedded or Type 3, and the number of open TODO / DRAFT / ?? markers. `bash paper/build.sh --clean` also removes the aux files. `main.pdf` is tracked so the draft can be previewed on GitHub; rebuild it with `build.sh` before committing paper changes.

To regenerate Figure 1 from the repository root, run
`.venv/bin/python research/volatile_trust/make_teaser.py` (Python with Matplotlib).
It reads the committed `research/volatile_trust/teaser_data.json` snapshot and writes
`figures/fig_teaser.pdf`, an editable-text SVG, and a 300-dpi PNG preview. The PDF is
exactly seven inches wide, with text at least 9pt at its included size. The snapshot
contains the five-model held-out results (48 episode pairs), source paths and SHA256 hashes; add
`--verify-sources` to compare every plotted value with the original local analysis
JSON files. Those larger analysis files are not required to redraw the figure.
The three prompt bubbles show the same past trial in separate history formats;
each model query receives its own history. The result panel directly labels the
equal-weight mean accuracy for each format. Its blue open diamond uses the same
models under all-trial-summary histories, rather than a standalone count rule.
Model-level accuracies and paired summary–raw confidence intervals appear in
Table 1; individual episode-pair bootstrap accuracy intervals are consolidated
with recovery ratios in Appendix C. Figure 1 does not display uncertainty intervals
for the model means.
Update the snapshot and caption together if the underlying analysis changes. PDF text is exported as vector outlines for the AAAI font requirements; the teaser SVG retains editable text. Ghostscript (`gs`) is required for PDF figure export.

AAAI-27 requires **pdfLaTeX**. `aaai2027.sty` aborts under XeTeX, LuaTeX or tectonic. TeX is installed user-locally through TinyTeX in `~/.TinyTeX`.

```bash
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
cd paper
latexmk -pdf main.tex        # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
latexmk -c                   # clean aux files
```
If a package is missing, install it with `tlmgr install <pkg>`. These were added on top of TinyTeX-1 (v2026.11): newtx, xpatch, fontaxes, placeins and caption, followed by `tlmgr update --all` and `fmtutil-sys --byfmt pdflatex`.
Tectonic 0.17 was also installed to `~/.local/bin/tectonic`, but it **cannot** build this paper because it is XeTeX-based and `aaai2027.sty` stops with "pdfTeX is required".

To check the PDF:
```bash
pdfinfo main.pdf | grep Pages
pdffonts main.pdf             # every font must be embedded ("emb yes"), and none may be Type 3
grep -n "Overfull" main.log   # must print nothing
```

## AAAI-27 and workshop formatting rules that matter
**Page limits (workshop CFP)**
- A long paper is at most **8 pages** of main content (the target since 2026-11-11). A short paper would be 4 pages.
- References and supplementary material do not count toward the limit and have no page limit.
- The main content runs from the Abstract through Discussion & Limitations. Any content appendix placed before the references counts toward the 8 pages.
- Section order:
  1. Main content
  2. (Ethical Statement)
  3. (Acknowledgments). Omit this for review.
  4. References
  5. Supplementary material (`\appendix`, after the references)

**Anonymity (double-blind)**
- Use `\usepackage[submission]{aaai2027}`. It removes the copyright block.
- The author must be "Anonymous Submission", and the affiliations must be empty.
- Leave out acknowledgments, grant numbers and identifying links. Any links must go in the `links` environment, use an anonymous host, and sit between the abstract and the main text.
- Cite your own prior work in the third person, or anonymize it.
- Strip the PDF metadata before uploading, for example with `exiftool -all= main.pdf` or `qpdf --linearize`. Keep `\pdfinfo` with only `/TemplateVersion (2027.1)`. Do not add `/Title` or `/Author`.

**Preamble.** Do not change the kit lines:
- `letterpaper` article
- `aaai2027` with the `submission` option
- `url[hyphens]`, `graphicx`, `\urlstyle{rm}` and `\UrlFont`
- `natbib` and `caption`, with no options
- `\frenchspacing`
- `\pdfinfo{/TemplateVersion (2027.1)}`

Do not add `\bibliographystyle`, because the style file sets it.

**Forbidden packages**
authblk, babel, balance, CJK, epsf, epsfig, euler, float, flushend, fullpage, geometry, graphics, hyperref, indentfirst, layout, lmodern, multicol, nameref, navigator, pdfcomment, pgfplots, psfig, pstricks, savetrees, setspace, stfloats, t1enc, tabu, times, titlesec, tocbibind, ulem, wrapfig. Any other font package or link-embedding package is also forbidden.

**Forbidden commands**
- `\nocopyright`, `\addtolength`, `\balance`, `\baselinestretch`, `\linespread`
- `\clearpage`, `\newpage`, `\pagebreak`, `\break`
- `\columnsep`, `\pagestyle`, `\textheight`, `\topmargin`, `\tiny`
- `\abovecaption`, `\belowcaption`, `\abovedisplay`, `\belowdisplay`
- `\float`, `\clip`, `\trim`
- `\setlength`. The one exception is `\setlength{\tabcolsep}{..}`.
- Negative `\vspace` or `\vskip` near captions, figures, tables or headings.
- `\resizebox` on tables.

**Fonts and text**
- Use the Times-like fonts that the style file loads.
- Body text is 10pt on 12pt leading. Do not shrink it.
- Embed all fonts. Type 3 fonts are not allowed, including inside figures.
- Do not add page numbers, headers or footers.
- Nothing may intrude into the margins, so there must be no overfull boxes.

**Figures**
- Use PDF, PNG or JPG files at 300 dpi or more. Do not use GIF or 72-dpi screenshots.
- Crop outside LaTeX. The `trim` and `clip` options are forbidden.
- Do not use pgfplots or TikZ plots in the source. Export plots to PDF instead.
- Text inside a figure must be at least **9pt**.
- Use color sparingly, with WCAG 2.0 contrast of at least 4.5:1.
- Floats go at the top of the page (`[t]`) where possible.

**Captions**
- The number and caption go **below** both figures and tables.
- Captions are 10pt roman. Do not make them smaller, bold or italic.
- Algorithm and listing captions are the exception: they go in the header.

**Tables**
- Use 10pt type, or 9pt if necessary.
- Use booktabs.
- If a table is too wide for one column, make it span both columns or split it.

**Title**
- The title is in mixed (title) case.
- Sections are unnumbered (`secnumdepth 0`), as the kit advises for short papers.

## Submission checklist
- [ ] Main content (Abstract through Discussion & Limitations) is **8 pages or fewer**. Check where the References heading begins.
- [ ] `grep -n "TODO\|todo{\|figplaceholder" main.tex` returns nothing. Remove the `\todo` and `\figplaceholder` macro definitions.
- [ ] All figures are real PDF or PNG files in `figures/`, with fonts embedded, text at least 9pt, and no Type 3 fonts (`pdffonts main.pdf`).
- [ ] Every caption sits below its figure or table, in 10pt roman.
- [ ] `refs.bib` is filled from `literature.json`, all entries have complete fields, and there are no `??` citations or BibTeX warnings about missing fields.
- [ ] The log shows no overfull boxes and no undefined references.
- [ ] No forbidden packages or commands. Run `grep -nE "hyperref|geometry|float\}|times|setspace|vspace\{-|newpage|clearpage|resizebox" main.tex`.
- [ ] `\pdfinfo` is unchanged, and `[submission]` is set.
- [ ] Anonymity: the author is "Anonymous Submission", there are no acknowledgments, links go to anonymous repositories, self-citations are in the third person, and figures, filenames and prompts contain no names or user paths.
- [ ] PDF metadata is stripped (`pdfinfo main.pdf` shows no author or creator identity).
- [ ] The paper uses US Letter paper and is a single PDF. Supplementary material comes after the references, or as a separate file if OpenReview asks for one.
- [ ] Claims stay within the non-claims in DESIGN.md: no brain-like mechanism and no human-alignment claims.
- [ ] Primary and secondary results follow the frozen plan; later-specified replications and exploratory diagnostics are identified separately.
- [ ] Upload to OpenReview before 2026-11-20 AoE. Double-check the workshop page for any form fields.

## Current analysis and synchronization workflow

Regenerate the six statistical/task figures from completed held-out outputs:

```bash
.venv/bin/python research/volatile_trust/make_figures.py --analysis-dir research/volatile_trust/analysis/h200_ho --split-label held-out --masking research/volatile_trust/analysis/h200_ho/masking/masking.json --masking-label held-out --api-analysis research/volatile_trust/analysis/api_behavior/api_behavior.json --skip-teaser
```

The figure canvas matches the final 7-inch text width or 3.3125-inch column width;
labels are at least 9pt. PDF text is outlined while all curves remain vector graphics.

Figure 1 preserves the full prediction--feedback loop, uses prompt bubbles for
the three trial-level history formats, and directly labels four model-mean
accuracies, including the all-trial-summary reference.
Table 1 carries the full-set, pair-averaged-logit results. Figure 2 shows only
per-copy scores on the matched API trial mask: one facet per history format and
seven shared model rows, with one confidence interval per row in each facet.
The full-set and matched-subset estimates remain separate. The former full-set
panel's 20 marginal accuracy CIs are preserved as interval-only columns in the
existing Appendix C recovery table, without duplicating the main means.
Figure 3 is a single-column correctness-integration kernel with four unchanged
10-lag curves. Table 2 replaces the former five masking matrices with five model
rows and four conditions grouped by outcome/report visibility. Correctness
statements remain visible in all four conditions. The 20 descriptive accuracy
means are unchanged. The 15 stored paired masking contrasts and their 95% CIs
remain in `fig_masking_effects.pdf` in Appendix I; they are not intervals for
absolute accuracy. Raw-history strategy comparisons remain in
`fig_strategies.pdf`; per-model reversal trajectories remain in
`fig_curves_models.pdf`. Development text/image/binding controls appear once in
a consolidated table; the generated `fig_controls.pdf` is retained as a research
asset but is no longer included in the reader PDF. The main accuracy table
reports percentages and percentage-point differences. Appendix C combines
secondary TOST tests and rule agreement, without repeating primary test results. Psych-101 hazard curves and
table medians use the declared sampled-choice estimator, consistently with the
paired comparisons; their pointwise CIs are unadjusted, and the figure states
that all five hidden-hazard comparisons fail the Holm threshold. Regeneration
does not rerun any model or change stored analysis results.

The review added three analyses using existing outputs only:

- `analyze_api_behavior.py`: genuine reversals only; separate correct-answer accuracy from hard-choice following of the scheduled-better advisor. Reversal trial has offset zero; windows cover offsets 0–2, 3–6 and 7–9.
- `analyze_readout_sensitivity.py`: per-copy choice accuracy on all 48 held-out episode pairs, alongside the original counterbalanced-logit estimator; no candidate-mass filtering.
- `analyze_reviewer_checks.py --staleness-only`: 2,000 participant-cluster bootstrap refits for stale-minus-fresh value weights and paired model–human contrasts. This updates the existing reviewer-check report.

Pending analyses remain part of the work plan. Empty result tables and unknown
GPU-hour columns are omitted from the reader PDF; pending work is described
explicitly and must not be treated as a failed or null experiment. In particular:

- Cognitive fits must use the corrected episode grouping: both cover stories of the same `(data, split, episode_id)` are left out together. Old development fits used 32 folds instead of 16 independent pairs; old held-out fits used 96 instead of 48. Both splits require corrected CPU fitting runs. `analyze_cogmodels.py` now applies this grouping, but old results cannot fill the corrected tables or support model-selection claims. Existing outputs were not overwritten.
- For the planned Thinking/Instruct comparison, `analyze_thinking.py --cross-checkpoint Qwen/Qwen3-VL-8B-Thinking Qwen/Qwen3-VL-8B-Instruct` works with `--direct`. Results are separate from same-checkpoint comparisons and cannot isolate reasoning from checkpoint/readout differences. The planned subset uses development episodes with text reports at trials 20, 40, 60 and 80, plus controls at 40 and 80, greedy generation up to 16,384 tokens. Report free-answer accuracy, valid-answer rate, truncation, forced answer-token readouts (used when no valid answer was generated) and token cost separately; these planned settings are not empirical support in the reader PDF.

### Appendix scope and layout (2026-11-13)

The supplement is organized around task reproducibility, alternative explanations,
readout validity, and limits of the human comparison. The primary result is shown
once; repeated control/strategy presentations, candidate-icon search logs, compact
prompt factorial logs and historical pilot effect sizes are omitted. The compact
prompt counterexample, unsuccessful equivalence tests, weak parameter identification,
chart-reading failures, Gemma readout failures, and Holm-null human comparisons are
retained. Pilot exposure and plan deviations remain explicit disclosures.

The task illustration now contains only the unique schedule/count-lag panel; the
icon figure shows original and selected inputs only. Both are native single-column
figures. Appendix double-column floats use standard `[!t]` placement, with late result
tables declared before their accompanying prose, to avoid an end-of-document
float queue; single-column tables use `[tbp]`, except the cover-story table, which
uses `[!ht]` to follow the appendix heading and its introduction. Raw analysis outputs and research assets are preserved;
removing their repeated presentation does not rerun or modify an experiment.

After compiling and checking the manuscript, `python3 paper/sync_overleaf.py overleaf`
copies the manuscript, bibliography and PDF figures into the Overleaf-linked checkout,
preserving its author/affiliation block and the anonymous submission switch. Fetch and
reconcile the Overleaf-linked repository before copying. Commit/push that checkout,
then use Overleaf's GitHub integration to pull the changes into the live editor;
GitHub synchronization is not automatic.

## Full-paper comparison (2026-11-13)

This review compared complete main texts and relevant methods, not only abstracts
or teaser figures. Lengths below are rough whitespace-token counts from PDF text
extraction, including titles, abstracts, captions, figure labels and equations,
but excluding references and appendices. Line wrapping, hyphenation and formula
extraction affect the counts; these are not exact prose word counts or sentence
statistics. The earlier compiled version of our main text was about 5,600 such
tokens. Use the current build, rather than that snapshot, to check the eight-page
limit.

| Paper and primary source | Approximate main-text scale | Main section sequence and relative focus | Measures and writing habits |
|---|---|---|---|
| Krishnamurthy et al., **Can Large Language Models Explore In-Context?** (NeurIPS 2024), [official PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/d951f73c521d069fefbb73396df01424-Paper-Conference.pdf) | 9 pages; 6,500 tokens; 35-page PDF | Introduction → Experimental Setup → Experimental Results → Discussion and Open Questions. Setup and results dominate (roughly 35% and 44%); results progress through failure diagnosis, successful configurations and root causes. | Reward and best-arm selection, supplemented by explicitly defined exploration failures and algorithmic baselines. Replicate-level uncertainty. Reuses *configuration*, *replicate*, *reward* and *failure*; a named failure has an observable definition. |
| Ge et al., **Mind the (DH) Gap! A Contrast in Risky Choices Between Reasoning and Conversational LLMs** (2026 preprint), [v2](https://arxiv.org/abs/2602.15173v2), [PDF](https://arxiv.org/pdf/2602.15173v2) | 9 pages; 5,800 tokens; 23-page PDF | Introduction, including related work → Research Design → Behavior and Convergence → Description–History Gap → Decision Explanations → Behavioral Models → Conclusion → Limitations. Design occupies approximately pages 2–4; results span pages 4–9. | Repeated-answer choice proportions; cross-context correlations, supplementary MSE and MSE-based behavioral fits. Uses *context*, *choice probability*, *correlation* and *consistency* to keep performance, response stability and model fit distinct. |
| Xue et al., **Counting on Thinking: Tracing Evidence Integration in Language Models** (2026 preprint), [v1](https://arxiv.org/abs/2609.32932v1), [PDF](https://arxiv.org/pdf/2609.32932v1) | About 9 pages plus a short page-10 tail; 5,300 tokens; 31-page, single-column PDF | Introduction → Methods (task, design, models, measures) → Results (integration weights, rereading/counting, feedback) → Discussion (direct answers, readout, costs, limitations and conclusion). Roughly 24%/18%/27%/30%. | Correct-answer negative log probability and regression-based integration weights; independent stimuli or conversation blocks are the sampling units. Some bands are across-model SEM, not CIs. Keeps *evidence*, *weight*, *count* and *readout* consistent; connects each result heading to a specific finding. |
| Zhang et al., **What Counts as Evidence? Epistemic Cue Utilization in Large Language Models** (2026 preprint), [v1 full text](https://arxiv.org/html/2610.12788v1), [PDF](https://arxiv.org/pdf/2610.12788v1) | Main argument through page 8; limitations/ethics on page 9; about 6,200 tokens before references; 26-page PDF | Introduction → Benchmark (design, tracks, measures, results) → Experiments (diagnosis, interventions) → Related Work → Conclusion → Limitations → Ethical Considerations. Most space goes to diagnostic contrasts. | Identification accuracy, signed matched-condition choice effects and verified accuracy; world-level bootstrap. Separates *identification* from *influence* and gives the contrast direction before interpreting its size. Unverified support choices have no unique gold answer. |
| Binz et al., **A foundation model to predict and capture human cognition** (Nature 2025), [published full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12390832/), [DOI](https://doi.org/10.1038/s41586-025-09215-4) | Main narrative about 7 pages/6,200 tokens; published pages 1002–1009 include references; separate Methods follow in the 22-page PDF | Introduction → Model Overview → Human Behavior → Generalization → Neural Alignment → Model-guided Discovery → Discussion → Conclusion → Methods. Several empirical questions receive separate sections. | Human-response NLL, simulated behavioral distributions and task-specific parameters, neural correlations, and AIC in the discovery example. Clearly separates prediction conditioned on participant histories from simulations driven by the model's own actions. Nature's separate Methods and extended data are not an eight-page workshop layout template. |

### Decisions applied to this manuscript

- Keep Introduction → Related Work → Task and Methods → Results → Discussion,
  with detailed protocols after the references. The comparison supports a focused
  behavioral paper; it does not call for a larger introduction or more metrics.
- Keep the core result sequence together: history-format effect → repetition
  diagnosis → masking → reliability judgment → computational reference → summary
  lag. Proprietary-model results now follow this explanation as a boundary
  condition. The former long mechanism paragraph is split into separate claims;
  the main masking account and Table 2 use all five held-out models.
- Define the outcome quota, higher-logit decision rule and half-credit ties in
  Methods. Introduce the Psych-101 participant-history replay and fitted
  forgetting-rate comparison there, before reporting the human comparison.
- Report accuracy and percentage-point changes in the main text. Keep the
  predeclared gap-normalized recovery ratio as an appendix diagnostic rather than
  presenting it as a standard score. Remove the old development cross-validation
  results and their main-text claims while corrected fits for both development
  and held-out episodes are pending.
- Cite the source-history identification/influence study and state our additional
  question: repeated reversals and the feedback components that trigger outcome
  repetition. Restrict the related-work comparison to bandit/reversal evaluations,
  rather than claiming every cognitive task concerns a model's own options.
- Use a finding → estimate/figure → interpretation sequence. Keep *history*,
  *report*, *outcome*, *correctness*, *choice*, *accuracy*, *episode* and
  *integration kernel* stable across sections. State supported findings directly;
  put a specific limitation beside the inference it constrains. No quantitative
  claim about sentence length or vocabulary frequency was used in this review.

### Metric choices and limits of comparison

Accuracy and the paired summary-minus-raw gap remain the primary outcomes because
this task has revealed correct outcomes. Kernels and Cohen's kappa answer different
diagnostic questions; they do not replace accuracy. Neither reward in an actively
explored bandit, correlation across repeated-choice contexts, nor NLL for predicting
human choices is interchangeable with our episode-level accuracy. Preserve the
episode pair as the independent unit, including both cover stories in each
cross-validation fold; do not copy another paper's response-level uncertainty.

The counterbalanced average-logit readout is a defined composite estimator, not a
free-generated choice. The added per-copy sensitivity analysis checks whether the
format gap survives a different readout. Soft cross-entropy is a valid predictive
loss for probability targets, but its absolute proximity to log(2) does not establish
an absence of tracking; compare improvement over the constant-bias baseline.
Integration kernels describe sensitivity and do not identify a unique internal
learning algorithm. The calibrated effective-learning-rate analysis remains a
descriptive appendix diagnostic because its positive control cannot reliably
resolve volatility adaptation on this schedule. TOST tests accuracy equivalence
within our prespecified two-percentage-point tolerance, not strategy identity or a
field-wide equivalence threshold. These distinctions motivate clearer reporting,
not an unplanned replacement of the primary analysis or new experiments.
