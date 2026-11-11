# Paper: Do Vision-Language Models Adapt Their Trust to Volatile Sources?

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
| `tables/` | Table bodies that `main.tex` pulls in with `\input` |
| `authorkit_original/` | The official AAAI-27 kit (`AuthorKit27.zip`, TemplateVersion 2027.1), downloaded from https://aaai.org/authorkit27/ and unzipped. Treat it as read-only reference material. |

## Compile
AAAI-27 requires **pdfLaTeX**. `aaai2027.sty` aborts under XeTeX, LuaTeX or tectonic. TeX is installed user-locally through TinyTeX in `~/.TinyTeX`.

```bash
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
cd paper
latexmk -pdf main.tex        # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
latexmk -c                   # clean aux files
```
If a package is missing, install it with `tlmgr install <pkg>`. These were added on top of TinyTeX-1 (v2026.11): newtx, xpatch, fontaxes, placeins and caption, followed by `tlmgr update --all` and `fmtutil-sys --byfmt pdflatex`.
Tectonic 0.17 was also installed to `~/.local/bin/tectonic`, but it **cannot** build this paper because it is XeTeX-based and `aaai2027.sty` stops with "pdfTeX is required".
While `refs.bib` is still empty, `main.tex` carries a temporary `\nocite{*}` so that BibTeX runs cleanly. Remove it once real `\cite` commands exist.

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
- [ ] Main content (Abstract through Discussion & Limitations) is **4 pages or fewer**. Check where the References heading begins.
- [ ] The temporary `\nocite{*}` above `\bibliography{refs}` is removed.
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
- [ ] Held-out results come from the single frozen-analysis run.
- [ ] Upload to OpenReview before 2026-11-20 AoE. Double-check the workshop page for any form fields.
