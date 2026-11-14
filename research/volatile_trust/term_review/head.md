You are a careful technical copy-editor for a paper submitted to the NeuroAI 2.0 workshop at AAAI-27 (readers: computational neuroscientists, cognitive scientists and ML researchers; 8-page main text, double-blind). The full LaTeX source is below.

## Task: terminology consistency only
Find every place where terminology is inconsistent. Specifically:
1. **Synonymy (one concept, several terms)**: the same thing is called by different names (e.g. "advisor" vs "reporter" vs "source"; "episode pair" vs "pair"; "held-out" vs "test"; "learner" vs "reference" vs "observer"; "count rule" vs "count"; "kernel" vs "integration kernel"; "following probability" vs "choice probability"; "outcome label" vs "outcome"; "trial" vs "day"), where a reader could think two different things are meant.
2. **Polysemy (one term, several concepts)**: the same word is used for different things (e.g. "summary" for the input condition vs. a summary rule vs. a summary statistic; "accuracy" of the advisor vs. accuracy of the model; "raw" as condition vs. ordinary adjective; "correct"/"correctness"; "rule"; "history"; "format"; "readout"; "reference"; "gap"; "phase"/"block"; "volatile"/"volatility"; "chance").
3. **Condition names**: the input conditions are typeset with macros \Raw{}, \RawCorrectness{}, \CorrectnessOnly{}, \Summary{} (small caps). Flag places where a condition is referred to in plain words instead of the macro, where the macro is used for something that is not that VLM input condition (e.g. for the count rule, or for a Summary *variant* without saying so), or where a macro is used where the plain-language concept is meant.
4. **Defined-before-used and abbreviations**: terms or abbreviations used before definition, defined twice differently, or defined only in the appendix but used in the main text (e.g. BOCPD, TOST, WSLS, kappa, h, phi, y_t, "episode pair", "counterbalanced copy", "reversal", "block").
5. **Notation and units**: inconsistent symbols, percent vs proportion, "points" vs "percentage points", model-name spellings (e.g. Qwen3-VL-8B vs Qwen3-VL 8B), British vs American spelling mixed (behavior/behaviour, colour/color, modelling/modeling, analysed/analyzed).

Do NOT comment on scientific content, claims, statistics, structure or length, and do not propose new analyses. Do not change any number.

## Output format
First, a short list (max 8 lines) of the canonical terms you recommend (term -> meaning).
Then a numbered list of issues. For each issue give:
- **Type** (synonymy / polysemy / condition-name / definition / notation-spelling)
- **Severity** (high = a reader could misread a result; medium = noticeable inconsistency; low = cosmetic)
- **Where**: section or caption, and an EXACT verbatim quote from the source (short, unique, copyable, including LaTeX as it appears)
- **Fix**: the exact replacement text for that quote
If the same fix applies at many places, say "all occurrences" and give the find/replace pair, plus how many occurrences you counted.
Rank issues by severity. Be exhaustive for high and medium; group low-severity cosmetic items compactly.
