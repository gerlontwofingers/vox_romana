# Cross-Lingual Emotion Flow in Political Oratory: Paper Outline

**Working title (draft):**
*Mapping the Affective Dynamics of Political Speech: A Cross-Lingual,
Reproducible Pipeline for Emotion-Flow Analysis*

**Intended venue tier:** computational social science or NLP+society
(e.g. EMNLP Findings, *Political Analysis*, *PNAS Nexus*, *Digital
Scholarship in the Humanities*).

**Length target:** 8–10 pages main text, unlimited appendix.

This outline is a **writing plan**, not a paper. Every section includes the
empirical claim it must defend, the specific artefact(s) from the
`vox_romana.emotion_flow` pipeline it cites, and the failure mode most
likely to sink it in review. Companion documents:
[emotion_flow_methods.md](./emotion_flow_methods.md),
[emotion_flow_repro.md](./emotion_flow_repro.md).

---

## 0. Abstract (≤ 250 words)

Four sentences:

1. **Gap:** Cross-lingual emotion analyses of political speech typically
   translate once and report aggregate class frequencies; the *sequence* of
   affective states — the flow — is discarded.
2. **Contribution:** We release a reproducible pipeline that estimates a
   first-order Markov transition matrix over a seven-class emotion alphabet
   for a given speaker, with Wilson-score per-cell confidence intervals,
   a conjunctive $\chi^2$/G-test/permutation rejection criterion, and a
   topic–emotion coupling coefficient (Pearson $r$ on cosine shift
   sequences).
3. **Result:** Applied to $k$ leaders across $m$ source languages, we find
   [stable / speaker-specific / language-invariant / language-dependent —
   **to be filled from actual runs**] structure in $x$ of $k$ cases; in the
   structured cases, Cramér's $V$ ranges over $[V_\text{min}, V_\text{max}]$.
4. **Why it matters:** Flow-level structure is invisible to bag-of-emotion
   analyses; it is stable under seed and permutation perturbations
   (Tier 3 reproducibility, see §7); it exposes rhetorical patterns
   (which emotion follows which) that aggregate class counts cannot.

Keywords: emotion flow, political rhetoric, cross-lingual NLP, Markov
analysis, reproducibility.

---

## 1. Introduction (≈ 1 page)

### 1.1 Motivating vignette

Short (150-word) opening on a *specific* speech moment where the *sequence*
of emotions carries meaning beyond the *set* of emotions.

> Example frame: "A speech that begins in *neutral* and shifts to *fear*
> at its midpoint signals something categorically different from one that
> begins in *fear* and shifts to *neutral* — even when the bag of emotions
> is identical."

### 1.2 The specific gap

Two paragraphs. Prior cross-lingual affective analyses:

- Aggregate: per-speech or per-speaker emotion *histograms*.
- Lexicon-based: monolingual, English-only, or per-language dictionaries with
  incomparable semantics across languages.
- Sentiment-only: collapses the seven-class affective palette into ±1,
  destroying rhetorical texture.

**What is missing:** a cross-lingual, *flow*-aware, *reproducible* estimator
with explicit uncertainty quantification and null-hypothesis testing against
the right null (memoryless permutation of the same tags).

### 1.3 Contributions

Enumerated, 4–5 items:

1. A reproducible pipeline (`vox_romana.emotion_flow`) estimating a
   first-order Markov transition matrix with Wilson-score CIs from
   translated-to-English paragraph-scale segments.
2. A conjunctive three-test null-rejection criterion
   ($\chi^2 \wedge G \wedge \text{perm}$) with an entropy-rate companion
   visualisation.
3. A topic–emotion coupling coefficient measuring whether semantic drift
   predicts affective drift.
4. A released dataset of $k$ leaders × $m$ source languages with
   pre-translated segments, emotion tags, and report artefacts, suitable
   for third-party replication.
5. A show-generation side product — the same `EmotionFlowReport` drives
   an explanatory audio narrative (§6.4) — demonstrating that the
   statistical output is communicable to a non-expert audience.

### 1.4 What we do *not* claim

Defensive paragraph. We do not claim: ground-truth emotion labels (we
measure classifier-induced transitions, not psychological states);
generalisation beyond the translated-English domain; any causal account
of *why* a given speaker exhibits a given pattern. Review-proofing.

---

## 2. Related work (≈ 0.75 page)

Three compact paragraphs, each ending with the specific handle the present
paper pulls that the cited line does not.

### 2.1 Emotion classification in text

Ekman (1992) → crowd-annotated corpora (GoEmotions, CARER) → fine-tuned
transformers (Hartmann 2022 DistilRoBERTa). **Handle:** we use the
classifier as a measurement instrument; our results depend on its
*stability*, not its ground-truth accuracy (§5.4).

### 2.2 Markov models of discourse

Conversational turn-taking Markov chains (Jurafsky & Martin), topic
chains (Chen et al.), stance transitions (Sobhani et al.).
**Handle:** prior work rarely applies rigorous null-hypothesis testing;
none combine $\chi^2$, G, and permutation into a conjunctive criterion
with explicit entropy-rate CIs.

### 2.3 Cross-lingual political text analysis

Comparative Manifesto Project, Wordfish/Wordshoal, multilingual-BERT
classifiers applied to party platforms. **Handle:** these produce
aggregate ideology scores; our target is *within-speaker* affective
dynamics, orthogonal to position estimation.

---

## 3. Data (≈ 0.5 page)

### 3.1 Corpus

Table 1: per-leader corpus statistics.

| Leader | Language(s) | # speeches $M$ | # segments $T$ | Year range |
| --- | --- | --- | --- | --- |
| *to fill* | *to fill* | *to fill* | *to fill* | *to fill* |

Acquisition provenance (source archives, licences) and an honest estimate
of representativeness (what corpus biases have we inherited from the
source archives?).

### 3.2 Segmentation choices

State and defend:

- Paragraph-scale segmentation ($c_\text{min} = 20$ characters).
- Retention of translation-flagged segments ($c_t < 0.85$) with
  sensitivity re-run dropping them (§7).
- `skip_cross_speech=True` so that end-of-speech → start-of-next does
  not contribute to $\mathbf{C}$.

### 3.3 Languages handled

List source languages and their routing:
- Back-translation pair available: `de`, `fr`, `es`, `it`, `pt`, `nl`,
  `ru`, `pl`, … (all OPUS-MT pairs we invoke).
- Fixed-confidence pass-through: `zh`, `ja`, `ko`, `ar`, `hi` (no reliable
  back-translation model; confidence pinned at 0.9).

---

## 4. Methods (≈ 1 page, pointer-heavy)

Summarise at publication density, defer full derivations to
[emotion_flow_methods.md](./emotion_flow_methods.md). Recommended
sub-structure:

- §4.1 Translation and confidence filtering (BERTScore F1).
- §4.2 Emotion tagging (DistilRoBERTa, 7 classes).
- §4.3 Transition matrix: counts → add-$k$ smoothing → row-normalised
  $\mathbf{P}$; Wilson CIs from raw counts.
- §4.4 Null-hypothesis testing: conjunctive
  $p_{\chi^2} \wedge p_G \wedge p_\text{perm} < 0.05$.
- §4.5 Entropy rate and permutation CI.
- §4.6 Cramér's $V$ as effect size.
- §4.7 Topic–emotion coupling via Pearson $r$ on cosine-shift sequences.

One figure: pipeline DAG (loader → translator → tagger → transition
builder → tester / semantic flow → reporter). Reuse the architecture in
[emotion_flow_methods.md §1](./emotion_flow_methods.md) rather than
redrawing.

---

## 5. Results (≈ 2 pages, 3–5 figures)

Present results in *increasing* interpretive commitment. Every figure must
pair a number with a confidence interval or p-value.

### 5.1 Headline statistics

Table 2: per-leader `(T, V, p_chi2, p_G, p_perm, is_structured, r_topic-emotion)`.

Column interpretation footnoted in the caption, not in the body text.

### 5.2 Transition heatmaps

Figure 1: heatmap of $\mathbf{P}$ for each structured leader. Use the
`visualizer.plot_transition_heatmap` output. Annotate cells where
$\text{CI}_\text{low} > 1/N = 0.143$ (significantly above uniform) and
cells where $\text{CI}_\text{high} < 1/N$ (significantly below). These
"above/below uniform" cells are the *interpretable* transitions.

### 5.3 Dominant cycles

Figure 2: run-length-encoded cycle per leader, rendered as a ribbon.
The qualitative summary a non-technical reader can remember.

### 5.4 Entropy rate vs null

Figure 3: observed $H_\text{obs}$ as a horizontal line plotted against
the permutation null distribution with 95% CI band. Cases where
$H_\text{obs} < Q_{2.5}$ are the *predictable* rhetorics; cases on the
null band are *unpredictable* (high entropy).

### 5.5 Topic–emotion coupling

Figure 4: scatter of $(u_t, v_t)$ with Pearson $r$ and $p_r$ in the
legend, one panel per leader. Tests §6.1 hypothesis.

### 5.6 Effect size vs significance

Figure 5 (optional appendix): Cramér's $V$ vs $-\log_{10} p_\text{perm}$.
Flags the "statistically significant but tiny effect" regime, which is
common with large $T$.

---

## 6. Discussion (≈ 1 page)

### 6.1 Hypothesis board

Write the hypotheses **before** looking at the numbers; mark which each
result supports, contradicts, or leaves open. Candidates:

- H1: Speakers with media-coached rhetoric show structured flow
  (low entropy rate).
- H2: Topic-emotion coupling ($|r| > 0.3$) is higher for speakers
  discussing a narrow set of themes vs wide-ranging orators.
- H3: Cross-language leaders (multilingual speeches) show *more* entropy
  than same-language leaders, even after translation — residual
  cross-lingual drift in the classifier.

### 6.2 What *patternedness* means

Structured ≠ manipulative. A speaker with high $V$ may be authentically
consistent, rhetorically trained, or stylistically fixed. The pipeline
reports *observable regularity*; inferring *intent* is out of scope.

### 6.3 Limitations

Four items, each one paragraph:

1. First-order Markov assumption (cf. methods §9).
2. Classifier domain mismatch for oratory vs social media training data.
3. Segmentation dependence ($c_\text{min}$, paragraph vs sentence).
4. Translation-induced artefacts — particularly for the no-back-translation
   languages.

### 6.4 Narrative delivery

Brief note: the same `EmotionFlowReport` feeds `run_emotion_flow_show.py`
to produce an audio explainer. A one-paragraph pointer and a URL to a
sample episode; full description of the show adapter is not in scope for
the methods paper.

---

## 7. Reproducibility (≈ 0.5 page)

Summarise from [emotion_flow_repro.md](./emotion_flow_repro.md) in three
paragraphs:

- **Three fidelity tiers** (byte-exact, bit-stable, conclusion-stable) and
  our guarantee of Tier 3 out of the box, Tier 2 with pinned packages,
  Tier 1 on CPU only.
- **Reproducibility manifest**: seed, Python / package / model versions,
  hardware, git commit, timestamp, deterministic-mode flag. Attached to
  every reported figure as a sidecar JSON.
- **Sensitivity probes actually run**: seed variation, permutation count
  (5000), flagged-segment drop, $c_\text{min}$ doubling. Confirm that
  qualitative conclusions survive all of the above.

Include a table mapping each figure in §5 to the specific input file,
seed, and git commit. Non-optional.

---

## 8. Conclusion (≤ 0.25 page)

Three sentences. What we built, what it found, what it lets others do.

---

## 9. Ethics statement (required at most CS venues)

- **Data:** public political speeches, no human-subjects review required.
- **Risk:** automated emotion classification can be misused to pathologise
  speakers; the pipeline is released with this caveat prominent in its
  README. We explicitly discourage use of per-cell emotion probabilities
  as character assessments.
- **Dual-use:** rhetorical-pattern detection could assist automated
  propaganda generation. We release the *analysis* pipeline only; no
  generative capability is bundled (the show adapter converts *findings*
  into narration, not *raw speeches* into mimicry).

---

## 10. Data and code availability

- Code: `github.com/<org>/vox_romana`, MIT-licensed, tagged
  `v1.0-emotion-flow`.
- Corpus: archived at [Zenodo / OSF] with DOI. Each speech carries its
  source-archive citation and licence.
- Per-figure reproducibility manifests: supplementary JSON bundle on the
  same archive.

---

## 11. Appendix (unlimited)

### A. Full formal derivations

Reference [emotion_flow_methods.md](./emotion_flow_methods.md) §§4–6.

### B. Hyperparameter sensitivity tables

For each leader in the main corpus: results at
$k \in \{0.0, 0.1, 0.5, 1.0\}$, $c_\text{min} \in \{10, 20, 40, 80\}$,
$B \in \{1000, 5000, 10000\}$. Qualitative-conclusion-stability matrix.

### C. Translation confidence breakdown

Per-language BERTScore distributions, fraction flagged, and results
before vs after dropping flagged segments.

### D. Worked example

One full speech walked end-to-end: source text → segments → translated
segments → emotion tags → transition counts → smoothed matrix →
null-test output → cycle visualisation. Pedagogical.

### E. Software environment

The full `pip freeze` output from the environment that produced the
main-text figures. Complements the abridged `package_versions` in the
manifest.

### F. Show adapter output

Sample `NewsConfig` JSON and one generated show script (truncated to a
single segment), demonstrating the report-to-narrative transform without
claiming it as a scientific result.

---

## Writing schedule (target: 6 weeks)

| Week | Milestone                                                          |
| ---- | ------------------------------------------------------------------ |
| 1    | Lock corpus, run full pipeline on all leaders, archive manifests   |
| 2    | Build §5 figures; freeze the headline table                        |
| 3    | Draft §§1–4 (introduction → methods)                               |
| 4    | Draft §§5–6 (results → discussion)                                 |
| 5    | Draft §§7–10 + appendix; internal review                           |
| 6    | Revisions, colour-blindness check on all figures, final submission |

---

## Review-killer checklist

Every reviewer tries to kill a paper. Pre-empt:

- [ ] A reviewer says "the classifier is off-domain." → §5.4 sensitivity,
      appendix C, cite Hartmann 2022 limitations.
- [ ] A reviewer says "you haven't tested against a sensible null." →
      §4.4, conjunctive three-test, §5.4 entropy CI.
- [ ] A reviewer says "translation drift dominates." → §3.3 language
      routing, §4.1 BERTScore filtering, appendix C per-language
      breakdown.
- [ ] A reviewer says "first-order Markov is too weak." → §6.3 limitation
      acknowledged, future work.
- [ ] A reviewer says "not reproducible." → §7, full manifest, sidecar
      JSON per figure, tier-2 recipe.
- [ ] A reviewer says "effect sizes are tiny." → §5.6 Cramér's $V$ vs
      significance plot; we distinguish rather than hide.
- [ ] A reviewer says "ethics?" → §9.
- [ ] A reviewer says "why not LLM-based emotion annotation?" → robustness
      (classifier is stable across runs; LLM annotations are not unless
      seeded), cost, and the reproducibility-tier argument (§7).

If all eight are answered before submission, we have shipped a defensible
paper.
