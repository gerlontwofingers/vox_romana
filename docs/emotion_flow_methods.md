# Cross-Lingual Emotion Flow Analysis: Methods

**Version:** 1.0 · **Package:** `vox_romana.emotion_flow` · **Last updated:** 2026-04-23

This document specifies the mathematical and algorithmic procedures implemented
in the `emotion_flow` pipeline. Every formula below corresponds directly to a
function in the codebase; cross-references are given as `module.function`.

---

## 1. Problem statement

Given a corpus of speeches $\mathcal{S} = \{s_1, \ldots, s_M\}$ attributable to a
single speaker $L$ (the *leader*), potentially drawn from multiple source
languages, we estimate:

1. A **first-order Markov transition matrix** $P$ over a fixed emotion alphabet
   $\mathcal{E}$ describing the speaker's local affective dynamics.
2. A **null-hypothesis rejection statistic** $\rho$ testing whether the observed
   transitions are structurally different from a memoryless permutation of the
   same emotion tags.
3. A **topic–emotion coupling coefficient** $r$ measuring whether semantic
   drift co-varies with affective drift across consecutive segments.

The analysis is **segment-level**: each speech $s$ is decomposed into an
ordered sequence of paragraph-scale segments, each independently classified.

### Emotion alphabet

$$
\mathcal{E} = \{\text{anger}, \text{disgust}, \text{fear}, \text{joy},
\text{neutral}, \text{sadness}, \text{surprise}\}, \qquad |\mathcal{E}| = N = 7
$$

Defined in `models.EMOTION_LABELS`. This alphabet matches the seven-class
output of the Ekman-plus-neutral classifier used downstream (§3.2).

---

## 2. Segmentation and language normalisation

### 2.1 Segmentation (`loader.FlexibleLoader`)

Input is one of: a `.txt` file, a directory of `.txt` files (one per speech),
or a `vox_romana` JSON file containing `events[]`, `segments[].events[]`, or
`chunks[].events[]`.

Text files are split on consecutive double-newlines (`"\n\n"`); JSON inputs are
split on event boundaries where `event.kind ∈ {utterance, narration, speech}`.
Each candidate segment $x$ is retained iff $|x| \geq c_\text{min}$ characters
(default $c_\text{min} = 20$).

The resulting ordered sequence is denoted

$$
X = (x_1, x_2, \ldots, x_T), \qquad x_t \in \mathcal{X}^*
$$

where $\mathcal{X}^*$ is the set of Unicode strings. Each segment carries a
`speech_id` so that cross-speech boundaries can be excluded in §3.3.

### 2.2 Translation to English (`translator.TranslatorAgent`)

For each segment $x_t$ let $\ell_t$ be its detected source language, estimated
by `langdetect.detect`. If $\ell_t = \text{en}$, the segment is passed
through unchanged with confidence $c_t = 1$.

Otherwise, let $f_{\ell \to \text{en}}$ and $f_{\text{en} \to \ell}$ denote
the forward and backward translation models (Helsinki-NLP opus-mt variants:
`Helsinki-NLP/opus-mt-{src}-en` and `Helsinki-NLP/opus-mt-en-{src}`). We
compute:

$$
\tilde{x}_t = f_{\ell_t \to \text{en}}(x_t), \qquad
\hat{x}_t = f_{\text{en} \to \ell_t}(\tilde{x}_t)
$$

**Translation confidence** is BERTScore F1 between the original and the
round-trip back-translation:

$$
c_t =
\begin{cases}
0.9 & \text{if } \ell_t \in \{zh, ja, ko, ar, hi\} \quad \text{(no reliable back-translation)} \\
F_1^{\text{BERT}}(x_t, \hat{x}_t) & \text{otherwise}
\end{cases}
$$

A segment is *flagged* (but retained) when $c_t < \tau$, with default
$\tau = 0.85$ (`BERT_FLAG_THRESHOLD`). The aggregate
$\bar{c} = \frac{1}{T}\sum_t c_t$ is reported alongside the final results as
the `back_translation_agreement` field on `EmotionFlowReport`.

The translation-to-English step exists because the downstream classifier (§3.2)
is monolingual English. All subsequent analysis operates on $\tilde{x}_t$.

---

## 3. Emotion tagging and sequence construction

### 3.1 Emotion classifier

Implemented in `tagger.EmotionTaggerAgent`. The model is
`j-hartmann/emotion-english-distilroberta-base`, a DistilRoBERTa classifier
fine-tuned on seven English emotion corpora (six Ekman emotions plus neutral).

For each segment $\tilde{x}_t$ the classifier emits a probability vector
$\mathbf{p}_t \in \Delta^{N-1}$, i.e. $\sum_{e \in \mathcal{E}} p_t(e) = 1$ and
$p_t(e) \geq 0$.

The **dominant tag** is

$$
\hat{e}_t = \arg\max_{e \in \mathcal{E}} p_t(e)
$$

Both the full distribution $\mathbf{p}_t$ and the dominant tag $\hat{e}_t$ are
stored on the `EmotionTag` dataclass (`models.EmotionTag`).

### 3.2 Emotion sequence

The ordered dominant-tag sequence

$$
E = (\hat{e}_1, \hat{e}_2, \ldots, \hat{e}_T) \in \mathcal{E}^T
$$

is the primary object of study. Speech boundaries are preserved via `speech_id`
so that transitions *across* speeches can be excluded (§4.1).

---

## 4. Transition matrix estimation

### 4.1 Count matrix (`transition.TransitionBuilderAgent.build`)

Let $\mathbf{C} \in \mathbb{Z}_{\geq 0}^{N \times N}$ be the raw count matrix
with

$$
C_{ij} = \sum_{t=1}^{T-1}
\mathbb{1}\{\hat{e}_t = e_i\}\cdot
\mathbb{1}\{\hat{e}_{t+1} = e_j\}\cdot
\mathbb{1}\{\text{speech\_id}(t) = \text{speech\_id}(t+1)\}
$$

The final indicator implements `skip_cross_speech=True` (default): a pair
$(\hat{e}_t, \hat{e}_{t+1})$ straddling a speech boundary contributes zero.
This prevents a spurious "endings → openings" pattern from dominating the
estimate when $M \gg 1$.

The total number of counted transitions is
$n_\text{trans} = \sum_{ij} C_{ij}$.

### 4.2 Add-$k$ smoothing

With smoothing constant $k > 0$ (default $k = 0.1$, Laplace-like):

$$
\tilde{C}_{ij} = C_{ij} + k, \qquad
P_{ij} = \frac{\tilde{C}_{ij}}{\sum_{j'} \tilde{C}_{ij'}}
$$

$\mathbf{P}$ is the row-stochastic smoothed transition matrix reported as
`TransitionMatrix.matrix`. Smoothing ensures $P_{ij} > 0$ everywhere, which is
required for the entropy rate (§5.3) to be finite and for the chi-square
expected frequencies (§5.1) to avoid zero divisors.

### 4.3 Wilson-score confidence intervals

Per-cell 95% confidence intervals are computed from the **raw** counts
$C_{ij}$, not the smoothed counts. Let $n_i = \sum_j C_{ij}$ be the row total
and $\hat{p} = C_{ij}/n_i$.

With $z = 1.9599639845400536 = \Phi^{-1}(0.975)$:

$$
\text{centre} = \frac{\hat{p} + z^2/(2n_i)}{1 + z^2/n_i}, \qquad
\text{margin} = \frac{z\sqrt{\hat{p}(1-\hat{p})/n_i + z^2/(4n_i^2)}}{1 + z^2/n_i}
$$

$$
\text{CI}_{ij}^{95\%} = \big[\max(0, \text{centre} - \text{margin}),\;
                             \min(1, \text{centre} + \text{margin})\big]
$$

When $n_i = 0$ (the emotion never appeared as the source of a transition),
the CI degenerates to $[0, 1]$ — maximum uncertainty. See
`transition._wilson_ci`.

Wilson score intervals are preferred over normal-approximation ("Wald")
intervals because they have correct coverage near $\hat{p} = 0$ and
$\hat{p} = 1$, which arise regularly for rare emotions.

---

## 5. Null-hypothesis testing

All tests are implemented in `tester.NullHypothesisTesterAgent`. The overall
reject criterion is conjunctive:

$$
\text{is\_structured} = \big(p_{\chi^2} < 0.05\big)\;\wedge\;
                        \big(p_{G} < 0.05\big)\;\wedge\;
                        \big(p_{\text{perm}} < 0.05\big)
$$

This three-way requirement is deliberately conservative: a single marginal
test failure does **not** license the claim that the transitions are
non-random.

### 5.1 Pearson $\chi^2$ test of independence

Under the null hypothesis of **row-column independence**, expected cell
frequencies are $E_{ij} = (n_i n_{\cdot j})/n_{\cdot\cdot}$ where
$n_i = \sum_j C_{ij}$, $n_{\cdot j} = \sum_i C_{ij}$, $n_{\cdot\cdot} = \sum_{ij} C_{ij}$.

$$
\chi^2 = \sum_{i,j} \frac{(C_{ij} - E_{ij})^2}{E_{ij}}
\;\sim\; \chi^2_{(N-1)^2} \quad \text{under } H_0
$$

Computed via `scipy.stats.chi2_contingency` with zero-row protection: any row
summing to zero is replaced by $10^{-10}$ before the call, preventing a NaN
return value without materially altering the statistic.

### 5.2 G-test (log-likelihood ratio)

Identical contingency test with the likelihood-ratio statistic:

$$
G = 2 \sum_{i,j} C_{ij}\, \ln\frac{C_{ij}}{E_{ij}} \;\sim\; \chi^2_{(N-1)^2}
$$

Computed by passing `lambda_="log-likelihood"` to `chi2_contingency`. The G-test
is included because it is known to have better small-sample behaviour than
Pearson's $\chi^2$ when expected frequencies are low — a common situation for
the rarer emotion cells (*disgust*, *surprise*).

### 5.3 Entropy rate

Let $\mathbf{w} = (w_i)$ with $w_i = n_i / \sum_{i'} n_{i'}$ be the empirical
marginal distribution over emotion rows. The weighted entropy rate is:

$$
H(\mathbf{P}) = \sum_{i} w_i \cdot H(P_i\,|\,i)
= -\sum_{i} w_i \sum_{j} P_{ij} \ln P_{ij}
$$

with the convention $0 \ln 0 = 0$ enforced numerically by adding $10^{-300}$
before the logarithm (`tester._entropy_rate`). Low $H$ indicates predictable,
structured transitions; high $H$ indicates nearly-uniform (random) transitions.

### 5.4 Permutation test

For $B$ permutations (default $B = 1000$) draw $E^{(b)} \sim \text{Unif}(\pi(E))$
— a uniformly random permutation of the dominant-tag sequence — using
`numpy.random.default_rng(seed)`. The **sequence is shuffled**, not the matrix,
so the permutation distribution preserves the emotion *marginals* but destroys
temporal ordering (true first-order Markov information).

For each permutation rebuild the count matrix and compute a $\chi^2$
statistic via the fast path `tester._chi2_statistic_only` (no p-value
computation). Let $\chi^2_{(b)}$ denote the result.

$$
p_\text{perm} = \frac{1}{B}\sum_{b=1}^{B}
\mathbb{1}\{\chi^2_{(b)} \geq \chi^2_\text{obs}\}
$$

The same permutations also yield an empirical null distribution of entropy
rates $\{H_{(b)}\}$, from which we report:

- **null mean:** $\bar{H}_\text{null} = \frac{1}{B}\sum_b H_{(b)}$
- **95% CI:** $[Q_{2.5}(\{H_{(b)}\}), Q_{97.5}(\{H_{(b)}\})]$

If $H_\text{obs} < Q_{2.5}$, the observed sequence is more predictable than
chance. This entropy-based framing complements the frequency-based
$\chi^2$/G-test framing and is the primary visualisation in
`visualizer.plot_entropy_rate`.

### 5.5 Cramér's V (effect size)

$$
V = \sqrt{\frac{\chi^2}{n_\text{trans} \cdot (\min(N, N) - 1)}}
= \sqrt{\frac{\chi^2}{n_\text{trans} \cdot (N-1)}}
$$

$V \in [0, 1]$. Cohen's rough guidelines (df = $N-1 = 6$): $V < 0.1$ is
negligible, $V \approx 0.2$ is weak, $V \approx 0.35$ is moderate, $V > 0.45$
is strong. Reported regardless of significance so readers can distinguish
"statistically significant but tiny" from "statistically significant and
substantively meaningful".

---

## 6. Topic–emotion coupling

Implemented in `semantic.SemanticFlowAgent`.

### 6.1 Semantic embeddings

Each translated segment $\tilde{x}_t$ is embedded into $\mathbb{R}^d$ via
`vox_romana.analysis.embedder.CorpusEmbedder.embed_texts`, yielding
$\mathbf{v}_t \in \mathbb{R}^d$. The default backend is a
sentence-transformers model with TF-IDF fallback; for testing,
`_embed_fn` may be injected to return deterministic vectors.

### 6.2 Shift sequences

Let $D(\mathbf{a}, \mathbf{b}) = 1 - \frac{\langle \mathbf{a}, \mathbf{b} \rangle}
{\|\mathbf{a}\|\,\|\mathbf{b}\|}$ be cosine distance (clipped to $[0, 2]$).
When either input has zero norm, $D$ returns 1 by convention.

- **Topic shift sequence:** $u_t = D(\mathbf{v}_t, \mathbf{v}_{t+1})$ for
  $t = 1, \ldots, T-1$.
- **Emotion shift sequence:** $v_t = D(\mathbf{p}_t, \mathbf{p}_{t+1})$ over
  the 7-dimensional probability vectors.

Both sequences have length $T-1$.

### 6.3 Pearson correlation

$$
r = \frac{\sum_t (u_t - \bar{u})(v_t - \bar{v})}
        {\sqrt{\sum_t (u_t - \bar{u})^2}\sqrt{\sum_t (v_t - \bar{v})^2}}
$$

Computed via `scipy.stats.pearsonr`, which also returns a two-sided p-value
under the null of zero correlation. When either $u_t$ or $v_t$ is constant
(zero variance), `pearsonr` returns NaN; we coerce this to $(r, p) = (0, 1)$
so downstream consumers can reason about the result without special cases.

### 6.4 Interpretation

- $r > 0$ and significant: topics and emotions shift together — "when the
  subject changes, the feeling changes."
- $r \approx 0$: affective modulation is decoupled from topical modulation —
  the speaker's emotion trajectory is not driven by what they are discussing.
- $r < 0$ and significant: an unusual anti-pattern — topical continuity
  coincides with affective change, or vice versa.

The pipeline requires $T \geq 3$ (so there are at least two shift pairs) to
compute $r$; otherwise $(r, p) = (0, 1)$ is returned with an empty
`topic_shifts` / `emotion_shifts`.

---

## 7. Deterministic seeding

All stochastic components are seeded from a single integer `seed` (default
$42$). Invoked before any agent that draws random numbers:

- `random.seed(seed)` (Python stdlib)
- `numpy.random.seed(seed)` (global NumPy RNG)
- `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`,
  `torch.backends.cudnn.deterministic = True`,
  `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- The permutation test (§5.4) uses its own `numpy.random.default_rng(seed)`
  stream for independence from global state.

See `reproducibility.freeze_randomness`. Non-determinism can still arise from:

1. Non-deterministic CUDA kernels that only support `warn_only` mode (we
   log a warning but proceed);
2. Model-download variance (a different checkpoint hash yields different
   embeddings); the reproducibility manifest records installed package
   versions to make this failure mode observable (§8).

---

## 8. Reported quantities

Every run produces an `EmotionFlowReport` (`models.EmotionFlowReport`)
containing:

| Field                          | Symbol / source                  | Type                         |
| ------------------------------ | -------------------------------- | ---------------------------- |
| `leader_name`                  | $L$                              | `str`                        |
| `n_speeches`                   | $M$                              | `int`                        |
| `n_segments`                   | $T$                              | `int`                        |
| `dominant_cycle`               | RLE of $E$                       | `list[str]`                  |
| `transition_matrix.matrix`     | $\mathbf{P}$                     | `np.ndarray (N, N) float32`  |
| `transition_matrix.count_matrix` | $\mathbf{C}$                   | `np.ndarray (N, N) int32`    |
| `transition_matrix.ci_low` / `ci_high` | Wilson CI                | `np.ndarray (N, N) float32`  |
| `null_test.chi2_p` / `g_test_p`| $p_{\chi^2}$, $p_G$              | `float`                      |
| `null_test.permutation_p`      | $p_\text{perm}$                  | `float`                      |
| `null_test.cramers_v`          | $V$                              | `float`                      |
| `null_test.entropy_observed`   | $H_\text{obs}$                   | `float`                      |
| `null_test.entropy_random_mean`| $\bar{H}_\text{null}$            | `float`                      |
| `null_test.entropy_ci`         | $[Q_{2.5}, Q_{97.5}]$ of $H_{(b)}$ | `tuple[float, float]`      |
| `null_test.is_structured`      | conjunctive rejection criterion  | `bool`                       |
| `semantic_flow.pearson_r`      | $r$                              | `float`                      |
| `semantic_flow.p_value`        | $p_r$                            | `float`                      |
| `back_translation_agreement`   | $\bar{c}$                        | `float`                      |
| `n_flagged_segments`           | $\sum_t \mathbb{1}\{c_t < \tau\}$ | `int`                       |
| `manifest`                     | environment snapshot             | `ReproducibilityManifest`    |

The "dominant cycle" is the run-length-encoded deduplication of $E$
(`pipeline._dominant_cycle`): consecutive repeats are collapsed. For
$E = (\text{neutral}, \text{neutral}, \text{joy}, \text{joy}, \text{anger})$
the cycle is $(\text{neutral}, \text{joy}, \text{anger})$. This is a
compact qualitative summary used by the show adapter (`show_adapter.py`) to
generate a storyboard.

---

## 9. Complexity and limitations

### 9.1 Complexity

Let $T$ be the number of segments, $B$ the number of permutations, $d$ the
embedding dimension, $N = 7$.

- Translation: $O(T)$ model forward passes (two per segment if
  $\ell \neq \text{en}$).
- Emotion tagging: $\lceil T/b \rceil$ forward passes where $b$ is the batch
  size (default 32). Dominates wall-clock time.
- Transition matrix build: $O(T + N^2)$.
- Chi-square / G-test: $O(N^2)$ once.
- Permutation test: $O(B \cdot T)$ matrix rebuilds + $O(B \cdot N^2)$ stats.
- Semantic flow: $O(T \cdot d)$ for embeddings, $O(T)$ cosine distances,
  $O(T)$ Pearson $r$.
- **Dominant cost:** classifier inference. For $T = 10^3$ and batch
  size 32, roughly 32 forward passes.

### 9.2 Known limitations

1. **First-order Markov assumption.** Higher-order dependencies (emotion
   patterns spanning 3+ segments) are not modelled. A second-order extension
   is a natural follow-up; the count-matrix data structure would become
   $N \times N \times N$.
2. **Segment granularity ambiguity.** "Paragraph-scale" segmentation is an
   editorial artefact of the source documents. Fine-grained segmentation
   (per-sentence) would shift $T$ upward and likely increase measured
   transition entropy. The segmentation choice should be reported in any
   published figure caption.
3. **Classifier domain mismatch.** `j-hartmann/emotion-english-distilroberta-base`
   is trained on short social-media and review text. Transfer to political
   oratory, literary prose, or technical exposition has not been
   independently validated. Transition *patterns* are still meaningful even
   if absolute per-class accuracy degrades, because the downstream statistics
   depend on the *stability* of the classifier's decision boundary, not its
   ground-truth accuracy.
4. **Translation confidence floor.** Back-translation BERTScore is a proxy,
   not a ground-truth measure of translation quality. Segments below
   $\tau = 0.85$ are flagged but retained; downstream analyses may therefore
   include noisy tags. A conservative analysis drops flagged segments and
   re-runs; the difference in conclusions is a useful sensitivity probe.
5. **Single-speaker framing.** The pipeline pools all speeches attributed to
   one leader. Inter-speaker comparisons require running the pipeline
   separately per leader and comparing `EmotionFlowReport` objects; no joint
   model is implemented.

---

## 10. References

- Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley. —
  Chi-square and G-tests, Cramér's V.
- Hartmann, J. (2022). "Emotion English DistilRoBERTa-base." HuggingFace
  model card. — Seven-class emotion classifier.
- Tiedemann, J., & Thottingal, S. (2020). "OPUS-MT — Building open
  translation services for the World." *EAMT 2020*. — Helsinki-NLP models.
- Wilson, E. B. (1927). "Probable inference, the law of succession, and
  statistical inference." *JASA*, 22(158), 209–212. — Score interval.
- Zhang, T., et al. (2020). "BERTScore: Evaluating Text Generation with
  BERT." *ICLR 2020*. — Translation-quality proxy.
