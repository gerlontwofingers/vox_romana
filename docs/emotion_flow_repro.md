# Cross-Lingual Emotion Flow: Reproducibility Guide

**Version:** 1.0 · **Package:** `vox_romana.emotion_flow` · **Last updated:** 2026-04-23

This guide tells you how to reproduce any result produced by the
`emotion_flow` pipeline, from scratch, on a different machine, with identical
numeric output up to documented tolerances. It is the operational companion to
[emotion_flow_methods.md](./emotion_flow_methods.md).

---

## 1. What "reproducible" means here

We distinguish three tiers of reproduction fidelity. The pipeline targets
Tier 1 and Tier 2 by default; Tier 3 requires extra effort.

| Tier   | Name              | Guarantee                                                   | How achieved                     |
| ------ | ----------------- | ----------------------------------------------------------- | -------------------------------- |
| **T1** | Byte-exact        | Every float matches to full precision                       | Same hardware + same versions    |
| **T2** | Bit-stable        | All p-values and CI bounds agree to ≥ 6 decimal places       | Pinned packages, CPU execution   |
| **T3** | Conclusion-stable | All qualitative conclusions (is_structured, dominant cycle, tone bracket) identical | Seeded RNG, same data |

**Tier 3 is the minimum scientific requirement** for any published figure
made with this pipeline. Tier 1 is *aspirational* and usually unattainable
across CUDA versions; we explain why in §5.

---

## 2. The reproducibility manifest

Every run calls
[`reproducibility.build_manifest(seed, model_versions)`](../vox_romana/emotion_flow/reproducibility.py)
and attaches the result to the final `EmotionFlowReport` (on the `manifest`
field). The manifest is serialised into every output artefact (JSON, HTML,
and stdout summary) and contains:

```python
ReproducibilityManifest(
    seed: int,                       # RNG seed (e.g. 42)
    python_version: str,             # e.g. "3.12.10"
    package_versions: dict[str, str],# numpy, scipy, torch, transformers, ...
    model_versions: dict[str, str],  # HuggingFace model revisions
    hardware: str,                   # "Apple M2 Pro, MPS" / "NVIDIA RTX 4090, CUDA 12.1"
    timestamp: str,                  # ISO-8601 UTC
    git_commit: str,                 # short SHA of the vox_romana checkout
    deterministic_mode: bool,        # torch.are_deterministic_algorithms_enabled()
)
```

Tracked packages (see `reproducibility._TRACKED_PACKAGES`):

    numpy, scipy, scikit-learn, torch, transformers, bert-score,
    langdetect, chromadb, sentence-transformers, statsmodels

**To reproduce a published finding, the verifier must match every field
of the manifest up to the fidelity tier they are targeting.** If the
verifier cannot match a field (e.g. no CUDA GPU), they should note the
substitution in any derivative work.

---

## 3. Seeding every stochastic component

`freeze_randomness(seed)` is called at the top of
`EmotionFlowPipeline.run()` before any agent. It seeds:

- `random.seed(seed)` — Python stdlib RNG (not directly used in this
  pipeline, seeded defensively)
- `numpy.random.seed(seed)` — global NumPy state
- If `torch` is installed:
  - `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - `torch.use_deterministic_algorithms(True, warn_only=True)`

The **permutation test** (`tester.NullHypothesisTesterAgent`) uses an
*independent* PRNG stream via `numpy.random.default_rng(self._seed)`. This
is deliberate: it decouples the permutation null distribution from any
global-state mutations made by upstream components. If you change the
number of permutations, the matrix stays the same; if you change the seed,
the null distribution changes deterministically.

---

## 4. Step-by-step reproduction recipe

### 4.1 Prerequisites

- Python 3.12.x (pipeline is tested on 3.12.10; 3.12.x minor bumps are
  safe for Tier 2)
- Git
- ≥ 4 GB of free disk for HuggingFace model caches
- Optional but recommended: a CUDA 12.x GPU (or Apple Silicon with MPS) for
  acceptable wall-clock; CPU-only execution works but is roughly 10× slower
  for emotion tagging

### 4.2 Clone at the recorded commit

Read `manifest.git_commit` from the target report and check out that exact
revision:

```bash
git clone <repo-url> vox_romana
cd vox_romana
git checkout <short-sha>
```

### 4.3 Create an environment with pinned packages

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Tier 2 fidelity, pin each package in `manifest.package_versions`:

```bash
python -c "import json; d=json.load(open('target/report.json'))['manifest']['package_versions']; \
[print(f'{k}=={v}') for k,v in d.items() if v != 'not-installed']" \
  > pinned-versions.txt
pip install --force-reinstall -r pinned-versions.txt
```

### 4.4 Cache-lock the models

`emotion_flow` downloads three HuggingFace models on first use:

1. `j-hartmann/emotion-english-distilroberta-base` (emotion tagger)
2. `Helsinki-NLP/opus-mt-{src}-en` and `-en-{src}` (per source language)
3. A sentence-transformers model for `CorpusEmbedder`

Set `HF_HUB_OFFLINE=1` after the first run to prevent silent version
updates. Better: pass an explicit `cache_dir` and freeze the directory:

```bash
export HF_HUB_CACHE=/path/to/vox_romana/.hf_cache
python run_emotion_flow.py your_data.txt --cache-dir $HF_HUB_CACHE
```

Alternatively, pin by model revision at the code level; the
`ReproducibilityManifest.model_versions` dict is designed to be populated
with specific checkpoint hashes, which can then be passed through
`pipeline.model_versions=`.

### 4.5 Run with the recorded seed

```bash
python run_emotion_flow.py <source> \
    --leader "<Leader Name>" \
    --output output/emotion_flow/ \
    --seed <manifest.seed> \
    --permutations <n> \
    --min-chars 20
```

Defaults (seed=42, permutations=1000, min-chars=20) match the pipeline
defaults and will reproduce any report generated with those defaults
unless the manifest indicates otherwise.

### 4.6 Compare outputs

Diff the JSON reports numerically:

```python
import json
import numpy as np

ref = json.load(open("reference_report.json"))
new = json.load(open("output/emotion_flow/.../report.json"))

# Tier 3 check — qualitative stability
assert ref["dominant_cycle"] == new["dominant_cycle"]
assert ref["null_test"]["is_structured"] == new["null_test"]["is_structured"]

# Tier 2 check — 6-decimal numeric agreement
for k in ("chi2_p", "g_test_p", "permutation_p", "cramers_v"):
    np.testing.assert_allclose(
        ref["null_test"][k], new["null_test"][k], atol=1e-6
    )
np.testing.assert_allclose(
    ref["semantic_flow"]["pearson_r"], new["semantic_flow"]["pearson_r"],
    atol=1e-6,
)

# Tier 2 matrix check
ref_P = np.array(ref["transition_matrix"]["matrix"])
new_P = np.array(new["transition_matrix"]["matrix"])
np.testing.assert_allclose(ref_P, new_P, atol=1e-6)
```

---

## 5. Known sources of non-determinism

Even with identical seeds, identical pinned versions, and identical input
data, the following can cause numeric drift:

1. **CUDA non-determinism.** Some cuDNN and cuBLAS kernels have no
   deterministic implementation; `torch.use_deterministic_algorithms` is
   set with `warn_only=True` because hard-erroring would make several
   transformer operations unusable. Matrix multiplies can differ at the
   last few bits between GPU generations. Mitigation: run on CPU for
   Tier 1 / Tier 2 fidelity, accepting a wall-clock hit.
2. **BLAS backend variance.** Different BLAS libraries (OpenBLAS, MKL,
   Accelerate) produce slightly different floating-point reductions. The
   Pearson $r$ from `scipy.stats.pearsonr` and the $\chi^2$ statistic from
   `scipy.stats.chi2_contingency` are most sensitive. Mitigation: pin the
   scipy build variant (`pip install scipy==<v> --no-binary scipy` in
   extreme cases).
3. **Model checkpoint drift.** HuggingFace model pages occasionally update
   weights without changing the model id. The only fully defensive posture
   is to pin by revision hash, e.g.
   `AutoModel.from_pretrained("org/model", revision="<sha>")`. Record the
   resolved hash in `manifest.model_versions`.
4. **`langdetect` non-determinism.** `langdetect` uses a probabilistic
   algorithm and has its own internal RNG. For deterministic language
   detection, call `DetectorFactory.seed = 0` at process start. The
   current pipeline does not do this because the specific source-language
   label affects which translation model is loaded — if detection flips
   between runs, downstream results diverge. Call `DetectorFactory.seed = 0`
   explicitly in your entry script if you observe drift.
5. **Dictionary ordering for small-count ties.** `arg_max` over the emotion
   scores is implemented as `scores.get(label, 0.0)` iteration. Python's
   dict preserves insertion order, which is stable across CPython 3.7+, so
   this is *not* a practical source of drift. It would be if you swapped
   to an implementation that iterates in hash order.

---

## 6. Minimum viable reproduction record

Alongside every published figure, the paper/dataset/supplement should
include the following ten items. This is the smallest set that makes
Tier 3 reproduction feasible for a third party:

1. **Source file(s)** — the exact text/JSON used as input, or a URL with
   a content hash.
2. **Leader name filter** if the source is multi-speaker.
3. **Seed** (`--seed`).
4. **Permutation count** (`--permutations`).
5. **Minimum character filter** (`--min-chars`).
6. **Git commit short SHA** of `vox_romana`.
7. **Pinned `requirements.txt`** generated from the exact environment.
8. **HuggingFace model revisions** for the three models in §4.4.
9. **Hardware description** (`manifest.hardware`).
10. **The `EmotionFlowReport` JSON file** itself, so that readers can
    diff against their own reproduction.

Items 3–9 are captured automatically in `ReproducibilityManifest` and
serialised to JSON. Items 1, 2, and 10 must be archived by the author;
the pipeline does not version-control the input data.

---

## 7. Sensitivity probes

A single reproduced number is not a finding. Before publishing, re-run
with at least the following perturbations and confirm that all
*qualitative* conclusions survive:

| Perturbation                         | Expected behaviour                                   |
| ------------------------------------ | ---------------------------------------------------- |
| `--seed 43` (and 44, 45)             | `is_structured` stable; $p_\text{perm}$ within $\pm 0.01$; $r$ within $\pm 0.01$ |
| `--permutations 5000`                | $p_\text{perm}$ estimate tightens but does not cross 0.05 |
| Drop flagged segments (re-run with confidence filter) | Cramér's V within $\pm 0.05$ of original |
| `--min-chars 40` (doubled)           | $T$ decreases; dominant cycle topology stable       |
| Finer segmentation (per-sentence split before loading) | $T$ increases; entropy rate may increase; significance should survive if the effect is real |

If `is_structured` flips under any of these perturbations, treat the
original result as exploratory, not confirmatory.

---

## 8. Testing the pipeline itself

The test suite in `tests/test_emotion_flow_*.py` covers every agent with
injected mocks and a pinned seed. Run:

```bash
python -m pytest tests/test_emotion_flow_*.py -v
```

All tests must pass. The pipeline ships with ~290 tests across loader,
translator, tagger, transition, semantic, tester, visualizer, reporter,
pipeline, and show-adapter modules. Any contribution that changes numeric
output must also update the corresponding golden-value test.

---

## 9. Troubleshooting

**"Results differ on the 7th decimal place."**
Expected for Tier 3 environments. Verify that every quantity stripped to 6
decimals matches. If so, the run is a valid reproduction.

**"`is_structured` flipped from True to False."**
Check `cramers_v` in both runs. If $V$ is very close to its effective
threshold (the observed $V$ for which all three p-values cross 0.05), the
result is fragile by construction. Re-run with $B = 10\,000$
permutations; the `perm_p` estimate will stabilise. If the flip persists,
the original result was an edge case and should not be reported as
confirmed.

**"`pearson_r` is 0.0 and `p_value` is 1.0."**
This is the explicit fallback when `scipy.stats.pearsonr` returns NaN
(zero variance in one of the shift sequences). Usually means every
segment received the same forced emotion label — check that the emotion
tagger ran correctly and did not hit the `_force_label` testing path.

**"Wall-clock is 10× slower than expected."**
You are likely running on CPU. Check `manifest.hardware`. For large
$T$ (> 1000) consider GPU execution; the numeric output will match to
Tier 3 fidelity but no better.

**"`HF_HUB_OFFLINE=1` causes a model-load failure."**
Your local cache is missing at least one of the models listed in §4.4.
Run once with `HF_HUB_OFFLINE` unset to populate the cache, then re-enable
offline mode.
