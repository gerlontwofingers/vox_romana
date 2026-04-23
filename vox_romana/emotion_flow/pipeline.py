"""EmotionFlowPipeline — end-to-end orchestrator for cross-lingual emotion analysis.

Chains seven agents + visualizer + reporter in a single call:

    FlexibleLoader
        ↓
    TranslatorAgent         (optional: --no-translate)
        ↓
    EmotionTaggerAgent
        ↓
    TransitionBuilderAgent
        ↓
    SemanticFlowAgent       (optional: --no-semantic)
        ↓
    NullHypothesisTesterAgent
        ↓
    EmotionFlowVisualizer   (via reporter)
        ↓
    EmotionFlowReporter     → HTML + JSON + CSV

QUICK START::

    from vox_romana.emotion_flow.pipeline import EmotionFlowPipeline

    pipeline = EmotionFlowPipeline(output_dir="output/emotion_flow/")
    report   = pipeline.run("speeches/caesar.txt", leader="Caesar")
    print(report.report_path)

All agents accept their usual constructor kwargs via ``agent_kwargs``
dictionaries on the pipeline constructor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .loader import FlexibleLoader
from .models import EmotionFlowReport, EmotionSequence
from .reporter import EmotionFlowReporter
from .reproducibility import build_manifest, freeze_randomness
from .tagger import EmotionTaggerAgent
from .tester import NullHypothesisTesterAgent
from .transition import TransitionBuilderAgent
from .translator import TranslatorAgent
from .semantic import SemanticFlowAgent
from .visualizer import EmotionFlowVisualizer

logger = logging.getLogger(__name__)


class EmotionFlowPipeline:
    """Full emotion-flow analysis pipeline.

    Args:
        output_dir:       If set, files are written here (HTML, JSON, CSV, plots).
        seed:             RNG seed applied via :func:`~.reproducibility.freeze_randomness`.
        cache_dir:        Shared HuggingFace model cache directory.
        n_permutations:   Permutation draws for the null-hypothesis test.
        run_semantic:     If ``False``, skip the semantic-flow correlation step.
        run_translation:  If ``False``, treat all segments as already English
                          (sets ``_force_passthrough=True`` on TranslatorAgent).
        loader_kwargs:    Extra kwargs forwarded to :class:`~.loader.FlexibleLoader`.
        translator_kwargs: Extra kwargs forwarded to :class:`~.translator.TranslatorAgent`.
        tagger_kwargs:    Extra kwargs forwarded to :class:`~.tagger.EmotionTaggerAgent`.
        transition_kwargs: Extra kwargs forwarded to :class:`~.transition.TransitionBuilderAgent`.
        semantic_kwargs:  Extra kwargs forwarded to :class:`~.semantic.SemanticFlowAgent`.
        tester_kwargs:    Extra kwargs forwarded to :class:`~.tester.NullHypothesisTesterAgent`.
        visualizer_kwargs: Extra kwargs forwarded to :class:`~.visualizer.EmotionFlowVisualizer`.
        reporter_kwargs:  Extra kwargs forwarded to :class:`~.reporter.EmotionFlowReporter`.
        model_versions:   Optional dict of ``{name: version}`` strings stored in the
                          :class:`~.models.ReproducibilityManifest`.
    """

    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        seed: int = 42,
        cache_dir: str | None = None,
        n_permutations: int = 1000,
        run_semantic: bool = True,
        run_translation: bool = True,
        loader_kwargs: dict[str, Any] | None = None,
        translator_kwargs: dict[str, Any] | None = None,
        tagger_kwargs: dict[str, Any] | None = None,
        transition_kwargs: dict[str, Any] | None = None,
        semantic_kwargs: dict[str, Any] | None = None,
        tester_kwargs: dict[str, Any] | None = None,
        visualizer_kwargs: dict[str, Any] | None = None,
        reporter_kwargs: dict[str, Any] | None = None,
        model_versions: dict[str, str] | None = None,
    ) -> None:
        self._output_dir = Path(output_dir) if output_dir else None
        self._seed = seed
        self._cache_dir = cache_dir
        self._n_permutations = n_permutations
        self._run_semantic = run_semantic
        self._run_translation = run_translation
        self._loader_kwargs: dict[str, Any] = loader_kwargs or {}
        self._translator_kwargs: dict[str, Any] = translator_kwargs or {}
        self._tagger_kwargs: dict[str, Any] = tagger_kwargs or {}
        self._transition_kwargs: dict[str, Any] = transition_kwargs or {}
        self._semantic_kwargs: dict[str, Any] = semantic_kwargs or {}
        self._tester_kwargs: dict[str, Any] = tester_kwargs or {}
        self._visualizer_kwargs: dict[str, Any] = visualizer_kwargs or {}
        self._reporter_kwargs: dict[str, Any] = reporter_kwargs or {}
        self._model_versions: dict[str, str] = model_versions or {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        source: str | Path,
        leader: str | None = None,
        *,
        min_chars: int = 20,
    ) -> EmotionFlowReport:
        """Run the full pipeline on a source and return a populated report.

        Args:
            source:    Path to a ``.txt`` file, directory of ``.txt`` files,
                       or a vox_romana JSON produced by a debate/news run.
            leader:    Speaker name filter.  Required for multi-speaker JSON.
            min_chars: Minimum segment character count (shorter are skipped).

        Returns:
            Populated :class:`~.models.EmotionFlowReport`.  If ``output_dir``
            was configured, ``report.report_path`` points to the saved HTML.
        """
        freeze_randomness(self._seed)
        manifest = build_manifest(self._seed, self._model_versions)

        leader_name = leader or str(Path(source).stem)

        # 1 — Load
        logger.info("[1/7] Loading segments from %s (leader=%s)", source, leader_name)
        segments = FlexibleLoader.load(
            source,
            leader,
            min_chars=min_chars,
            **self._loader_kwargs,
        )
        logger.info("  → %d segments loaded", len(segments))

        # 2 — Translate
        logger.info("[2/7] Translating segments to English")
        translator_kwargs = dict(self._translator_kwargs)
        if not self._run_translation:
            translator_kwargs["_force_passthrough"] = True
        translator = TranslatorAgent(self._cache_dir, **translator_kwargs)
        segments = translator.translate(segments)
        quality = translator.aggregate_quality(segments)
        n_flagged = quality.get("n_flagged", 0)
        back_translation_agreement = quality.get("mean_confidence", 1.0)
        logger.info(
            "  → mean confidence=%.3f, flagged=%d",
            back_translation_agreement,
            n_flagged,
        )

        # 3 — Tag emotions
        logger.info("[3/7] Tagging emotions")
        tagger = EmotionTaggerAgent(self._cache_dir, **self._tagger_kwargs)
        tags = tagger.tag(segments)
        logger.info("  → %d tags produced", len(tags))

        # 4 — Build sequence
        sequence = EmotionSequence(
            leader_name=leader_name,
            segments=segments,
            tags=tags,
            speech_ids=list({seg.speech_id for seg in segments}),
        )

        # 5 — Transition matrix
        logger.info("[4/7] Building transition matrix")
        transition_agent = TransitionBuilderAgent(**self._transition_kwargs)
        tm = transition_agent.build(sequence)
        logger.info("  → %d transitions recorded", tm.n_transitions)

        # 6 — Semantic flow (optional)
        semantic_result = None
        if self._run_semantic:
            logger.info("[5/7] Computing semantic-emotion flow correlation")
            semantic_agent = SemanticFlowAgent(
                self._cache_dir, **self._semantic_kwargs
            )
            semantic_result = semantic_agent.compute(sequence)
            logger.info(
                "  → Pearson r=%.3f (p=%.4f)",
                semantic_result.pearson_r,
                semantic_result.p_value,
            )
        else:
            logger.info("[5/7] Semantic flow skipped (run_semantic=False)")

        # 7 — Null-hypothesis test
        logger.info("[6/7] Running null-hypothesis tests")
        tag_sequence = [t.dominant for t in tags]
        tester = NullHypothesisTesterAgent(
            seed=self._seed, **self._tester_kwargs
        )
        null_result = tester.test(tm, tag_sequence, self._n_permutations)
        logger.info(
            "  → is_structured=%s, chi2_p=%.4f, perm_p=%.4f",
            null_result.is_structured,
            null_result.chi2_p,
            null_result.permutation_p,
        )

        # 8 — Assemble report
        dominant_cycle = _dominant_cycle(tag_sequence)
        report = EmotionFlowReport(
            leader_name=leader_name,
            n_speeches=len(sequence.speech_ids),
            n_segments=len(segments),
            dominant_cycle=dominant_cycle,
            transition_matrix=tm,
            null_test=null_result,
            semantic_flow=semantic_result,
            manifest=manifest,
            back_translation_agreement=back_translation_agreement,
            n_flagged_segments=n_flagged,
        )

        # 9 — Render and save
        logger.info("[7/7] Rendering report")
        reporter_kwargs = dict(self._reporter_kwargs)
        if "visualizer" not in reporter_kwargs:
            reporter_kwargs["visualizer"] = EmotionFlowVisualizer(**self._visualizer_kwargs)
        reporter = EmotionFlowReporter(
            output_dir=self._output_dir,
            **reporter_kwargs,
        )
        report = reporter.build(report)

        if report.report_path:
            logger.info("Report saved → %s", report.report_path)
        else:
            logger.info("Report built in memory (no output_dir configured)")

        return report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dominant_cycle(tag_sequence: list[str]) -> list[str]:
    """Extract run-length-encoded dominant emotion cycle (deduplicated runs).

    Example: ["joy", "joy", "fear", "fear", "joy"] → ["joy", "fear", "joy"]
    """
    if not tag_sequence:
        return []
    result: list[str] = [tag_sequence[0]]
    for label in tag_sequence[1:]:
        if label != result[-1]:
            result.append(label)
    return result
