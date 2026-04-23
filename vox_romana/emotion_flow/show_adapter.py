"""Convert an EmotionFlowReport into a NewsConfig for the show pipeline.

The adapter is a pure translation layer — no LLM calls, no I/O.

    Input:  EmotionFlowReport  (produced by run_emotion_flow.py)
    Output: NewsConfig          (ready for NewsConductor — no bootstrap needed)

MAPPING LOGIC
─────────────
dominant_cycle
    → One SegmentConfig per unique emotion (ordered by first appearance),
      capped at adapter.n_segments.  Always includes an overview segment
      (first) and a synthesis segment (last).  Emotion segments fill the
      middle.

null_test.is_structured
    → DirectorBrief.show_arc:
          True  → "structured arc — emotions follow a statistically
                   significant pattern; build toward the dominant theme."
          False → "exploratory survey — emotional transitions appear
                   stochastic; treat each segment as independent discovery."

semantic_flow.pearson_r
    → Overall tone:
          |r| > 0.5  → "analytical"    (topic and emotion arc strongly aligned)
          |r| > 0.2  → "investigative" (moderate alignment)
          else       → "exploratory"   (topics and emotions diverge)

transition_matrix
    → key_facts injected into each segment's SegmentConfig
      (top-3 transitions by probability, stated as plain-language facts).

leader_name
    → Config title, anchor name, anchor voice_hint.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import ShowHostConfig
from ..shows.news.config import AnchorConfig, DirectorBrief, NewsConfig, SegmentConfig
from .models import (
    EMOTION_LABELS,
    EmotionFlowReport,
    NullTestResult,
    SemanticFlowResult,
    TransitionMatrix,
)

# ── Per-emotion descriptors ────────────────────────────────────────────────────

_EMOTION_TONE: dict[str, str] = {
    "anger":    "confrontational",
    "disgust":  "critical",
    "fear":     "urgent",
    "joy":      "upbeat",
    "neutral":  "measured",
    "sadness":  "somber",
    "surprise": "dramatic",
}

_EMOTION_ANGLE: dict[str, str] = {
    "anger":    "When does {name}'s rhetoric turn combative, and what triggers the shift?",
    "disgust":  "What patterns of moral rejection and contempt appear in {name}'s speeches?",
    "fear":     "How does {name} deploy fear and urgency as rhetorical devices?",
    "joy":      "What moments of aspiration and positivity punctuate {name}'s rhetoric?",
    "neutral":  "Where does {name}'s delivery become detached, measured, or informational?",
    "sadness":  "When does {name}'s language turn somber — loss, mourning, or empathy?",
    "surprise": "How does {name} use shock and reversal to capture audience attention?",
}

_DEFAULT_N_SEGMENTS = 6  # overview + up to 4 emotion segments + synthesis


# ── EmotionFlowShowAdapter ─────────────────────────────────────────────────────


class EmotionFlowShowAdapter:
    """Convert an EmotionFlowReport into a fully populated NewsConfig.

    The resulting NewsConfig can be passed directly to NewsConductor without
    running a bootstrap/architect stage — all segments, tone, anchor, and
    director brief are pre-filled from the statistical findings.

    Parameters
    ----------
    reporter_name:
        The name of the show / reporting entity.  Appears in the title.
    n_segments:
        Maximum number of segments (including overview and synthesis).
        If None, defaults to _DEFAULT_N_SEGMENTS (6).
    target_duration_minutes:
        Approximate target show duration.  Passed through to NewsConfig.
    """

    def __init__(
        self,
        *,
        reporter_name: str = "Emotion Flow",
        n_segments: int | None = None,
        target_duration_minutes: int = 10,
    ) -> None:
        self.reporter_name = reporter_name
        self.n_segments = n_segments or _DEFAULT_N_SEGMENTS
        self.target_duration_minutes = target_duration_minutes

    # ── Public API ─────────────────────────────────────────────────────────────

    def adapt(self, report: EmotionFlowReport) -> NewsConfig:
        """Convert an EmotionFlowReport to a NewsConfig."""
        segments = self._build_segments(report)
        anchor = self._build_anchor(report)
        director_brief = self._build_director_brief(report)
        tone = self._derive_tone(report)
        show_host = ShowHostConfig(
            name="Narrator",
            role="narrator",
            style="Clear, analytical, documentary-style narration.",
            voice_hint="calm, authoritative, documentary narrator",
        )

        leader = report.leader_name or "the speaker"
        title = f"{self.reporter_name}: Emotional Patterns in {leader}'s Speeches"
        brief = self._build_brief(report)

        return NewsConfig(
            brief=brief,
            title=title,
            topic=f"Emotion flow analysis of {leader}",
            tone=tone,
            anchors=[anchor],
            segments=segments,
            show_host=show_host,
            director_brief=director_brief,
            target_segments=len(segments),
            target_duration_minutes=float(self.target_duration_minutes),
        )

    def adapt_from_dict(self, data: dict[str, Any]) -> NewsConfig:
        """Convert a raw report dict (from JSON) to a NewsConfig."""
        report = self._report_from_dict(data)
        return self.adapt(report)

    def adapt_from_json(self, json_str: str) -> NewsConfig:
        """Parse a JSON string and convert to a NewsConfig."""
        data = json.loads(json_str)
        return self.adapt_from_dict(data)

    def adapt_from_file(self, path: str | Path) -> NewsConfig:
        """Load a report JSON file and convert to a NewsConfig."""
        text = Path(path).read_text(encoding="utf-8")
        return self.adapt_from_json(text)

    def adapt_to_dict(self, report: EmotionFlowReport) -> dict[str, Any]:
        """Convert a report to a NewsConfig, then return it as a plain dict."""
        config = self.adapt(report)
        return asdict(config)

    # ── Private: report reconstruction ────────────────────────────────────────

    @staticmethod
    def _report_from_dict(data: dict[str, Any]) -> EmotionFlowReport:
        """Reconstruct an EmotionFlowReport from a raw JSON dict."""
        # Null test
        nt_data: dict[str, Any] = data.get("null_test") or {}
        null_test: NullTestResult | None = None
        if nt_data:
            entropy_ci_raw = nt_data.get("entropy_ci", [0.0, 0.0])
            null_test = NullTestResult(
                permutation_p=float(nt_data.get("permutation_p", 1.0)),
                chi2_p=float(nt_data.get("chi2_p", 1.0)),
                chi2_statistic=float(nt_data.get("chi2_statistic", 0.0)),
                g_test_p=float(nt_data.get("g_test_p", 1.0)),
                g_statistic=float(nt_data.get("g_statistic", 0.0)),
                cramers_v=float(nt_data.get("cramers_v", 0.0)),
                entropy_observed=float(nt_data.get("entropy_observed", 0.0)),
                entropy_random_mean=float(nt_data.get("entropy_random_mean", 0.0)),
                entropy_ci=(float(entropy_ci_raw[0]), float(entropy_ci_raw[1])),
                is_structured=bool(nt_data.get("is_structured", False)),
                n_permutations=int(nt_data.get("n_permutations", 0)),
            )

        # Semantic flow
        sf_data: dict[str, Any] = data.get("semantic_flow") or {}
        semantic_flow: SemanticFlowResult | None = None
        if sf_data:
            semantic_flow = SemanticFlowResult(
                pearson_r=float(sf_data.get("pearson_r", 0.0)),
                p_value=float(sf_data.get("p_value", 1.0)),
                topic_shifts=list(sf_data.get("topic_shifts", [])),
                emotion_shifts=list(sf_data.get("emotion_shifts", [])),
                n_pairs=int(sf_data.get("n_pairs", 0)),
            )

        # Transition matrix
        tm_data: dict[str, Any] = data.get("transition_matrix") or {}
        transition_matrix: TransitionMatrix | None = None
        raw_matrix = tm_data.get("matrix")
        if tm_data and raw_matrix is not None:
            matrix = np.array(raw_matrix, dtype=np.float32)
            n = len(matrix)
            count_raw = tm_data.get("count_matrix")
            ci_low_raw = tm_data.get("ci_low")
            ci_high_raw = tm_data.get("ci_high")
            transition_matrix = TransitionMatrix(
                leader_name=str(data.get("leader_name", "")),
                matrix=matrix,
                count_matrix=(
                    np.array(count_raw, dtype=np.int32)
                    if count_raw is not None
                    else np.zeros((n, n), dtype=np.int32)
                ),
                ci_low=(
                    np.array(ci_low_raw, dtype=np.float32)
                    if ci_low_raw is not None
                    else np.zeros((n, n), dtype=np.float32)
                ),
                ci_high=(
                    np.array(ci_high_raw, dtype=np.float32)
                    if ci_high_raw is not None
                    else np.zeros((n, n), dtype=np.float32)
                ),
                emotion_labels=list(tm_data.get("emotion_labels", EMOTION_LABELS)),
                smoothing_k=float(tm_data.get("smoothing_k", 0.1)),
                n_transitions=int(tm_data.get("n_transitions", 0)),
            )

        return EmotionFlowReport(
            leader_name=str(data.get("leader_name", "")),
            n_speeches=int(data.get("n_speeches", 1)),
            n_segments=int(data.get("n_segments", 0)),
            dominant_cycle=list(data.get("dominant_cycle", [])),
            null_test=null_test,
            semantic_flow=semantic_flow,
            transition_matrix=transition_matrix,
            back_translation_agreement=float(data.get("back_translation_agreement", 1.0)),
            n_flagged_segments=int(data.get("n_flagged_segments", 0)),
            narrative=str(data.get("narrative", "")),
            report_path=str(data.get("report_path", "")),
        )

    # ── Private: segment building ──────────────────────────────────────────────

    def _build_segments(self, report: EmotionFlowReport) -> list[SegmentConfig]:
        """Build ordered SegmentConfig list from the report."""
        transition_facts = self._top_transitions(report.transition_matrix)
        leader = report.leader_name or "the speaker"

        # Deduplicate cycle, preserving order of first appearance.
        seen: list[str] = []
        for emotion in report.dominant_cycle:
            if emotion not in seen:
                seen.append(emotion)

        segments: list[SegmentConfig] = []

        # 1. Overview — always present.
        cycle_str = (
            " → ".join(report.dominant_cycle)
            if report.dominant_cycle
            else "inconclusive"
        )
        segments.append(
            SegmentConfig(
                headline=f"Mapping the Emotional Landscape of {leader}'s Rhetoric",
                angle=(
                    f"Overview of emotion flow analysis across {report.n_speeches} "
                    f"speech(es). What emotions dominate and how do they sequence?"
                ),
                key_facts=[
                    f"Analyzed {report.n_speeches} speech(es), {report.n_segments} segments.",
                    f"Dominant emotional cycle: {cycle_str}.",
                    *transition_facts[:2],
                ],
                tone="measured",
                duration_hint="short",
                segment_type="anchor_read",
            )
        )

        # 2. One segment per unique emotion (budget = n_segments - 2, min 0).
        if self.n_segments >= 3:
            emotion_budget = self.n_segments - 2
            for emotion in seen[:emotion_budget]:
                angle_template = _EMOTION_ANGLE.get(
                    emotion, "Analysis of {{name}}'s {emotion} rhetoric."
                )
                angle = angle_template.format(name=leader, emotion=emotion)
                segments.append(
                    SegmentConfig(
                        headline=f"{leader}'s {emotion.capitalize()} Rhetoric",
                        angle=angle,
                        key_facts=[*transition_facts, f"Dominant emotion: {emotion}."],
                        tone=_EMOTION_TONE.get(emotion, "measured"),
                        duration_hint="medium",
                        segment_type="anchor_read",
                    )
                )

        # 3. Synthesis — present when n_segments >= 2.
        if self.n_segments >= 2:
            is_structured = (
                report.null_test.is_structured
                if report.null_test is not None
                else False
            )
            synthesis_angle = (
                "These patterns are statistically significant — not random. "
                "What does a structured emotional arc reveal about rhetorical strategy?"
                if is_structured
                else (
                    "The emotional transitions appear stochastic. What does an absence of "
                    "patterned structure tell us about this speaker?"
                )
            )
            segments.append(
                SegmentConfig(
                    headline=f"What the Numbers Tell Us About {leader}",
                    angle=synthesis_angle,
                    key_facts=transition_facts,
                    tone="analytical" if is_structured else "exploratory",
                    duration_hint="medium",
                    segment_type="anchor_read",
                )
            )

        return segments

    def _build_anchor(self, report: EmotionFlowReport) -> AnchorConfig:
        leader = report.leader_name or "the speaker"
        return AnchorConfig(
            name=f"{self.reporter_name} Analyst",
            style="Analytical, measured, evidence-driven delivery.",
            voice_fingerprint=(
                f"Precise and methodical. Introduces {leader}'s emotional patterns "
                f"as statistical findings rather than editorial opinion. "
                f"Uses data as anchor points throughout."
            ),
            voice_hint="clear, authoritative, documentary-style, measured pace",
            voice_gender=None,
        )

    def _build_director_brief(self, report: EmotionFlowReport) -> DirectorBrief:
        is_structured = (
            report.null_test.is_structured
            if report.null_test is not None
            else False
        )
        cramers_v = (
            report.null_test.cramers_v if report.null_test is not None else 0.0
        )
        leader = report.leader_name or "the speaker"

        if is_structured:
            show_arc = (
                f"Structured arc — {leader}'s emotional transitions are statistically "
                f"significant (Cramér's V = {cramers_v:.2f}). "
                f"Open with overview, build through individual emotions toward the "
                f"dominant theme, close with synthesis of what the pattern means."
            )
            mood_progression = ["curious", "investigative", "analytical", "conclusive"]
        else:
            show_arc = (
                f"Exploratory survey — {leader}'s emotional transitions appear stochastic "
                f"(Cramér's V = {cramers_v:.2f}). "
                f"Each segment stands independently. Maintain measured tone throughout; "
                f"do not force a narrative arc onto uncertain findings."
            )
            mood_progression = ["curious", "measured", "reflective", "open"]

        return DirectorBrief(
            show_arc=show_arc,
            mood_progression=mood_progression,
            check_interval=2,
            closing_strategy=(
                "Return to the central question: does this speaker's emotional "
                "rhetoric reveal strategic design or authentic expression?"
            ),
        )

    def _derive_tone(self, report: EmotionFlowReport) -> str:
        if report.semantic_flow is None:
            return "analytical"
        r = abs(report.semantic_flow.pearson_r)
        if r > 0.5:
            return "analytical"
        if r > 0.2:
            return "investigative"
        return "exploratory"

    def _build_brief(self, report: EmotionFlowReport) -> str:
        leader = report.leader_name or "the speaker"
        cycle = (
            " → ".join(report.dominant_cycle) if report.dominant_cycle else "undetermined"
        )
        is_structured = (
            report.null_test.is_structured if report.null_test is not None else None
        )
        structure_note = (
            "The transitions are statistically structured."
            if is_structured is True
            else "The transitions appear stochastic."
            if is_structured is False
            else ""
        )
        pearson_note = ""
        if report.semantic_flow is not None:
            r = report.semantic_flow.pearson_r
            pearson_note = f" Topic-emotion Pearson r = {r:.2f}."

        return (
            f"An analytical news show examining the emotion flow patterns in "
            f"{leader}'s rhetoric, based on automated analysis of {report.n_speeches} "
            f"speech(es) ({report.n_segments} segments). "
            f"Dominant emotional cycle: {cycle}. "
            f"{structure_note}{pearson_note} "
            f"Present the statistical findings with clarity and analytical precision."
        ).strip()

    @staticmethod
    def _top_transitions(
        tm: TransitionMatrix | None, n: int = 3
    ) -> list[str]:
        """Return top-n cross-emotion transitions as plain-language fact strings."""
        if tm is None:
            return []
        labels = tm.emotion_labels
        matrix = tm.matrix
        facts: list[tuple[float, str]] = []
        for i, from_label in enumerate(labels):
            for j, to_label in enumerate(labels):
                if from_label == to_label:
                    continue
                p = float(matrix[i, j])
                if p > 0.0:
                    facts.append(
                        (p, f"After {from_label}, {to_label} follows {p:.0%} of the time.")
                    )
        facts.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in facts[:n]]
