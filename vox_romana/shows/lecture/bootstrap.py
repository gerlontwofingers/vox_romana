"""Lecture bootstrap — brief → fully populated LectureConfig.

Mirrors the audiobook bootstrap chain:
    1. ARCHITECT   call: brief → title, sections, professor, subject area
    2. PROFESSOR PROFILER call: name + style → voice fingerprint
    3. DIRECTOR    call: section list → pedagogical arc
"""

from __future__ import annotations

import json

from ...core.bootstrap import BaseShowBootstrap
from ...core.client import DeepSeekClient
from ...core.names import build_curated_name_map, remap_name
from ...core.search import BaseSearchProvider
from .config import (
    LectureConfig,
    LectureDirectorBrief,
    LectureSection,
    ProfessorConfig,
)
from .prompts import (
    ARCHITECT_SYSTEM_PROMPT,
    PROFESSOR_PROFILER_SYSTEM_PROMPT,
)


def _strip_json_fences(raw: str) -> str:
    """Strip Markdown code fences and isolate the outermost JSON object.

    LLMs frequently wrap JSON in ```json ... ``` fences and sometimes
    add a leading "Here is the JSON:" prelude. This helper produces a
    clean substring suitable for ``json.loads``.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    return cleaned


# ═══════════════════════════════════════════════════════════════════════════
# DIRECTOR SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

LECTURE_DIRECTOR_PROMPT = """\
You are the Director for a lecture production. Your job is to plan the
pedagogical arc — how understanding builds across sections.

You will receive details about the lecture: title, subject area, sections,
and tone.

You must produce a JSON object with EXACTLY this schema:

{{
  "thesis": "string — the one-sentence statement of what this lecture teaches \
or argues. The throughline.",
  "progression": ["array of strings — the conceptual milestones the lecture \
hits in order. 3-6 entries. Example: ['establish stakes', 'introduce framework', \
'apply to case', 'reveal complication', 'synthesize']"],
  "pedagogical_notes": "string — high-level notes on teaching style: when to \
use analogies, when to recap, where listeners are likely to get lost",
  "closing_strategy": "string — how to land the lecture. Example: 'Close with \
a single open question that invites the listener to think further, after a \
brief recap of the three main takeaways.'"
}}

RULES:
- A lecture should build understanding cumulatively. Each section earns the next.
- The closing strategy should match the tone — definitive synthesis for
  technical lectures, open question for philosophical, call to action for
  applied/practical.
- ALWAYS respond with valid JSON. Nothing else.
"""


class LectureBootstrap(BaseShowBootstrap):
    """Orchestrates the brief-to-config bootstrap chain for lectures.

    Chain: brief → [Researcher] → Architect → Professor profiler → Director → config
    """

    def __init__(
        self,
        client: DeepSeekClient | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        super().__init__(client, api_key=api_key)

    def from_brief(
        self,
        brief: str,
        *,
        reference_material: str | None = None,
        fact_check: str = "off",
        overrides: dict | None = None,
        auto_research: bool = False,
        search_provider: BaseSearchProvider | None = None,
        target_words: int | None = None,
        memory=None,
        series_context: str | None = None,
    ) -> LectureConfig:
        """Run the full bootstrap chain: brief → fully populated LectureConfig."""
        overrides = overrides or {}

        if series_context:
            brief = f"{series_context}\n\n{brief}"

        # ── Stage 0: Auto-research (optional) ────────────────────────
        if auto_research and not reference_material:
            from ...core.search import build_research_provider
            from ..debates.researcher import DebateResearcher
            effective_provider = build_research_provider(search_provider, memory)
            if effective_provider is None:
                researcher = DebateResearcher(self._client)
            else:
                researcher = DebateResearcher(self._client, effective_provider)
            reference_material = researcher.run(
                brief,
                perspective=overrides.get("perspective"),
            ) or None

        # ── Stage 1: Architect ──────────────────────────────────────
        config = self._run_architect(
            brief, reference_material, fact_check, target_words=target_words,
        )

        # ── Stage 2: Professor profiling ─────────────────────────────
        if config.professor and config.professor.name:
            config.professor = self._profile_professor(
                config.professor,
                subject_area=config.subject_area or "general",
                tone=config.tone or "engaging",
            )

        # ── Stage 3: Director ─────────────────────────────────────────
        config.director_brief = self._run_director(config)

        # ── Stage 4: Apply overrides ─────────────────────────────────
        self._apply_overrides(config, overrides)

        return config

    def _run_architect(
        self,
        brief: str,
        reference_material: str | None,
        fact_check: str,
        *,
        target_words: int | None = None,
    ) -> LectureConfig:
        """Call the Architect LLM to produce a skeleton LectureConfig."""
        user_parts = [f"BRIEF:\n{brief}"]
        if reference_material:
            user_parts.append(f"\nREFERENCE MATERIAL:\n{reference_material}")
        if target_words:
            ideal_sections = max(4, target_words // 800)
            beats_per_section = max(1, target_words // (150 * ideal_sections))
            user_parts.append(
                f"\nTARGET LENGTH: {target_words:,} words. "
                f"Plan approximately {ideal_sections} sections with "
                f"~{beats_per_section} beats each "
                f"({ideal_sections} \u00d7 {beats_per_section} \u00d7 ~150 words/beat "
                f"\u2248 {ideal_sections * beats_per_section * 150:,} words). "
                f"Always include exactly one 'intro' section first and one "
                f"'outro' section last; the rest should be a mix of 'concept', "
                f"'case_study', 'exercise', and 'recap' sections."
            )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            temperature=0,
        )

        return self._parse_architect_response(
            raw, brief=brief, reference_material=reference_material,
            fact_check=fact_check,
        )

    def _parse_architect_response(
        self,
        raw_json: str,
        *,
        brief: str = "",
        reference_material: str | None = None,
        fact_check: str = "off",
    ) -> LectureConfig:
        """Parse the Architect's JSON into a LectureConfig."""
        cleaned = _strip_json_fences(raw_json)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Lecture Architect produced invalid JSON. First 200 chars: "
                f"{raw_json[:200]!r}\nParse error: {e}"
            ) from e

        prof_data = data.get("professor", {})
        name_map = build_curated_name_map(
            [
                (
                    prof_data.get("name", ""),
                    prof_data.get("voice_gender"),
                    "lecture.professor",
                )
            ],
            brief=brief,
        )

        # Build professor.
        professor = ProfessorConfig(
            name=remap_name(prof_data.get("name", ""), name_map) or "",
            title=prof_data.get("title"),
            style=prof_data.get("style"),
            expertise=prof_data.get("expertise", []),
            voice_gender=prof_data.get("voice_gender"),
        )

        # Build sections.
        sections = [
            LectureSection(
                title=s.get("title", ""),
                section_type=s.get("section_type", "concept"),
                learning_objective=s.get("learning_objective"),
                key_points=s.get("key_points", []),
                examples=s.get("examples", []),
                duration_hint=s.get("duration_hint", "medium"),
                target_beats=s.get("target_beats", 2),
            )
            for s in data.get("sections", [])
        ]

        return LectureConfig(
            brief=brief,
            reference_material=reference_material,
            fact_check=fact_check,
            title=data.get("title"),
            topic=data.get("topic"),
            subject_area=data.get("subject_area"),
            tone=data.get("tone"),
            setting=data.get("setting"),
            professor=professor,
            sections=sections,
        )

    def _profile_professor(
        self,
        professor: ProfessorConfig,
        *,
        subject_area: str,
        tone: str,
    ) -> ProfessorConfig:
        """Give the professor a voice_fingerprint via LLM."""
        stub = (
            f"Name: {professor.name}\n"
            f"Title: {professor.title or 'Professor'}\n"
            f"Style: {professor.style or 'methodical and warm'}\n"
            f"Expertise: {', '.join(professor.expertise) or subject_area}\n"
            f"Subject area: {subject_area}\n"
            f"Lecture tone: {tone}\n"
        )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": PROFESSOR_PROFILER_SYSTEM_PROMPT},
                {"role": "user", "content": stub},
            ],
            temperature=0,
        )

        cleaned = _strip_json_fences(raw)

        try:
            data = json.loads(cleaned)
            professor.voice_fingerprint = data.get("voice_fingerprint", "")
        except (json.JSONDecodeError, TypeError):
            professor.voice_fingerprint = (
                f"(measured, methodical lecturer; {professor.style or 'warm and clear'})"
            )

        # Derive a TTS hint if missing.
        if not professor.tts_style:
            professor.tts_style = (
                "Speak with the deliberate, paced delivery of a teacher: "
                "let key sentences breathe, pause before pivots, vary "
                "volume to underline emphasis."
            )

        return professor

    def _run_director(self, config: LectureConfig) -> LectureDirectorBrief:
        """Call the Director LLM to plan the pedagogical arc."""
        section_list = "\n".join(
            f"  {i+1}. [{s.section_type}] {s.title} ({s.duration_hint})"
            for i, s in enumerate(config.sections)
        )

        user_message = (
            f"TITLE: {config.title or 'Untitled'}\n"
            f"SUBJECT AREA: {config.subject_area or 'general'}\n"
            f"TONE: {config.tone or 'engaging'}\n"
            f"TOTAL SECTIONS: {len(config.sections)}\n"
            f"SECTIONS:\n{section_list}"
        )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": LECTURE_DIRECTOR_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )

        return self._parse_director_response(raw)

    def _parse_director_response(self, raw_json: str) -> LectureDirectorBrief:
        """Parse the Director's JSON into a LectureDirectorBrief."""
        cleaned = _strip_json_fences(raw_json)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Lecture Director produced invalid JSON. First 200 chars: "
                f"{raw_json[:200]!r}\nParse error: {e}"
            ) from e

        return LectureDirectorBrief(
            thesis=data.get("thesis", ""),
            progression=data.get("progression", []),
            pedagogical_notes=data.get("pedagogical_notes"),
            closing_strategy=data.get("closing_strategy"),
        )
