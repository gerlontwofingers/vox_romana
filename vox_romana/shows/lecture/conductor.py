"""The Lecture Conductor — runtime engine that generates a lecture.

Modeled on the AudiobookConductor: a single-voice scripted show that walks
section-by-section through an outline. Differences:
    - Sections are TYPED (intro / concept / case_study / exercise / recap /
      outro). Each type triggers a different overlay in the prompt.
    - Director check-ins use pedagogical thesis/progression rather than a
      dramatic mood arc.
    - Pedagogical beats are typically 1 per section (lectures are denser
      than narrative prose).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from ...core.client import DeepSeekClient
from ...core.conductor import BaseConductor, ConductorResult
from ...core.session import AgentSession
from .bootstrap import LectureBootstrap
from .config import LectureConfig, LectureSection as LectureSectionConfig
from .models import LectureEvent, LectureSection as LectureSectionModel, MergedLecture
from .prompts import (
    DIRECTOR_CHECK_PROMPT,
    OVERLAY_BY_TYPE,
    PROFESSOR_SYSTEM_PROMPT,
    SECTION_DELIVERY_PROMPT,
)


# How many delivered section transcripts to keep as verbatim context.
DEFAULT_RECENT_WINDOW = 3

# Duration hints map to approximate sentence counts per beat.
DURATION_SENTENCES = {
    "short": (3, 6),
    "medium": (6, 12),
    "long": (12, 20),
}

# Per-beat hard word cap (Q-5). Overnight battery showed lecture avg
# 171w/event was too dense for TTS pacing. Applied unconditionally.
MAX_BEAT_WORDS = 140


# ═══════════════════════════════════════════════════════════════════════════
# LectureConductorResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LectureConductorResult(ConductorResult):
    """Typed result from a LectureConductor run."""

    lecture_events: list[LectureEvent] = field(default_factory=list)
    lecture_sections: list[LectureSectionModel] = field(default_factory=list)
    lecture_id: str = ""

    def to_merged_lecture(self) -> MergedLecture:
        """Convert this result into a MergedLecture for serialization."""
        config = self.config
        if not isinstance(config, LectureConfig):
            raise TypeError(f"Expected a LectureConfig, got {type(config)}")

        voice_casting: dict[str, str] = {}
        voice_hints: dict[str, str] = {}
        tts_styles: dict[str, str] = {}

        if config.professor:
            p = config.professor
            if p.voice_gender:
                voice_casting[p.name] = p.voice_gender
            if p.voice_hint:
                voice_hints[p.name] = p.voice_hint
            if p.tts_style:
                tts_styles[p.name] = p.tts_style

        return MergedLecture(
            lecture_id=self.lecture_id,
            title=config.title or "Untitled Lecture",
            sections=tuple(self.lecture_sections),
            events=tuple(self.lecture_events),
            professor=config.professor.name if config.professor else "",
            professor_title=config.professor.title if config.professor else None,
            subject_area=config.subject_area,
            tone=config.tone,
            section_count=len(self.lecture_sections),
            voice_casting=voice_casting,
            voice_hints=voice_hints,
            tts_styles=tts_styles,
        )


# ═══════════════════════════════════════════════════════════════════════════
# THE LECTURE CONDUCTOR
# ═══════════════════════════════════════════════════════════════════════════

class LectureConductor(BaseConductor):
    """Generates a complete lecture by orchestrating a single professor agent.

    The section loop walks the outline in order. Each section triggers a
    type-specific overlay (intro/concept/case_study/exercise/recap/outro)
    that shapes the pedagogical mode of that section.
    """

    _bootstrap_cls = LectureBootstrap

    def __init__(
        self,
        config: LectureConfig,
        *,
        client: DeepSeekClient | None = None,
        api_key: str | None = None,
        **base_kwargs,
    ) -> None:
        resolved_client = client or DeepSeekClient(api_key=api_key)
        super().__init__(config, client=resolved_client, **base_kwargs)

        self._professor_prompt: str = ""
        self._professor_session: AgentSession | None = None

        self._transcript: list[dict] = []
        self._director_note: str = ""
        self._llm_calls: int = 0
        self._lecture_id: str = ""

    @classmethod
    def from_brief(
        cls,
        brief: str,
        *,
        reference_material: str | None = None,
        fact_check: str = "off",
        overrides: dict | None = None,
        client: DeepSeekClient | None = None,
        api_key: str | None = None,
        target_words: int | None = None,
    ) -> LectureConductor:
        """Bootstrap a LectureConfig from a brief and create a conductor."""
        return super().from_brief(  # type: ignore[return-value]
            brief,
            reference_material=reference_material,
            fact_check=fact_check,
            overrides=overrides,
            client=client,
            api_key=api_key,
            target_words=target_words,
        )

    # ═══════════════════════════════════════════════════════════════════
    # BaseConductor interface
    # ═══════════════════════════════════════════════════════════════════

    def _build_agents(self) -> None:
        """Build the professor system prompt and session."""
        config = self.config
        if not isinstance(config, LectureConfig):
            raise TypeError(f"LectureConductor requires LectureConfig, got {type(config)}")

        self._professor_prompt = self._build_professor_prompt()
        self._professor_session = self._create_session(
            self._professor_prompt,
            recent_window=DEFAULT_RECENT_WINDOW,
            session_name="professor",
        )

    def _run_turn_loop(self) -> list[dict]:
        """Run the section-by-section generation loop."""
        config = self.config
        if not isinstance(config, LectureConfig):
            raise TypeError(f"LectureConductor requires LectureConfig, got {type(config)}")

        self._lecture_id = str(uuid.uuid4())[:8]
        all_events: list[dict] = []
        global_order = 0
        sections = config.sections
        total_sections = len(sections)

        # ── Auto-scale beats to hit target_words ─────────────────────
        if config.target_words and total_sections > 0:
            words_per_beat = 150  # Lectures: denser than dialogue, lighter than prose
            total_beats_needed = max(total_sections, config.target_words // words_per_beat)
            weight_map = {"short": 1, "medium": 2, "long": 3}
            total_weight = sum(weight_map.get(s.duration_hint, 2) for s in sections)
            for s in sections:
                w = weight_map.get(s.duration_hint, 2)
                derived = max(1, round(total_beats_needed * w / total_weight))
                s.target_beats = max(s.target_beats, derived)

        # ── Section loop ─────────────────────────────────────────────
        for s_idx, section in enumerate(sections):
            s_num = s_idx + 1

            section_events = self._generate_section(
                section=section,
                section_number=s_num,
                total_sections=total_sections,
                global_order_start=global_order,
            )

            for evt in section_events:
                all_events.append(evt)
                self._transcript.append(evt)
                global_order += 1
                self._record_words(evt.get("text", ""))

            if self._over_budget():
                break

            director_brief = config.director_brief
            if (
                director_brief
                and s_num > 0
                and s_num % director_brief.check_interval == 0
            ):
                self._director_check(s_num, total_sections)

        return all_events

    def _expansion_pass(
        self,
        *,
        deficit_words: int,
        deficit_seconds: float,
    ) -> list[dict]:
        """Add a pedagogical extension beat to close a length deficit.

        Continues teaching from where the lecture ended — typically deepens
        the final concept, adds another example, or raises an open question
        the listener can take away.
        """
        if self._professor_session is None:
            return []
        config = self.config
        if not isinstance(config, LectureConfig):
            return []
        if not config.sections:
            return []

        last_section = config.sections[-1]
        s_id = f"sec_bonus_{len(config.sections):03d}"

        length_contract = self._length_contract_block(
            deficit_words, tolerance_pct=10.0,
        )
        beat_instruction = (
            "Continue teaching from where the previous section left off. "
            "This is NOT a recap of the whole lecture — extend the final "
            "thread with one more example, application, or open question "
            "the listener can take away. Stay in your teaching voice."
            f"\n\n{length_contract}"
        )

        user_message = SECTION_DELIVERY_PROMPT.format(
            section_number=len(config.sections) + 1,
            total_sections=len(config.sections) + 1,
            title=f"{last_section.title} — extension",
            section_type=last_section.section_type,
            learning_objective=last_section.learning_objective or "(deepen the prior section)",
            key_points="(continue from prior beats; do not restart)",
            examples="(use a fresh illustration if helpful)",
            type_overlay=OVERLAY_BY_TYPE.get(last_section.section_type, ""),
            reference_section="",
            beat_instruction=beat_instruction,
        )

        try:
            raw_response = self._professor_session.send(user_message)
        except Exception as exc:  # noqa: BLE001
            self._length_warnings.append(
                f"[expansion sec_bonus] LLM call failed: {exc!r}"
            )
            return []

        event_dict = self._parse_response(raw_response, from_agent="professor")
        prof_name = config.professor.name if config.professor else "Professor"
        event_dict["speaker"] = prof_name
        event_dict["role"] = "professor"
        event_dict["event_kind"] = "narration"
        event_dict["chapter_id"] = s_id
        event_dict["beat_label"] = "extension"
        event_dict.setdefault("pause_before", "long")
        event_dict.setdefault("pause_after", "long")
        self._record_words(event_dict.get("text", ""))
        return [event_dict]

    def _package_result(self, raw_events: list[dict]) -> LectureConductorResult:
        """Convert raw events into a typed LectureConductorResult."""
        config = self.config
        if not isinstance(config, LectureConfig):
            raise TypeError(f"LectureConductor requires LectureConfig, got {type(config)}")

        lecture_events: list[LectureEvent] = []
        for i, evt in enumerate(raw_events):
            tone_raw = evt.get("tone", "neutral")
            if isinstance(tone_raw, str):
                tone_raw = (tone_raw,)
            elif isinstance(tone_raw, list):
                tone_raw = tuple(tone_raw)

            nonverbal_raw = evt.get("nonverbal", ())
            if isinstance(nonverbal_raw, list):
                nonverbal_raw = tuple(nonverbal_raw)

            lecture_events.append(LectureEvent(
                global_order=evt.get("global_order", i),
                local_order=evt.get("local_order", i),
                speaker_role=evt.get("role", "professor"),
                speaker_name=evt.get("speaker", ""),
                target=evt.get("target", "audience"),
                text=evt.get("text", ""),
                tone=tone_raw,
                volume=evt.get("volume"),
                pace=evt.get("pace"),
                pause_before=evt.get("pause_before"),
                pause_after=evt.get("pause_after"),
                interruption=False,
                nonverbal=nonverbal_raw,
                event_kind=evt.get("event_kind", "narration"),
                segment_id=evt.get("chapter_id"),
            ))

        # Group events into sections.
        section_map: dict[str, list[LectureEvent]] = {}
        for evt in lecture_events:
            if evt.segment_id:
                section_map.setdefault(evt.segment_id, []).append(evt)

        prof_name = config.professor.name if config.professor else ""
        lecture_sections: list[LectureSectionModel] = []
        for idx, sec_cfg in enumerate(config.sections):
            s_id = f"{self._lecture_id}_sec_{idx}"
            evts = section_map.get(s_id, [])
            lecture_sections.append(LectureSectionModel(
                section_id=s_id,
                section_order=idx,
                title=sec_cfg.title,
                section_type=sec_cfg.section_type,
                learning_objective=sec_cfg.learning_objective,
                professor_name=prof_name,
                events=tuple(evts),
            ))

        transcript_lines = [
            f"[{evt.get('speaker', '?')}]: {evt.get('text', '')}"
            for evt in raw_events
        ]

        return LectureConductorResult(
            config=config,
            events=raw_events,
            transcript=transcript_lines,
            total_llm_calls=self._total_session_calls + self._llm_calls,
            director_notes=[self._director_note] if self._director_note else [],
            lecture_events=lecture_events,
            lecture_sections=lecture_sections,
            lecture_id=self._lecture_id,
        )

    # ═══════════════════════════════════════════════════════════════════
    # PROMPT BUILDING
    # ═══════════════════════════════════════════════════════════════════

    def _build_professor_prompt(self) -> str:
        """Build the system prompt for the professor."""
        config = self.config
        if not isinstance(config, LectureConfig):
            raise TypeError(f"Expected LectureConfig, got {type(config)}")

        prof = config.professor
        if not prof:
            return ""

        title_clause = f"{prof.title}, " if prof.title else ""
        expertise_str = ", ".join(prof.expertise) if prof.expertise else (
            config.subject_area or "general scholarship"
        )

        return PROFESSOR_SYSTEM_PROMPT.format(
            name=prof.name,
            title_clause=title_clause,
            topic=config.topic or config.title or "(unspecified)",
            subject_area=config.subject_area or "general",
            setting=config.setting or "(intimate seminar)",
            style=prof.style or "warm, methodical, fond of analogies",
            voice_fingerprint=prof.voice_fingerprint or "(measured, clear, teacherly)",
            expertise=expertise_str,
        )

    # ═══════════════════════════════════════════════════════════════════
    # SECTION GENERATION
    # ═══════════════════════════════════════════════════════════════════

    def _generate_section(
        self,
        *,
        section: LectureSectionConfig,
        section_number: int,
        total_sections: int,
        global_order_start: int,
    ) -> list[dict]:
        """Generate audio events for one lecture section."""
        config = self.config
        if not isinstance(config, LectureConfig):
            raise TypeError(f"Expected LectureConfig, got {type(config)}")

        s_id = f"{self._lecture_id}_sec_{section_number - 1}"
        professor = config.professor

        if section.key_points:
            key_points_block = "\n".join(
                f"  {i+1}. {p}" for i, p in enumerate(section.key_points)
            )
        else:
            key_points_block = "  (Develop this section from the learning objective.)"

        if section.examples:
            examples_block = "\n".join(f"  - {e}" for e in section.examples)
        else:
            examples_block = "  (Use illustrations of your own choosing.)"

        reference_section = ""
        if config.reference_material:
            reference_section = (
                f"REFERENCE MATERIAL:\n{config.reference_material}\n"
            )

        type_overlay = OVERLAY_BY_TYPE.get(section.section_type, "")

        num_beats = max(1, section.target_beats)
        events: list[dict] = []

        # Beat labels
        if num_beats == 1:
            beat_labels = ["full"]
        elif num_beats == 2:
            beat_labels = ["opening", "body"]
        else:
            beat_labels = (
                ["opening"]
                + [f"body_{i}" for i in range(1, num_beats - 1)]
                + ["closing"]
            )

        points_per_beat = self._distribute_points(section.key_points, num_beats)

        for beat_idx, beat_label in enumerate(beat_labels):
            min_s, max_s = DURATION_SENTENCES.get(section.duration_hint, (6, 12))
            duration_instruction = f"Aim for {min_s}-{max_s} sentences."

            remaining_sections = total_sections - section_number + 1
            remaining_beats = remaining_sections * num_beats - beat_idx
            words_per_beat = 0
            if config.target_words and num_beats > 0:
                words_per_section = config.target_words // total_sections
                words_per_beat = words_per_section // num_beats
                words_per_beat = self._rebalanced_target(words_per_beat, remaining_beats)
                # Q-5: clamp to per-beat hard cap.
                words_per_beat = min(words_per_beat, MAX_BEAT_WORDS)
                duration_instruction += "\n" + self._length_contract_block(
                    words_per_beat, tolerance_pct=10.0,
                )

            # Q-5: unconditional per-beat cap, even without target_words.
            duration_instruction += (
                f"\nHARD CAP: this beat must not exceed {MAX_BEAT_WORDS} words."
            )

            budget_status = self._budget_status_for_prompt(
                remaining_beats, total_steps=total_sections * num_beats,
            )
            if budget_status:
                duration_instruction += f"\n{budget_status}"

            beat_points = points_per_beat[beat_idx] if beat_idx < len(points_per_beat) else []
            points_hint = ""
            if beat_points:
                points_hint = (
                    " Focus on these teaching beats:\n"
                    + "\n".join(f"  - {p}" for p in beat_points)
                    + "\n"
                )

            if beat_label == "full":
                beat_instruction = (
                    f"Teach this section start-to-finish. Hit the learning "
                    f"objective. Use the type overlay above to pick the "
                    f"pedagogical mode. {points_hint}{duration_instruction}"
                )
            elif beat_label == "opening":
                beat_instruction = (
                    f"Open this section — introduce the idea and hook attention. "
                    f"{points_hint}{duration_instruction}"
                )
            elif beat_label == "closing":
                beat_instruction = (
                    f"Close this section — recap what was taught and prepare "
                    f"the listener for what comes next. "
                    f"{points_hint}{duration_instruction}"
                )
            else:
                beat_instruction = (
                    f"Continue developing this section — deepen the explanation, "
                    f"add an example, or address a complication. Do NOT repeat "
                    f"earlier beats. {points_hint}{duration_instruction}"
                )

            user_message = SECTION_DELIVERY_PROMPT.format(
                section_number=section_number,
                total_sections=total_sections,
                title=section.title,
                section_type=section.section_type,
                learning_objective=section.learning_objective or "(no explicit objective)",
                key_points=key_points_block,
                examples=examples_block,
                type_overlay=type_overlay,
                reference_section=reference_section,
                beat_instruction=beat_instruction,
            )

            if self._director_note:
                user_message = f"DIRECTOR'S NOTE:\n{self._director_note}\n\n{user_message}"

            if self._professor_session is not None:
                raw_response = self._professor_session.send(user_message)
            else:
                raw_response = self._client.chat(
                    messages=[
                        {"role": "system", "content": self._professor_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
                self._llm_calls += 1

            event_dict = self._parse_response(raw_response, from_agent="professor")
            event_dict["speaker"] = professor.name if professor else "Professor"
            event_dict["role"] = "professor"
            event_dict["event_kind"] = "narration"
            event_dict["chapter_id"] = s_id
            event_dict["global_order"] = global_order_start + beat_idx
            event_dict["local_order"] = beat_idx
            event_dict["beat_label"] = beat_label

            if words_per_beat > 0:
                actual = len((event_dict.get("text") or "").split())
                warning = self._validate_word_count(
                    declared=event_dict.get("word_count"),
                    actual=actual,
                    target_words=words_per_beat,
                    tolerance_pct=10.0,
                )
                if warning:
                    self._length_warnings.append(
                        f"[sec {section_number}/{total_sections} "
                        f"beat {beat_label}] {warning}"
                    )

            beat_global_index = global_order_start + beat_idx
            self._checkpoint_segment(
                beat_global_index, event_dict.get("text") or "",
            )

            if beat_idx == 0:
                event_dict.setdefault("pause_before", "medium")
            else:
                event_dict.setdefault("pause_before", "short")

            if beat_idx == num_beats - 1:
                event_dict.setdefault("pause_after", "long")
            else:
                event_dict.setdefault("pause_after", "short")

            events.append(event_dict)

        return events

    @staticmethod
    def _distribute_points(key_points: list[str], num_beats: int) -> list[list[str]]:
        """Distribute key_points roughly evenly across beats."""
        if not key_points or num_beats <= 0:
            return [[] for _ in range(max(1, num_beats))]
        result: list[list[str]] = [[] for _ in range(num_beats)]
        for i, point in enumerate(key_points):
            result[i % num_beats].append(point)
        return result

    # ═══════════════════════════════════════════════════════════════════
    # RESPONSE PARSING
    # ═══════════════════════════════════════════════════════════════════

    def _parse_response(self, raw: str, *, from_agent: str | None = None) -> dict:
        """Parse the professor's JSON response into an event dict."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        try:
            data = json.loads(cleaned)
            return self._capture_issue_flag(data, from_agent=from_agent)
        except (json.JSONDecodeError, TypeError):
            return {
                "text": raw.strip()[:500],
                "tone": "neutral",
                "volume": "moderate",
                "pace": "medium",
                "pause_before": "none",
                "pause_after": "none",
            }

    # ═══════════════════════════════════════════════════════════════════
    # DIRECTOR CHECK
    # ═══════════════════════════════════════════════════════════════════

    def _director_check(self, section_number: int, total_sections: int) -> dict:
        """Ask the Director to review progress and issue a pedagogical note."""
        config = self.config
        if not isinstance(config, LectureConfig):
            return {}

        director = config.director_brief
        if not director:
            return {}

        recent = self._transcript[-DEFAULT_RECENT_WINDOW:]
        recent_text = "\n".join(
            f"[{t.get('speaker', '?')}]: {t.get('text', '')}" for t in recent
        )

        progression_str = (
            "\n".join(f"  - {p}" for p in director.progression)
            if director.progression
            else "(no specific progression)"
        )

        prompt = DIRECTOR_CHECK_PROMPT.format(
            section_number=section_number,
            total_sections=total_sections,
            thesis=director.thesis or "(no specific thesis)",
            progression=progression_str,
            closing_strategy=director.closing_strategy or "(natural ending)",
            recent_content=recent_text or "(lecture just started)",
        )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": "You are a pedagogical director. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        self._llm_calls += 1

        result = self._parse_response(raw)
        previous_note = self._director_note
        mood = result.get("mood_note", "")
        nudge = result.get("nudge", "")
        self._director_note = f"{mood} {nudge}".strip() if (mood or nudge) else ""
        if self._director_note != previous_note:
            self._broadcast_director_note(
                self._director_note,
                summary="Director updated lecture guidance.",
            )

        return result
