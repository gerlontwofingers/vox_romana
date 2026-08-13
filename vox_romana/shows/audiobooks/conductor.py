"""The Audiobook Conductor — runtime engine that generates an audiobook.

This is the audiobook counterpart to DebateConductor and NewsConductor.
It takes a fully populated AudiobookConfig and generates a complete
audiobook by orchestrating a narrator agent chapter-by-chapter.

STRUCTURAL DIFFERENCE FROM OTHER SHOW TYPES:

    Debate:  Reactive. Multiple agents argue. Turn order is interleaved.
    News:    Segment-based. Anchors deliver independent stories in sequence.
    Audiobook: SCRIPTED. A single narrator generates content from an outline,
               chapter by chapter. Continuity matters — each chapter builds
               on what came before.

THE CHAPTER LOOP:

    For each chapter in the outline:
    1. Narrator delivers the chapter content (1-N beats depending on length)
    2. Director checks in periodically for pacing adjustments
    3. Move to next chapter

CONTEXT MANAGEMENT:

    The narrator maintains an AgentSession with sliding-window compression,
    so earlier chapter content is summarized as the book progresses. This
    ensures continuity without exceeding context limits.

USAGE:

    # From a config:
    conductor = AudiobookConductor(config, client=client)
    result = conductor.run()

    # One-shot from brief:
    conductor = AudiobookConductor.from_brief("A short sci-fi story about first contact")
    result = conductor.run()

    # Save:
    merged = result.to_merged_audiobook()
    AudiobookSerializer().save(merged, "output/audiobook.json")
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from ...core.client import DeepSeekClient
from .bootstrap import AudiobookBootstrap
from ...core.conductor import BaseConductor, ConductorResult
from ...core.content_mode import ContentModeClassifier
from ...core.fact_checker import FactChecker
from ...core.session import AgentSession
from .config import AudiobookConfig, ChapterConfig
from .models import AudiobookChapter, AudiobookEvent, MergedAudiobook
from .prompts import (
    CHAPTER_DELIVERY_PROMPT,
    DIRECTOR_CHECK_PROMPT,
    NARRATOR_SYSTEM_PROMPT,
)


# How many delivered chapter transcripts to keep as verbatim context.
DEFAULT_RECENT_WINDOW = 4

# Duration hints map to approximate sentence counts per beat.
DURATION_SENTENCES = {
    "short": (3, 6),
    "medium": (6, 12),
    "long": (12, 20),
}


# ═══════════════════════════════════════════════════════════════════════════
# AudiobookConductorResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AudiobookConductorResult(ConductorResult):
    """Typed result from an AudiobookConductor run.

    Adds:
        audiobook_events:   List of typed AudiobookEvent objects.
        audiobook_chapters: List of completed AudiobookChapter objects.
        audiobook_id:       Unique ID for this audiobook.
    """

    audiobook_events: list[AudiobookEvent] = field(default_factory=list)
    audiobook_chapters: list[AudiobookChapter] = field(default_factory=list)
    audiobook_id: str = ""

    def to_merged_audiobook(self) -> MergedAudiobook:
        """Convert this result into a MergedAudiobook for serialization."""
        config = self.config
        if not isinstance(config, AudiobookConfig):
            raise TypeError(f"Expected an AudiobookConfig, got {type(config)}")

        # Build voice casting from narrator and character profiles.
        voice_casting: dict[str, str] = {}
        voice_hints: dict[str, str] = {}
        tts_styles: dict[str, str] = {}

        if config.narrator:
            if config.narrator.voice_gender:
                voice_casting[config.narrator.name] = config.narrator.voice_gender
            if config.narrator.voice_hint:
                voice_hints[config.narrator.name] = config.narrator.voice_hint
            if config.narrator.tts_style:
                tts_styles[config.narrator.name] = config.narrator.tts_style

        for char in config.characters:
            if char.voice_gender:
                voice_casting[char.name] = char.voice_gender
            if char.voice_hint:
                voice_hints[char.name] = char.voice_hint
            if char.tts_style:
                tts_styles[char.name] = char.tts_style

        return MergedAudiobook(
            audiobook_id=self.audiobook_id,
            title=config.title or "Untitled Audiobook",
            chapters=tuple(self.audiobook_chapters),
            events=tuple(self.audiobook_events),
            narrator=config.narrator.name if config.narrator else "",
            characters=tuple(c.name for c in config.characters),
            genre=config.genre,
            tone=config.tone,
            chapter_count=len(self.audiobook_chapters),
            voice_casting=voice_casting,
            voice_hints=voice_hints,
            tts_styles=tts_styles,
            language=config.language,
            music_style=config.music_style,
        )


# ═══════════════════════════════════════════════════════════════════════════
# THE AUDIOBOOK CONDUCTOR
# ═══════════════════════════════════════════════════════════════════════════

class AudiobookConductor(BaseConductor):
    """Generates a complete audiobook by orchestrating a narrator agent.

    The chapter loop walks through the outline in order, generating narration
    beat by beat. Unlike debates (reactive) or news (segment-independent),
    audiobook chapters build on each other — continuity is critical.
    """

    _bootstrap_cls = AudiobookBootstrap

    def __init__(
        self,
        config: AudiobookConfig,
        *,
        client: DeepSeekClient | None = None,
        api_key: str | None = None,
        **base_kwargs,
    ) -> None:
        resolved_client = client or DeepSeekClient(api_key=api_key)
        super().__init__(config, client=resolved_client, **base_kwargs)

        # Populated by _build_agents():
        self._narrator_prompt: str = ""
        self._narrator_session: AgentSession | None = None

        # Runtime state:
        self._transcript: list[dict] = []
        self._director_note: str = ""
        self._llm_calls: int = 0
        self._audiobook_id: str = ""
        self._content_mode: str = "factual"
        self._fact_checker: FactChecker | None = None
        self._approved_chapter_texts: list[str] = []

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
    ) -> AudiobookConductor:
        """Bootstrap an AudiobookConfig from a brief and create a conductor.

        Thin wrapper over :meth:`BaseConductor.from_brief`.
        """
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
        """Build the narrator system prompt and session."""
        config = self.config
        if not isinstance(config, AudiobookConfig):
            raise TypeError(f"AudiobookConductor requires AudiobookConfig, got {type(config)}")

        self._narrator_prompt = self._build_narrator_prompt()

        # Create an AgentSession for the narrator with sliding-window memory.
        self._narrator_session = self._create_session(
            self._narrator_prompt,
            recent_window=DEFAULT_RECENT_WINDOW,
            session_name="narrator",
        )

        # Classify content mode and build FactChecker (only when needed).
        if config.fact_check != "off":
            # Use explicitly-set content_mode; only call classifier when
            # config.content_mode is still the default ("factual") meaning
            # the user hasn't overridden it and auto-detection should run.
            if config.content_mode != "factual":
                self._content_mode = config.content_mode
            else:
                self._content_mode = ContentModeClassifier(self._client).classify(
                    brief=config.brief,
                    show_type="audiobook",
                )
            self._fact_checker = FactChecker.build_default(self._client)
        else:
            self._content_mode = "factual"
            self._fact_checker = None
        self._approved_chapter_texts = []

    def _preflight_check_bootstrap(self) -> None:
        """Fact-check the bootstrap output before generation begins.

        Scans chapter titles/synopses and the director arc for fabricated
        claims and aborts (or warns) based on
        ``config.allow_bootstrap_fabrications``.
        """
        if self._fact_checker is None:
            return
        config = self.config
        if not isinstance(config, AudiobookConfig):
            return
        surfaces: list[str] = []
        if config.topic:
            surfaces.append(config.topic)
        for ch in config.chapters:
            if ch.title:
                surfaces.append(ch.title)
            if ch.synopsis:
                surfaces.append(ch.synopsis)
        if config.director_brief and config.director_brief.narrative_arc:
            surfaces.append(config.director_brief.narrative_arc)
        bootstrap_text = " ".join(surfaces).strip()
        if not bootstrap_text:
            return
        try:
            report = self._fact_checker.check_content(
                bootstrap_text,
                content_id="bootstrap",
                reference_material=config.reference_material,
                mode=config.fact_check,
                content_mode=self._content_mode,
                perspective=config.perspective,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("AudiobookConductor: bootstrap preflight raised: %r", exc)
            return
        if report.has_fabrications:
            print(
                "\n\u26a0  BOOTSTRAP PRE-FLIGHT: Fabricated facts detected in "
                "the generated outline."
            )
            print(
                "   The fact-checker flagged the following based on LLM knowledge:"
            )
            for r in report.fabricated_claims:
                print(f"   \u2022 {r.claim}")
                print(f"     \u2192 {r.explanation}")
            if not config.allow_bootstrap_fabrications:
                print(
                    "   Generation aborted. Fix your brief or supply "
                    "--reference-material."
                )
                print(
                    "   Use --allow-bootstrap-fabrications to override "
                    "(not recommended).\n"
                )
                raise SystemExit(1)
            else:
                print(
                    "   Continuing anyway (--allow-bootstrap-fabrications is set).\n"
                )

    def _run_turn_loop(self) -> list[dict]:
        """Run the chapter-by-chapter generation loop.

        Returns:
            A list of raw event dicts (all chapters).
        """
        config = self.config
        if not isinstance(config, AudiobookConfig):
            raise TypeError(f"AudiobookConductor requires AudiobookConfig, got {type(config)}")

        self._audiobook_id = str(uuid.uuid4())[:8]
        all_events: list[dict] = []
        global_order = 0
        chapters = config.chapters
        total_chapters = len(chapters)

        # ── Auto-scale beats to hit target_words ─────────────────────
        if config.target_words and total_chapters > 0:
            words_per_beat = 200  # Audiobooks produce denser prose per beat
            total_beats_needed = max(total_chapters, config.target_words // words_per_beat)
            weight_map = {"short": 1, "medium": 2, "long": 3}
            total_weight = sum(weight_map.get(ch.duration_hint, 2) for ch in chapters)
            for ch in chapters:
                w = weight_map.get(ch.duration_hint, 2)
                derived = max(1, round(total_beats_needed * w / total_weight))
                # Use the word-budget-derived count as a floor — never reduce
                # what the architect requested, but always boost when the
                # budget demands more beats than the architect planned.
                # Previously guarded by `if ch.target_beats <= 1` which
                # silently skipped chapters whose architect default was 2.
                ch.target_beats = max(ch.target_beats, derived)

        # ── Chapter loop ─────────────────────────────────────────────
        for ch_idx, chapter in enumerate(chapters):
            ch_num = ch_idx + 1

            ch_events = self._generate_chapter(
                chapter=chapter,
                chapter_number=ch_num,
                total_chapters=total_chapters,
                global_order_start=global_order,
            )

            for evt in ch_events:
                all_events.append(evt)
                self._transcript.append(evt)
                global_order += 1
                # ── Adaptive duration tracking ───────────────────────
                self._record_words(evt.get("text", ""))

            # ── Check if we've hit the budget ────────────────────────
            if self._over_budget():
                break

            # Director check-in at intervals.
            director_brief = config.director_brief
            if (
                director_brief
                and ch_num > 0
                and ch_num % director_brief.check_interval == 0
            ):
                self._director_check(ch_num, total_chapters)

        return all_events

    # ═══════════════════════════════════════════════════════════════════
    # EXPANSION PASS — narrative continuation, not meta
    # ═══════════════════════════════════════════════════════════════════

    def _expansion_pass(
        self,
        *,
        deficit_words: int,
        deficit_seconds: float,
    ) -> list[dict]:
        """Add narrative continuation to close a length deficit.

        For audiobooks the bonus content MUST be in-fiction prose — not a
        meta "and that's the end of chapter X" recap. We extend the final
        chapter with a continuation beat that picks up where the narrator
        left off, in the same voice, same tense, same scene logic.
        """
        if self._narrator_session is None:
            return []
        config = self.config
        if not isinstance(config, AudiobookConfig):
            return []
        if not config.chapters:
            return []

        last_chapter = config.chapters[-1]
        ch_id = f"ch_bonus_{len(config.chapters):03d}"

        length_contract = self._length_contract_block(
            deficit_words, tolerance_pct=10.0,
        )
        beat_instruction = (
            "Continue the narrative from exactly where the previous chapter "
            "left off. This is NOT a recap, summary, or epilogue — write the "
            "next stretch of the story/argument in the same voice, same "
            "tense, same diegetic register. Stay in-world. Add specific "
            "imagery, dialogue, or detail. End at a natural breath point."
            f"\n\n{length_contract}"
        )

        user_message = CHAPTER_DELIVERY_PROMPT.format(
            chapter_number=len(config.chapters) + 1,
            total_chapters=len(config.chapters) + 1,
            title=f"{last_chapter.title} — continued",
            synopsis=last_chapter.synopsis or "(continue the prior chapter)",
            key_points="(continue the prior beats; do not restart)",
            chapter_tone=last_chapter.tone or config.tone or "engaging",
            reference_section="",
            beat_instruction=beat_instruction,
        )

        try:
            raw_response = self._narrator_session.send(user_message)
        except Exception as exc:  # noqa: BLE001
            self._length_warnings.append(
                f"[expansion ch_bonus] LLM call failed: {exc!r}"
            )
            return []

        event_dict = self._parse_response(raw_response, from_agent="narrator")
        narrator_name = (
            config.narrator.name if config.narrator else "Narrator"
        )
        event_dict["speaker"] = narrator_name
        event_dict["role"] = "narrator"
        event_dict["event_kind"] = "narration"
        event_dict["chapter_id"] = ch_id
        event_dict["beat_label"] = "continuation"
        event_dict.setdefault("pause_before", "long")
        event_dict.setdefault("pause_after", "long")
        self._record_words(event_dict.get("text", ""))
        return [event_dict]

    def _package_result(self, raw_events: list[dict]) -> AudiobookConductorResult:
        """Convert raw events into a typed AudiobookConductorResult."""
        config = self.config
        if not isinstance(config, AudiobookConfig):
            raise TypeError(f"AudiobookConductor requires AudiobookConfig, got {type(config)}")

        # Build typed AudiobookEvent objects.
        audiobook_events = []
        for i, evt in enumerate(raw_events):
            tone_raw = evt.get("tone", "neutral")
            if isinstance(tone_raw, str):
                tone_raw = (tone_raw,)
            elif isinstance(tone_raw, list):
                tone_raw = tuple(tone_raw)

            nonverbal_raw = evt.get("nonverbal", ())
            if isinstance(nonverbal_raw, list):
                nonverbal_raw = tuple(nonverbal_raw)

            audiobook_events.append(AudiobookEvent(
                global_order=evt.get("global_order", i),
                local_order=evt.get("local_order", i),
                speaker_role=evt.get("role", "narrator"),
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

        # Group events into chapters.
        chapter_map: dict[str, list[AudiobookEvent]] = {}
        for evt in audiobook_events:
            if evt.segment_id:
                chapter_map.setdefault(evt.segment_id, []).append(evt)

        audiobook_chapters = []
        for ch_cfg in config.chapters:
            ch_id = f"{self._audiobook_id}_ch_{config.chapters.index(ch_cfg)}"
            evts = chapter_map.get(ch_id, [])
            narrator_name = ""
            if config.narrator:
                narrator_name = config.narrator.name
            audiobook_chapters.append(AudiobookChapter(
                chapter_id=ch_id,
                chapter_order=config.chapters.index(ch_cfg),
                title=ch_cfg.title,
                tone=ch_cfg.tone,
                narrator_name=narrator_name,
                events=tuple(evts),
            ))

        transcript_lines = [
            f"[{evt.get('speaker', '?')}]: {evt.get('text', '')}"
            for evt in raw_events
        ]

        return AudiobookConductorResult(
            config=config,
            events=raw_events,
            transcript=transcript_lines,
            total_llm_calls=self._total_session_calls + self._llm_calls,
            director_notes=[self._director_note] if self._director_note else [],
            audiobook_events=audiobook_events,
            audiobook_chapters=audiobook_chapters,
            audiobook_id=self._audiobook_id,
        )

    # ═══════════════════════════════════════════════════════════════════
    # PROMPT BUILDING
    # ═══════════════════════════════════════════════════════════════════

    def _build_narrator_prompt(self) -> str:
        """Build the system prompt for the narrator."""
        config = self.config
        if not isinstance(config, AudiobookConfig):
            raise TypeError(f"Expected AudiobookConfig, got {type(config)}")

        narrator = config.narrator
        if not narrator:
            return ""

        return NARRATOR_SYSTEM_PROMPT.format(
            name=narrator.name,
            genre=config.genre or "general",
            title=config.title or "Untitled",
            setting=config.setting or "(not specified)",
            style=narrator.style or "engaging and clear",
            voice_fingerprint=narrator.voice_fingerprint or "(no specific voice style)",
        )

    # ═══════════════════════════════════════════════════════════════════
    # CHAPTER GENERATION
    # ═══════════════════════════════════════════════════════════════════

    def _generate_chapter(
        self,
        *,
        chapter: ChapterConfig,
        chapter_number: int,
        total_chapters: int,
        global_order_start: int,
    ) -> list[dict]:
        """Generate audio events for one audiobook chapter.

        The narrator delivers the chapter in one or more beats:
          - 1 beat:   Short chapter — single delivery.
          - 2 beats:  Opening + body.
          - 3+ beats: Opening → body sections → closing.
        """
        config = self.config
        if not isinstance(config, AudiobookConfig):
            raise TypeError(f"Expected AudiobookConfig, got {type(config)}")

        ch_id = f"{self._audiobook_id}_ch_{chapter_number - 1}"
        narrator = config.narrator

        # Build key points block.
        if chapter.key_points:
            key_points_block = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(chapter.key_points))
        else:
            key_points_block = "  (Follow the synopsis to develop this chapter)"

        # Build reference section.
        reference_section = ""
        if config.reference_material:
            reference_section = (
                f"REFERENCE MATERIAL:\n{config.reference_material}\n"
            )

        num_beats = max(1, chapter.target_beats)
        events: list[dict] = []

        # ── Beat label assignment ──────────────────────────────────────
        # Each chapter is generated as a sequence of "beats" — discrete
        # narration calls to the LLM. We label each beat to give the
        # narrator agent positional context inside the chapter so it can
        # adjust pacing (set the scene early, deepen mid-chapter, land a
        # close at the end). Pattern by beat count:
        #
        #     1 beat:  ["full"]                              # whole chapter in one call
        #     2 beats: ["opening", "body"]                   # setup + payoff
        #     N beats: ["opening", "body_1", … "body_{N-2}", "closing"]
        #
        # The body indices start at 1 (not 0) and stop at `num_beats - 1`
        # (exclusive upper bound) so we leave the final slot for "closing".
        # The labels stay internal to the conductor. They choose which
        # natural-language instruction block to send, but the raw labels
        # themselves should not leak into the narrator prompt.
        if num_beats == 1:
            beat_labels = ["full"]
        elif num_beats == 2:
            beat_labels = ["opening", "body"]
        else:
            beat_labels = ["opening"] + [f"body_{i}" for i in range(1, num_beats - 1)] + ["closing"]

        # ── Key-point distribution ─────────────────────────────────────
        # `_distribute_points` slices `chapter.key_points` across the beats
        # so each beat carries roughly `len(key_points) / num_beats` points
        # (with leftovers absorbed by the earlier beats). The result is a
        # list-of-lists indexed by `beat_idx`, used inside the loop below
        # to scope each beat's prompt to its share of plot/argument beats.
        points_per_beat = self._distribute_points(chapter.key_points, num_beats)

        for beat_idx, beat_label in enumerate(beat_labels):
            min_s, max_s = DURATION_SENTENCES.get(chapter.duration_hint, (6, 12))
            duration_instruction = f"Aim for {min_s}-{max_s} sentences."

            # Add word budget hint when target_words is set.
            remaining_chapters = total_chapters - chapter_number + 1
            remaining_beats = remaining_chapters * num_beats - beat_idx
            words_per_beat = 0
            if config.target_words and num_beats > 0:
                words_per_chapter = config.target_words // total_chapters
                words_per_beat = words_per_chapter // num_beats
                words_per_beat = self._rebalanced_target(words_per_beat, remaining_beats)
                duration_instruction += "\n" + self._length_contract_block(
                    words_per_beat, tolerance_pct=10.0,
                )

            # Append runtime budget status for the LLM.
            budget_status = self._budget_status_for_prompt(
                remaining_beats, total_steps=total_chapters * num_beats,
            )
            if budget_status:
                duration_instruction += f"\n{budget_status}"

            # Build beat-specific instruction.
            beat_points = points_per_beat[beat_idx] if beat_idx < len(points_per_beat) else []
            points_hint = ""
            if beat_points:
                points_hint = (
                    " Focus on these narrative beats:\n"
                    + "\n".join(f"  - {p}" for p in beat_points)
                    + "\n"
                )

            if beat_label == "full":
                beat_instruction = (
                    f"Narrate this chapter from start to finish. "
                    f"Cover all the key points in order, with vivid prose. "
                    f"{points_hint}{duration_instruction}"
                )
            elif beat_label == "opening":
                beat_instruction = (
                    f"Begin this chapter by setting the scene, establishing "
                    f"the mood, and drawing the listener in. "
                    f"{points_hint}{duration_instruction}"
                )
            elif beat_label == "closing":
                beat_instruction = (
                    f"Land this chapter at a satisfying pause point, with a "
                    f"hook or reflection "
                    f"for what comes next. "
                    f"{points_hint}{duration_instruction}"
                )
            else:
                beat_instruction = (
                    f"Continue this chapter by developing the narrative, "
                    f"deepen the story or argument. Do NOT repeat what you "
                    f"already covered in earlier beats. "
                    f"{points_hint}{duration_instruction}"
                )

            user_message = CHAPTER_DELIVERY_PROMPT.format(
                chapter_number=chapter_number,
                total_chapters=total_chapters,
                title=chapter.title,
                synopsis=chapter.synopsis or "(develop from key points)",
                key_points=key_points_block,
                chapter_tone=chapter.tone or config.tone or "engaging",
                reference_section=reference_section,
                beat_instruction=beat_instruction,
            )

            # Add director note if present.
            if self._director_note:
                user_message = f"DIRECTOR'S NOTE:\n{self._director_note}\n\n{user_message}"

            # Send via the narrator's AgentSession.
            if self._narrator_session is not None:
                raw_response = self._narrator_session.send(user_message)
            else:
                raw_response = self._client.chat(
                    messages=[
                        {"role": "system", "content": self._narrator_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
                self._llm_calls += 1

            event_dict = self._parse_response(raw_response, from_agent="narrator")
            event_dict["speaker"] = narrator.name if narrator else "Narrator"
            event_dict["role"] = "narrator"
            event_dict["event_kind"] = "narration"
            event_dict["chapter_id"] = ch_id
            event_dict["global_order"] = global_order_start + beat_idx
            event_dict["local_order"] = beat_idx
            event_dict["beat_label"] = beat_label

            # ── Length contract validation ──────────────
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
                        f"[ch {chapter_number}/{total_chapters} "
                        f"beat {beat_label}] {warning}"
                    )

            # ── Checkpoint render: measure THIS beat's audio length ──
            beat_global_index = global_order_start + beat_idx
            self._checkpoint_segment(
                beat_global_index, event_dict.get("text") or "",
            )

            # Pauses: longer between chapters, shorter between beats.
            if beat_idx == 0:
                event_dict.setdefault("pause_before", "medium")
            else:
                event_dict.setdefault("pause_before", "short")

            if beat_idx == num_beats - 1:
                event_dict.setdefault("pause_after", "long")
            else:
                event_dict.setdefault("pause_after", "short")

            events.append(event_dict)

        # ── Fact-check gate (whole chapter, blocking) ─────────────────
        if (
            self._fact_checker is not None
            and isinstance(config, AudiobookConfig)
            and config.fact_check != "off"
        ):
            ch_text = " ".join(
                (e.get("text") or "").strip() for e in events
            ).strip()
            for fc_attempt in range(config.max_retries):
                try:
                    fc_report = self._fact_checker.check_content(
                        ch_text,
                        content_id=ch_id,
                        reference_material=config.reference_material,
                        mode=config.fact_check,
                        content_mode=self._content_mode,
                        perspective=config.perspective,
                        prior_context=self._build_prior_context(),
                    )
                except Exception as exc:
                    logger.warning(
                        "AudiobookConductor: fact-check raised for %s: %r",
                        ch_id, exc,
                    )
                    break

                if fc_report.has_fabrications:
                    fc_report = self._challenge_fabrications(
                        fc_report,
                        generator_agent="narrator",
                        perspective=config.perspective,
                    )

                if not (fc_report.has_fabrications or fc_report.has_contradictions):
                    break  # Chapter passed.

                blocking = (
                    fc_report.fabricated_claims + fc_report.contradicted_claims
                )
                if fc_attempt < config.max_retries - 1:
                    claims_lines = "\n".join(
                        f"  • {r.claim}: {r.explanation}"
                        for r in blocking
                    )
                    rejection_msg = (
                        f"FACT-CHECK REJECTION (attempt {fc_attempt + 1}/"
                        f"{config.max_retries}): The previous chapter "
                        f"narration contained fabricated facts or contradictions "
                        f"with earlier content:\n{claims_lines}\n\n"
                        f"Please re-narrate this chapter. Remove all fabricated "
                        f"claims and ensure consistency with what has already "
                        f"been narrated."
                    )
                    print(
                        f"  [chapter {chapter_number}/{total_chapters}] "
                        f"fact-check fail ({len(blocking)} issue(s)) "
                        f"— retry {fc_attempt + 1}/{config.max_retries}",
                        flush=True,
                    )
                    try:
                        retry_raw = self._narrator_session.send(rejection_msg)
                        retry_event = self._parse_response(
                            retry_raw,
                            from_agent="narrator",
                        )
                        retry_event["speaker"] = (
                            narrator.name if narrator else "Narrator"
                        )
                        retry_event["role"] = "narrator"
                        retry_event["event_kind"] = "narration"
                        retry_event["chapter_id"] = ch_id
                        # Replace last beat with the regenerated version.
                        if events:
                            events[-1] = retry_event
                        ch_text = " ".join(
                            (e.get("text") or "").strip() for e in events
                        ).strip()
                    except Exception as exc:
                        logger.warning(
                            "AudiobookConductor: fact-check retry failed "
                            "for %s: %r", ch_id, exc,
                        )
                        break
                else:
                    self._report_issue_to_director(
                        from_agent="narrator",
                        summary=(
                            f"Could not resolve fact-check objections for {ch_id} "
                            f"after {config.max_retries} attempts."
                        ),
                        proposed_remediation=(
                            "Shift the next chapter toward reference-backed narration "
                            "and reduce disputed specifics."
                        ),
                    )
                    for r in blocking:
                        self._length_warnings.append(
                            f"[{ch_id}] FACT_CHECK_WARN: Published with "
                            f"unresolved claim after "
                            f"{config.max_retries} attempts: {r.claim}"
                        )
                    print(
                        f"  [chapter {chapter_number}/{total_chapters}] "
                        f"fact-check EXHAUSTED — {len(blocking)} issue(s) remain",
                        flush=True,
                    )
                    break

            # Record approved chapter text for future consistency checks.
            if ch_text:
                self._approved_chapter_texts.append(ch_text)

        return events

    def _build_prior_context(self) -> str | None:
        """Return concatenated approved chapter texts for consistency checking."""
        if not self._approved_chapter_texts:
            return None
        return "\n---\n".join(self._approved_chapter_texts)

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
        """Parse the narrator's JSON response into an event dict."""
        cleaned = raw.strip()

        # Strip markdown fences.
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        # Find JSON object.
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

    def _director_check(self, chapter_number: int, total_chapters: int) -> dict:
        """Ask the Director to review progress and issue a pacing note."""
        config = self.config
        if not isinstance(config, AudiobookConfig):
            return {}

        director = config.director_brief
        if not director:
            return {}

        # Current mood target.
        if director.mood_progression:
            progress = chapter_number / max(total_chapters, 1)
            mood_index = min(
                int(progress * len(director.mood_progression)),
                len(director.mood_progression) - 1,
            )
            current_mood = director.mood_progression[mood_index]
        else:
            current_mood = "neutral"

        # Recent content for context.
        recent = self._transcript[-DEFAULT_RECENT_WINDOW:]
        recent_text = "\n".join(
            f"[{t.get('speaker', '?')}]: {t.get('text', '')}" for t in recent
        )

        prompt = DIRECTOR_CHECK_PROMPT.format(
            chapter_number=chapter_number,
            total_chapters=total_chapters,
            narrative_arc=director.narrative_arc or "(no specific arc)",
            current_mood=current_mood,
            closing_strategy=director.closing_strategy or "(natural ending)",
            recent_content=recent_text or "(narration just started)",
        )
        if pending_issues := self._consume_pending_issue_reports():
            prompt = f"PENDING ISSUE REPORTS FROM CAST:\n{pending_issues}\n\n{prompt}"

        raw = self._client.chat(
            messages=[
                {
                    "role": "system",
                    "content": self._augmented_director_system_prompt(
                        "You are a narrative director. Respond with JSON only."
                    ),
                },
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
                summary="Director updated narrative guidance.",
            )

        return result
