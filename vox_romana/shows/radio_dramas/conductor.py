"""The Radio Drama Conductor — runtime engine that generates a radio drama.

This is the radio-drama counterpart to DebateConductor, PodcastConductor,
and AudiobookConductor. It takes a fully populated RadioDramaConfig and
generates a complete episode by orchestrating a single scriptwriter agent
that voices EVERY cast member scene by scene.

STRUCTURAL DIFFERENCE FROM OTHER SHOW TYPES:

    Debate:    Reactive. Multiple agents argue turn-by-turn.
    News:      Segment-based. Anchors deliver independent stories.
    Audiobook: Scripted. A single narrator reads chapter-by-chapter.
    Podcast:   Segment-level. One LLM call produces a multi-speaker
               conversational segment.
    RadioDrama: SCENE-LEVEL SCRIPTED FICTION. One LLM call produces an
               entire scene of multi-character dialogue (like the podcast's
               segment shape), but the content is plotted drama with an
               announcer cold-open/sign-off and era-styled music cues.

THE SCENE LOOP:

    For each scene in the outline:
    1. Build context: previous summary + director note for this scene
    2. Inject the SCENE_DELIVERY_PROMPT with era + announcer bookends
    3. Parse response: JSON array → list of RadioDramaEvent dicts
    4. Record scene summary for continuity
    5. Compress context periodically
    6. Director check-in at intervals

CONTEXT MANAGEMENT:

    Same sliding-window pattern as audiobooks and podcasts — scene
    summaries are kept verbatim for the most recent scenes and compressed
    into a short context block as the episode progresses.

USAGE:

    # One-shot from brief:
    conductor = RadioDramaConductor.from_brief(
        "A noir mystery set in 1940s Philadelphia",
        target_words=3500,
    )
    result = conductor.run()

    # From existing config:
    config = RadioDramaBootstrap().from_brief("...")
    conductor = RadioDramaConductor(config)
    result = conductor.run()

    # Save:
    merged = result.to_merged_radio_drama()
    RadioDramaSerializer.save(merged, "output/text/radio_dramas/noir.json")
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from dataclasses import dataclass, field

from ...core.client import DeepSeekClient
from .bootstrap import RadioDramaBootstrap
from ...core.conductor import BaseConductor, ConductorResult
from ...core.content_mode import ContentModeClassifier
from ...core.fact_checker import FactChecker
from ...core.session import AgentSession
from .config import (
    CharacterConfig,
    RadioDramaConfig,
    RadioDramaDirectorBrief,
    SceneConfig,
)
from .models import MergedRadioDrama, RadioDramaEvent, RadioDramaScene
from .prompts import (
    CONTEXT_SUMMARY_PROMPT,
    DIRECTOR_CHECK_PROMPT,
    SCENE_DELIVERY_PROMPT,
    SCRIPTWRITER_SYSTEM_PROMPT,
    era_guidance,
)

logger = logging.getLogger(__name__)


# ── Defaults ─────────────────────────────────────────────────────────────
# How many recent scene summaries to keep as verbatim context.
DEFAULT_RECENT_WINDOW = 4

# Fallback word targets by duration_hint when a scene has no target_words.
DURATION_WORD_TARGETS = {
    "short": 150,
    "medium": 350,
    "long": 550,
}

# When a scene comes back below this fraction of its word target, the
# conductor sends one continuation request to bring it up to spec.
SCENE_RETRY_THRESHOLD = 0.70

# Continuation prompt sent when a scene came back too short.
_SCENE_RETRY_TEMPLATE = (
    "Your previous scene was only {actual} words but the target was "
    "{target}. Continue the scene from EXACTLY where you left off — add "
    "roughly {deficit} more words of substantive new drama (more dialogue, "
    "a revelation, a new complication). Do NOT recap. Do NOT start over. "
    "Stay in the same JSON format."
)


# ═══════════════════════════════════════════════════════════════════════════
# RadioDramaConductorResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RadioDramaConductorResult(ConductorResult):
    """Typed result from a RadioDramaConductor run.

    Adds:
        drama_events:   List of typed RadioDramaEvent objects.
        drama_scenes:   List of completed RadioDramaScene objects.
        drama_id:       Unique ID for this episode.
    """

    drama_events: list[RadioDramaEvent] = field(default_factory=list)
    drama_scenes: list[RadioDramaScene] = field(default_factory=list)
    drama_id: str = ""

    def to_merged_radio_drama(self) -> MergedRadioDrama:
        """Convert this result into a MergedRadioDrama for serialization."""
        config = self.config
        if not isinstance(config, RadioDramaConfig):
            raise TypeError(
                f"Expected a RadioDramaConfig, got {type(config)}"
            )

        voice_casting: dict[str, str] = {}
        voice_hints: dict[str, str] = {}
        tts_styles: dict[str, str] = {}

        def _collect(character: CharacterConfig) -> None:
            if not character.name:
                return
            if character.voice_gender:
                voice_casting[character.name] = character.voice_gender
            if character.voice_hint:
                voice_hints[character.name] = character.voice_hint
            if character.tts_style:
                tts_styles[character.name] = character.tts_style

        if config.announcer:
            _collect(config.announcer)
        for character in config.characters:
            _collect(character)

        all_names = [config.announcer.name] if config.announcer else []
        all_names += [c.name for c in config.characters]

        return MergedRadioDrama(
            drama_id=self.drama_id,
            title=config.title or "Untitled Radio Drama",
            series_mode=config.series_mode,
            series_id=config.series_id,
            episode_number=config.episode_number,
            genre=config.genre,
            era=config.era,
            scenes=tuple(self.drama_scenes),
            events=tuple(self.drama_events),
            announcer=config.announcer.name if config.announcer else "",
            characters=tuple(n for n in all_names if n),
            voice_casting=voice_casting,
            voice_hints=voice_hints,
            tts_styles=tts_styles,
            language=config.language,
            music_style=config.music_style,
        )


# ═══════════════════════════════════════════════════════════════════════════
# THE RADIO DRAMA CONDUCTOR
# ═══════════════════════════════════════════════════════════════════════════

class RadioDramaConductor(BaseConductor):
    """Generates a complete radio drama by orchestrating a scriptwriter agent.

    The scene loop walks through the episode outline, generating each scene
    as a complete multi-character dialogue in a single LLM call. The
    announcer bookends the show (cold open + sign-off); music cues are
    emitted inline as ``[music: ...]`` lines that the TTS renderer routes
    to the music engine.
    """

    _bootstrap_cls = RadioDramaBootstrap

    def __init__(
        self,
        config: RadioDramaConfig,
        *,
        client: DeepSeekClient | None = None,
        api_key: str | None = None,
        **base_kwargs,
    ) -> None:
        resolved_client = client or DeepSeekClient(api_key=api_key)
        super().__init__(config, client=resolved_client, **base_kwargs)

        # Populated by _build_agents():
        self._system_prompt: str = ""
        self._session: AgentSession | None = None
        self._fact_checker: FactChecker | None = None
        self._content_mode: str = "factual"

        # Runtime state (reset each run):
        self._context_log: list[str] = []
        self._compressed_ctx: str = ""
        self._director_note: str = ""
        self._llm_calls: int = 0
        self._drama_id: str = ""
        self._approved_scene_texts: list[str] = []

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
    ) -> RadioDramaConductor:
        """Bootstrap a RadioDramaConfig from a brief and create a conductor.

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
    # BaseConductor interface implementation
    # ═══════════════════════════════════════════════════════════════════

    def _build_agents(self) -> None:
        """Build the scriptwriter system prompt and session.

        Like podcasts, radio drama uses ONE session that generates lines for
        ALL cast members (including the announcer). The system prompt is the
        full cast sheet + show bible + era guidance.
        """
        config = self.config
        if not isinstance(config, RadioDramaConfig):
            raise TypeError(
                f"RadioDramaConductor requires RadioDramaConfig, "
                f"got {type(config)}"
            )

        cast_descriptions = self._build_cast_descriptions(config)
        all_names = [config.announcer.name] if config.announcer else []
        all_names += [c.name for c in config.characters]
        character_names = ", ".join(f'"{n}"' for n in all_names if n)

        self._system_prompt = SCRIPTWRITER_SYSTEM_PROMPT.format(
            show_name=config.title or "Untitled Radio Drama",
            genre=config.genre or "drama",
            title=config.title or "Untitled",
            setting=config.setting or "(not specified)",
            era_guidance=era_guidance(config.era),
            cast_descriptions=cast_descriptions,
            character_names=character_names,
        )

        self._session = self._create_session(
            self._system_prompt,
            recent_window=DEFAULT_RECENT_WINDOW,
            session_name="radio_drama_cast",
        )

        if config.fact_check != "off":
            self._fact_checker = FactChecker.build_default(
                self._client,
                profile_text=self._crew_profiles.get("factchecker", ""),
            )
            if config.content_mode != "factual":
                self._content_mode = config.content_mode
            else:
                classifier = ContentModeClassifier(self._client)
                self._content_mode = classifier.classify(
                    brief=config.brief,
                    show_type="radio_drama",
                )
        else:
            self._fact_checker = None
            self._content_mode = config.content_mode

        # Reset runtime state.
        self._context_log = []
        self._compressed_ctx = ""
        self._director_note = ""
        self._llm_calls = 0
        self._drama_id = str(uuid.uuid4())[:8]
        self._approved_scene_texts = []

    def _run_turn_loop(self) -> list[dict]:
        """The scene-by-scene generation loop.

        For each scene in the config:
        1. Build context from previous scenes + director note
        2. Build the scene prompt (with announcer bookends where relevant)
        3. Send to the LLM (via AgentSession)
        4. Parse the JSON array response into event dicts
        5. Record scene summary for continuity
        6. Compress context periodically
        7. Director check-in at intervals
        """
        config = self.config
        if not isinstance(config, RadioDramaConfig):
            raise TypeError(
                f"RadioDramaConductor requires RadioDramaConfig, "
                f"got {type(config)}"
            )

        all_events: list[dict] = []
        scenes = config.scenes
        total_scenes = len(scenes)
        global_order = 0

        compress_interval = 5
        if config.director_brief:
            compress_interval = config.director_brief.check_interval

        for scene_idx, scene in enumerate(scenes):
            scene_id = f"scene_{scene_idx:03d}"

            print(
                f"  [scene {scene_idx + 1}/{total_scenes}] "
                f"{(scene.title or '')[:60]}",
                flush=True,
            )

            context_summary = self._build_context_summary()

            # ── Announcer bookends + director note ───────────────────
            guidance_parts: list[str] = []
            if self._director_note:
                guidance_parts.append(f"DIRECTOR'S NOTE: {self._director_note}")
            if scene_idx == 0:
                guidance_parts.append(
                    "ANNOUNCER COLD OPEN: begin the show with the announcer "
                    "naming the show, the episode, and the cast, followed by "
                    "an era-styled music cue, before the first scene's "
                    "dialogue begins."
                )
            if scene_idx == total_scenes - 1:
                guidance_parts.append(
                    "ANNOUNCER SIGN-OFF: end the show with the announcer's "
                    "sign-off and a final music cue."
                )
            director_note_str = "\n".join(guidance_parts)

            # ── Cliffhanger instruction for the final scene ──────────
            cliffhanger_instruction = ""
            if scene_idx == total_scenes - 1:
                if config.series_mode == "serial":
                    cliffhanger_instruction = (
                        "SERIAL CLIFFHANGER: end the final scene on a "
                        "cliffhanger hook that leaves the listener needing "
                        "the next episode — a new question, a reveal, or a "
                        "threat left unresolved."
                    )
                else:
                    cliffhanger_instruction = (
                        "PILOT RESOLUTION: resolve the story fully. Tie up "
                        "the central thread and let the final beat land."
                    )

            # ── Determine target words ───────────────────────────────
            target_words = scene.target_words
            if target_words is None:
                target_words = DURATION_WORD_TARGETS.get(
                    scene.duration_hint, 350
                )
            remaining_scenes = total_scenes - scene_idx
            target_words = self._rebalanced_target(target_words, remaining_scenes)

            runtime_status = self._budget_status_for_prompt(
                remaining_scenes, total_steps=total_scenes,
            )

            characters_present = ", ".join(scene.characters) or "(all cast)"

            length_contract = self._length_contract_block(target_words)
            user_message = SCENE_DELIVERY_PROMPT.format(
                scene_title=scene.title or f"Scene {scene_idx + 1}",
                location=scene.location or "(unspecified)",
                synopsis=scene.synopsis or "(develop from the story so far)",
                characters_present=characters_present,
                scene_tone=scene.synopsis or config.tone or "dramatic",
                length_contract=length_contract,
                context_summary=context_summary or "(This is the first scene.)",
                director_note=director_note_str,
                runtime_status=runtime_status,
                cliffhanger_instruction=cliffhanger_instruction,
            )

            if self._session is not None:
                raw_response = self._session.send(user_message)
            else:
                raw_response = self._client.chat(
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
                self._llm_calls += 1

            scene_events = self._parse_scene_response(
                raw_response,
                scene_id=scene_id,
                global_order_start=global_order,
            )

            actual_words = sum(
                len((evt.get("text") or "").split()) for evt in scene_events
            )

            # ── Retry once if the scene came back too short ──────────
            if (
                self._session is not None
                and target_words > 0
                and actual_words < target_words * SCENE_RETRY_THRESHOLD
            ):
                deficit = max(50, target_words - actual_words)
                retry_msg = _SCENE_RETRY_TEMPLATE.format(
                    actual=actual_words,
                    target=target_words,
                    deficit=deficit,
                )
                print(
                    f"  [scene {scene_idx + 1}/{total_scenes}] retry "
                    f"(was {actual_words}, target {target_words})",
                    flush=True,
                )
                try:
                    retry_raw = self._session.send(retry_msg)
                except Exception as exc:  # noqa: BLE001
                    self._length_warnings.append(
                        f"[scene {scene_idx + 1}] retry LLM call failed: {exc!r}"
                    )
                    retry_raw = ""
                if retry_raw:
                    extra_events = self._parse_scene_response(
                        retry_raw,
                        scene_id=scene_id,
                        global_order_start=global_order + len(scene_events),
                    )
                    scene_events.extend(extra_events)
                    actual_words = sum(
                        len((evt.get("text") or "").split())
                        for evt in scene_events
                    )

            warning = self._validate_word_count(
                declared=None,
                actual=actual_words,
                target_words=target_words,
            )
            if warning:
                self._length_warnings.append(
                    f"[scene {scene_idx + 1}/{total_scenes}] {warning}"
                )

            # ── Record events ────────────────────────────────────────
            for evt in scene_events:
                all_events.append(evt)
                global_order += 1
                self._record_words(evt.get("text", ""))

            scene_text = " ".join(
                (e.get("text") or "").strip() for e in scene_events
            ).strip()
            if scene_text:
                self._approved_scene_texts.append(scene_text)

            self._checkpoint_segment(scene_idx, scene_text)

            if self._over_budget():
                break

            summary = self._summarize_scene(scene_events, scene)
            self._context_log.append(summary)

            if (
                compress_interval > 0
                and len(self._context_log) > compress_interval
                and len(self._context_log) % compress_interval == 0
            ):
                self._compress_context()

            # ── Director check-in ────────────────────────────────────
            if (
                config.director_brief
                and scene_idx > 0
                and (scene_idx + 1) % config.director_brief.check_interval == 0
            ):
                self._director_check(scene_idx + 1, total_scenes)

        return all_events

    # ═══════════════════════════════════════════════════════════════════
    # EXPANSION PASS
    # ═══════════════════════════════════════════════════════════════════

    def _expansion_pass(
        self,
        *,
        deficit_words: int,
        deficit_seconds: float,
    ) -> list[dict]:
        """Generate one bonus scene to close a length deficit."""
        if self._session is None:
            return []
        config = self.config
        if not isinstance(config, RadioDramaConfig):
            return []

        scene_id = "scene_bonus"
        length_contract = self._length_contract_block(deficit_words)
        user_message = SCENE_DELIVERY_PROMPT.format(
            scene_title="Continuation",
            location="(as established)",
            synopsis=(
                "Extend the story from exactly where it left off. Add new "
                "substance — a fresh complication, a deeper revelation — in "
                "the same voices. Do NOT recap."
            ),
            characters_present="(all cast)",
            scene_tone=config.tone or "dramatic",
            length_contract=length_contract,
            context_summary=self._build_context_summary() or "(see prior scenes)",
            director_note=(
                "DIRECTOR'S NOTE: We need extra runtime — add real dramatic "
                "substance, not filler."
            ),
            runtime_status=(
                f"BONUS SCENE — episode is short by ~{deficit_seconds:.0f}s. "
                f"Target ~{deficit_words} additional words."
            ),
            cliffhanger_instruction="",
        )

        try:
            raw_response = self._session.send(user_message)
        except Exception as exc:  # noqa: BLE001
            self._length_warnings.append(
                f"[expansion] LLM call failed: {exc!r}"
            )
            return []

        new_events = self._parse_scene_response(
            raw_response,
            scene_id=scene_id,
            global_order_start=0,
        )
        for evt in new_events:
            self._record_words(evt.get("text", ""))
        return new_events

    def _package_result(self, raw_events: list[dict]) -> RadioDramaConductorResult:
        """Convert raw event dicts into a typed RadioDramaConductorResult."""
        config = self.config
        if not isinstance(config, RadioDramaConfig):
            raise TypeError(
                f"RadioDramaConductor requires RadioDramaConfig, "
                f"got {type(config)}"
            )

        drama_events: list[RadioDramaEvent] = []
        for i, evt in enumerate(raw_events):
            tone_raw = evt.get("tone", "neutral")
            if isinstance(tone_raw, str):
                tone_tuple = (tone_raw,)
            elif isinstance(tone_raw, list):
                tone_tuple = tuple(tone_raw)
            else:
                tone_tuple = ("neutral",)

            drama_events.append(RadioDramaEvent(
                global_order=evt.get("global_order", i),
                local_order=evt.get("local_order", i),
                speaker_role=evt.get("role", "character"),
                speaker_name=evt.get("speaker_name", evt.get("speaker", "Unknown")),
                target=evt.get("target"),
                text=evt.get("text", ""),
                tone=tone_tuple,
                volume=evt.get("volume"),
                pace=evt.get("pace"),
                pause_before=evt.get("pause_before"),
                pause_after=evt.get("pause_after"),
                interruption=evt.get("interruption", False),
                nonverbal=tuple(evt.get("nonverbal", [])),
                event_kind=evt.get("event_kind", "utterance"),
                segment_id=evt.get("scene_id"),
            ))

        scene_map: dict[str, list[RadioDramaEvent]] = {}
        for evt in drama_events:
            if evt.segment_id:
                scene_map.setdefault(evt.segment_id, []).append(evt)

        drama_scenes: list[RadioDramaScene] = []
        for scene_idx, scene_cfg in enumerate(config.scenes):
            scene_id = f"scene_{scene_idx:03d}"
            evts = scene_map.get(scene_id, [])
            drama_scenes.append(RadioDramaScene(
                scene_id=scene_id,
                scene_order=scene_idx + 1,
                title=scene_cfg.title,
                location=scene_cfg.location,
                characters=tuple(scene_cfg.characters),
                events=tuple(evts),
            ))

        transcript_lines = [
            f"[{evt.get('speaker_name', evt.get('speaker', '?'))}] "
            f"{evt.get('text', '')}"
            for evt in raw_events
        ]

        return RadioDramaConductorResult(
            config=config,
            events=raw_events,
            transcript=transcript_lines,
            total_llm_calls=self._total_session_calls + self._llm_calls,
            director_notes=[self._director_note] if self._director_note else [],
            drama_events=drama_events,
            drama_scenes=drama_scenes,
            drama_id=self._drama_id,
        )

    # ═══════════════════════════════════════════════════════════════════
    # CAST DESCRIPTION BUILDER
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_cast_descriptions(config: RadioDramaConfig) -> str:
        """Build the cast description block for the system prompt."""
        lines: list[str] = []
        if config.announcer and config.announcer.name:
            lines.append(
                f"  {config.announcer.name} (announcer): "
                f"{config.announcer.speech_style or 'formal radio announcer'}"
            )
        for c in config.characters:
            detail = c.speech_style or c.description or "distinct voice"
            lines.append(f"  {c.name} ({c.role or 'character'}): {detail}")
        return "\n".join(lines) if lines else "(no cast)"

    # ═══════════════════════════════════════════════════════════════════
    # RESPONSE PARSING
    # ═══════════════════════════════════════════════════════════════════

    def _parse_scene_response(
        self,
        raw: str,
        *,
        scene_id: str,
        global_order_start: int,
    ) -> list[dict]:
        """Parse the LLM's JSON array response into a list of event dicts.

        Expected: a JSON array of line objects with speaker_name, text, tone,
        etc. Falls back to a single event (attributed to the announcer) when
        parsing fails entirely.
        """
        cleaned = raw.strip()

        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        fallback_speaker = ""
        config = self.config
        if isinstance(config, RadioDramaConfig) and config.announcer:
            fallback_speaker = config.announcer.name

        try:
            data = json.loads(cleaned)
            if not isinstance(data, list):
                raise ValueError("Expected a JSON array")
        except (json.JSONDecodeError, ValueError):
            return [{
                "speaker_name": fallback_speaker,
                "text": raw.strip()[:1000],
                "tone": "neutral",
                "volume": "moderate",
                "pace": "medium",
                "role": "character",
                "event_kind": "utterance",
                "scene_id": scene_id,
                "global_order": global_order_start,
                "local_order": 0,
            }]

        events = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            speaker_name = item.get("speaker_name", fallback_speaker)
            item = self._capture_issue_flag(
                item,
                from_agent=(
                    f"character:{speaker_name}"
                    if speaker_name
                    else "radio_drama_cast"
                ),
            )

            events.append({
                "speaker_name": speaker_name,
                "text": item.get("text", ""),
                "tone": item.get("tone", "neutral"),
                "volume": item.get("volume", "moderate"),
                "pace": item.get("pace", "medium"),
                "pause_before": item.get("pause_before", "none"),
                "pause_after": item.get("pause_after", "none"),
                "interruption": item.get("interruption", False),
                "nonverbal": item.get("nonverbal", []),
                "role": item.get("role", "character"),
                "event_kind": item.get("event_kind", "utterance"),
                "scene_id": scene_id,
                "global_order": global_order_start + i,
                "local_order": i,
            })

        if not events:
            events.append({
                "speaker_name": fallback_speaker,
                "text": "(scene generation produced no content)",
                "tone": "neutral",
                "volume": "moderate",
                "pace": "medium",
                "role": "character",
                "event_kind": "utterance",
                "scene_id": scene_id,
                "global_order": global_order_start,
                "local_order": 0,
            })

        return events

    # ═══════════════════════════════════════════════════════════════════
    # CONTEXT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _build_prior_context(self) -> str | None:
        """Return concatenated approved scene texts for consistency checking."""
        if not self._approved_scene_texts:
            return None
        return "\n---\n".join(self._approved_scene_texts)

    @staticmethod
    def _summarize_scene(
        events: list[dict],
        scene_config: SceneConfig,
    ) -> str:
        """Create a brief summary of a scene for the context log (no LLM)."""
        speakers = set()
        snippets = []
        for evt in events:
            name = evt.get("speaker_name", "?")
            speakers.add(name)
            text = evt.get("text", "")
            if text:
                snippets.append(f"{name}: {text[:80]}")
        speaker_list = ", ".join(sorted(s for s in speakers if s))
        return (
            f"{scene_config.title} — speakers: {speaker_list}. "
            f"{' | '.join(snippets[:3])}"
        )

    def _build_context_summary(self) -> str:
        """Build the context summary from compressed + recent log entries."""
        parts = []
        if self._compressed_ctx:
            parts.append(f"Earlier: {self._compressed_ctx}")

        recent_count = 3
        recent = self._context_log[-recent_count:] if self._context_log else []
        if recent:
            parts.append(
                "The story so far:\n" + "\n".join(f"  - {s}" for s in recent)
            )

        return "\n\n".join(parts) if parts else ""

    def _compress_context(self) -> None:
        """Compress older scene summaries into a shorter context block."""
        if not self._context_log:
            return

        keep_recent = 2
        to_compress = (
            self._context_log[:-keep_recent]
            if len(self._context_log) > keep_recent
            else []
        )
        if not to_compress:
            return

        transcript_block = "\n".join(to_compress)
        prompt = CONTEXT_SUMMARY_PROMPT.format(transcript=transcript_block)

        raw = self._client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You summarize radio dramas concisely.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        self._llm_calls += 1

        self._compressed_ctx = raw.strip()[:500]
        self._context_log = self._context_log[-keep_recent:]

    # ═══════════════════════════════════════════════════════════════════
    # DIRECTOR CHECK
    # ═══════════════════════════════════════════════════════════════════

    def _director_check(self, scene_number: int, total_scenes: int) -> dict:
        """Ask the Director to review progress and issue a pacing note."""
        config = self.config
        if not isinstance(config, RadioDramaConfig):
            return {}

        director = config.director_brief
        if not director:
            return {}

        if director.mood_progression:
            progress = scene_number / max(total_scenes, 1)
            mood_index = min(
                int(progress * len(director.mood_progression)),
                len(director.mood_progression) - 1,
            )
            current_mood = director.mood_progression[mood_index]
        else:
            current_mood = "neutral"

        recent = self._context_log[-2:] if self._context_log else ["(no content yet)"]
        recent_text = "\n".join(recent)

        prompt = DIRECTOR_CHECK_PROMPT.format(
            scene_number=scene_number,
            total_scenes=total_scenes,
            arc=director.narrative_arc or "(no specific arc)",
            current_mood=current_mood,
            closing_strategy=director.closing_strategy or "(natural ending)",
            recent_content=recent_text,
        )
        if pending_issues := self._consume_pending_issue_reports():
            prompt = f"PENDING ISSUE REPORTS FROM CAST:\n{pending_issues}\n\n{prompt}"

        raw = self._client.chat(
            messages=[
                {
                    "role": "system",
                    "content": self._augmented_director_system_prompt(
                        "You are a radio drama Director. Respond with JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        self._llm_calls += 1

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
        except (json.JSONDecodeError, TypeError):
            return {}

        previous_note = self._director_note
        mood_note = data.get("mood_note", "")
        nudge = data.get("nudge")
        if mood_note:
            self._director_note = mood_note
            if nudge:
                self._director_note += f" | {nudge}"
        elif nudge:
            self._director_note = str(nudge)
        else:
            self._director_note = ""
        if self._director_note != previous_note:
            self._broadcast_director_note(
                self._director_note,
                summary="Director updated dramatic guidance.",
            )

        return data
