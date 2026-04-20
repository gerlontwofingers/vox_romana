"""The News Conductor — runtime engine that generates a news broadcast.

This is the news counterpart to DebateConductor. It takes a fully populated
NewsConfig and generates a complete news broadcast by orchestrating LLM
agents segment-by-segment.

STRUCTURAL DIFFERENCE FROM DEBATES:

    Debate:  One continuous conversation. Agents react to each other.
             Turn order is interleaved (A → B → moderator → A → ...).

    News:    A sequence of SEGMENTS. Each segment is mostly independent.
             The anchor delivers each story, the show host provides
             transitions between segments. There's no argument — just
             authoritative delivery of facts in sequence.

THE SEGMENT LOOP:

    For each segment in the rundown:
    1. (Optional) Show host delivers a transition/intro for this segment
    2. Anchor delivers the segment content (1-3 beats depending on length)
    3. Director checks in periodically for pacing adjustments
    4. Move to next segment

    Bookending the whole show:
    - Show host intro before segment 1
    - Show host outro after the last segment

CONTEXT MANAGEMENT:

    Simpler than debates. Each segment is mostly self-contained — the anchor
    gets the segment's key facts and delivers them. We still maintain a
    running summary for continuity (so the anchor doesn't repeat itself
    and transitions feel natural), but we don't need the complex sliding
    window that debates require.

USAGE:

    # From a config:
    conductor = NewsConductor(config, client=client)
    result = conductor.run()

    # One-shot from brief:
    conductor = NewsConductor.from_brief("Top AI stories this week")
    result = conductor.run()

    # Save:
    merged = result.to_merged_news_show()
    NewsSerializer().save(merged, "output/news.json")
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field

from ...core.client import DeepSeekClient
from ...core.conductor import BaseConductor, ConductorResult
from ...core.session import AgentSession
from .config import NewsConfig, AnchorConfig, SegmentConfig
from .models import MergedNewsShow, NewsEvent, NewsSegment
from .prompts import (
    ANCHOR_SYSTEM_PROMPT,
    CONTEXT_SUMMARY_PROMPT,
    DIRECTOR_CHECK_PROMPT,
    EMCEE_ROLE_INSTRUCTIONS,
    NARRATOR_ROLE_INSTRUCTIONS,
    SEGMENT_DELIVERY_PROMPT,
    SHOW_HOST_SYSTEM_PROMPT,
)


# How many delivered segment transcripts to keep as verbatim context.
DEFAULT_RECENT_WINDOW = 3

# Duration hints map to approximate sentence counts per beat.
# "short" beats are quick hits, "long" beats are deep-dives.
# Multi-beat segments use multiple beats to build longer coverage.
DURATION_SENTENCES = {
    "short": (2, 5),
    "medium": (5, 10),
    "long": (10, 18),
}


# ═══════════════════════════════════════════════════════════════════════════
# NewsConductorResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NewsConductorResult(ConductorResult):
    """Typed result from a NewsConductor run.

    Adds:
        news_events:   List of typed NewsEvent objects.
        news_segments: List of completed NewsSegment objects.
        show_id:       Unique ID for this broadcast.
    """

    news_events: list[NewsEvent] = field(default_factory=list)
    news_segments: list[NewsSegment] = field(default_factory=list)
    show_id: str = ""

    def to_merged_news_show(self) -> MergedNewsShow:
        """Convert this result into a MergedNewsShow for serialization."""
        config = self.config
        if not isinstance(config, NewsConfig):
            raise TypeError(f"Expected a NewsConfig, got {type(config)}")

        # Build voice casting from anchor and host profiles.
        voice_casting: dict[str, str] = {}
        voice_hints: dict[str, str] = {}
        tts_styles: dict[str, str] = {}
        for anchor in config.anchors:
            if anchor.voice_gender:
                voice_casting[anchor.name] = anchor.voice_gender
            if anchor.voice_hint:
                voice_hints[anchor.name] = anchor.voice_hint
            if anchor.tts_style:
                tts_styles[anchor.name] = anchor.tts_style
        if config.show_host:
            host = config.show_host
            if host.voice_gender:
                voice_casting[host.name] = host.voice_gender
            if host.voice_hint:
                voice_hints[host.name] = host.voice_hint
            if host.tts_style:
                tts_styles[host.name] = host.tts_style

        return MergedNewsShow(
            show_id=self.show_id,
            title=config.title or "Untitled News Show",
            segments=tuple(self.news_segments),
            events=tuple(self.news_events),
            anchors=tuple(a.name for a in config.anchors),
            hosts=(
                (config.show_host.name,) if config.show_host else ()
            ),
            guests=tuple(g.name for g in config.guests),
            topic=config.topic,
            tone=config.tone,
            segment_count=len(self.news_segments),
            voice_casting=voice_casting,
            voice_hints=voice_hints,
            tts_styles=tts_styles,
        )


# ═══════════════════════════════════════════════════════════════════════════
# THE NEWS CONDUCTOR
# ═══════════════════════════════════════════════════════════════════════════

class NewsConductor(BaseConductor):
    """Generates a complete news broadcast by orchestrating LLM agents.

    The segment loop walks through the rundown in order, generating anchor
    deliveries and show host transitions for each story.
    """

    def __init__(
        self,
        config: NewsConfig,
        *,
        client: DeepSeekClient | None = None,
        api_key: str | None = None,
        **base_kwargs,
    ) -> None:
        resolved_client = client or DeepSeekClient(api_key=api_key)
        super().__init__(config, client=resolved_client, **base_kwargs)

        # Populated by _build_agents():
        self._anchor_prompts: dict[str, str] = {}
        self._host_prompt: str | None = None
        self._anchor_sessions: dict[str, AgentSession] = {}

        # Runtime state:
        self._transcript: list[dict] = []
        self._summary: str = ""
        self._director_note: str = ""
        self._llm_calls: int = 0
        self._show_id: str = ""

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
    ) -> NewsConductor:
        """Bootstrap a config from a brief and create a conductor.

        Args:
            brief:              Natural language description of the news show.
            reference_material: Factual source text to ground the output.
            fact_check:         "off", "quick", or "thorough".
            overrides:          Dict of NewsConfig field overrides.
            client:             Shared DeepSeekClient.
            api_key:            API key if client is not provided.

        Returns:
            A NewsConductor ready to run().
        """
        # Import here to avoid circular imports.
        from .bootstrap import NewsBootstrap

        shared_client = client or DeepSeekClient(api_key=api_key)
        bootstrap = NewsBootstrap(client=shared_client)
        config = bootstrap.from_brief(
            brief,
            reference_material=reference_material,
            fact_check=fact_check,
            overrides=overrides,
        )
        return cls(config, client=shared_client)

    # ═══════════════════════════════════════════════════════════════════
    # BaseConductor interface
    # ═══════════════════════════════════════════════════════════════════

    def _build_agents(self) -> None:
        """Build system prompts for anchor(s) and show host."""
        config = self.config
        if not isinstance(config, NewsConfig):
            raise TypeError(f"NewsConductor requires NewsConfig, got {type(config)}")

        # Build prompt for each anchor.
        for anchor in config.anchors:
            self._anchor_prompts[anchor.name] = self._build_anchor_prompt(anchor)

        # Build show host prompt (if configured).
        if config.show_host:
            self._host_prompt = self._build_host_prompt()

        # Create AgentSessions for anchors — each maintains its own
        # sliding-window memory of past segment deliveries.
        self._anchor_sessions = {}
        for name, prompt in self._anchor_prompts.items():
            self._anchor_sessions[name] = self._create_session(
                prompt, recent_window=DEFAULT_RECENT_WINDOW,
            )

    def _run_turn_loop(self) -> list[dict]:
        """Run the segment-by-segment generation loop.

        Returns:
            A list of raw event dicts (all segments + host narrations).
        """
        config = self.config
        if not isinstance(config, NewsConfig):
            raise TypeError(f"NewsConductor requires NewsConfig, got {type(config)}")

        self._show_id = str(uuid.uuid4())[:8]
        all_events: list[dict] = []
        global_order = 0
        segments = config.segments
        total_segments = len(segments)

        # ── Auto-scale beats to hit target_words ─────────────────────
        # If the user requested a word count, distribute beats across
        # segments to approximately reach it. Each beat produces ~100-200
        # words, so we estimate total beats needed and spread them.
        if config.target_words and total_segments > 0:
            # Estimate: ~150 words per beat (mid-range for medium duration).
            words_per_beat = 150
            total_beats_needed = max(total_segments, config.target_words // words_per_beat)
            # Distribute extra beats proportionally by duration hint weight.
            weight_map = {"short": 1, "medium": 2, "long": 3}
            total_weight = sum(weight_map.get(s.duration_hint, 2) for s in segments)
            for s in segments:
                if s.target_beats <= 1:  # Only auto-scale if not already set
                    w = weight_map.get(s.duration_hint, 2)
                    s.target_beats = max(1, round(total_beats_needed * w / total_weight))

        # ── Show host intro ──────────────────────────────────────────
        if self._host_prompt:
            headlines_preview = "\n".join(
                f"  - {s.headline}" for s in segments[:5]
            )
            intro_event = self._generate_host_event(
                beat="intro",
                context_hint=(
                    f"Open the broadcast. Tonight's stories:\n{headlines_preview}\n\n"
                    "Hook the listener and set the tone for the show."
                ),
            )
            intro_event["global_order"] = global_order
            global_order += 1
            all_events.append(intro_event)
            self._transcript.append(intro_event)

        # ── Segment loop ─────────────────────────────────────────────
        for seg_idx, segment in enumerate(segments):
            seg_num = seg_idx + 1

            # Transition from show host (except before first segment — that's the intro).
            if self._host_prompt and seg_idx > 0:
                prev_headline = segments[seg_idx - 1].headline
                transition_event = self._generate_host_event(
                    beat="transition",
                    context_hint=(
                        f"We just covered: \"{prev_headline}\"\n"
                        f"Next up: \"{segment.headline}\"\n\n"
                        "Bridge smoothly between these stories."
                    ),
                )
                transition_event["global_order"] = global_order
                global_order += 1
                all_events.append(transition_event)
                self._transcript.append(transition_event)

            # Pick anchor for this segment (round-robin if multiple).
            anchor = config.anchors[seg_idx % len(config.anchors)]

            # Generate segment delivery.
            seg_events = self._generate_segment(
                segment=segment,
                anchor=anchor,
                segment_number=seg_num,
                total_segments=total_segments,
                global_order_start=global_order,
            )

            for evt in seg_events:
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
                and seg_num > 0
                and seg_num % director_brief.check_interval == 0
            ):
                self._director_check(seg_num, total_segments)

        # ── Show host outro ──────────────────────────────────────────
        if self._host_prompt:
            delivered_headlines = "\n".join(
                f"  - {s.headline}" for s in segments
            )
            outro_event = self._generate_host_event(
                beat="outro",
                context_hint=(
                    f"The broadcast is ending. Stories we covered:\n{delivered_headlines}\n\n"
                    "Recap the headlines briefly and sign off the show."
                ),
            )
            outro_event["global_order"] = global_order
            global_order += 1
            all_events.append(outro_event)
            self._transcript.append(outro_event)

        return all_events

    def _package_result(self, raw_events: list[dict]) -> NewsConductorResult:
        """Convert raw events into a typed NewsConductorResult."""
        config = self.config
        if not isinstance(config, NewsConfig):
            raise TypeError(f"NewsConductor requires NewsConfig, got {type(config)}")

        # Build typed NewsEvent objects.
        news_events = []
        for i, evt in enumerate(raw_events):
            tone_raw = evt.get("tone", "neutral")
            if isinstance(tone_raw, str):
                tone_raw = (tone_raw,)
            elif isinstance(tone_raw, list):
                tone_raw = tuple(tone_raw)

            nonverbal_raw = evt.get("nonverbal", ())
            if isinstance(nonverbal_raw, list):
                nonverbal_raw = tuple(nonverbal_raw)

            news_events.append(NewsEvent(
                global_order=evt.get("global_order", i),
                local_order=evt.get("local_order", i),
                speaker_role=evt.get("role", "anchor"),
                speaker_name=evt.get("speaker", ""),
                target=evt.get("target", "audience"),
                text=evt.get("text", ""),
                tone=tone_raw,
                volume=evt.get("volume"),
                pace=evt.get("pace"),
                pause_before=evt.get("pause_before"),
                pause_after=evt.get("pause_after"),
                interruption=evt.get("interruption", False),
                nonverbal=nonverbal_raw,
                event_kind=evt.get("event_kind", "utterance"),
                segment_id=evt.get("segment_id"),
            ))

        # Group events into segments.
        segment_map: dict[str, list[NewsEvent]] = {}
        for evt in news_events:
            if evt.segment_id:
                segment_map.setdefault(evt.segment_id, []).append(evt)

        news_segments = []
        for seg_cfg in config.segments:
            seg_id = f"{self._show_id}_seg_{config.segments.index(seg_cfg)}"
            evts = segment_map.get(seg_id, [])
            anchor_name = ""
            if evts:
                anchor_name = evts[0].speaker_name
            news_segments.append(NewsSegment(
                segment_id=seg_id,
                segment_order=config.segments.index(seg_cfg),
                headline=seg_cfg.headline,
                segment_type=seg_cfg.segment_type,
                anchor_name=anchor_name,
                tone=seg_cfg.tone,
                events=tuple(evts),
            ))

        transcript_lines = [
            f"[{evt.get('speaker', '?')}]: {evt.get('text', '')}"
            for evt in raw_events
        ]

        return NewsConductorResult(
            config=config,
            events=raw_events,
            transcript=transcript_lines,
            total_llm_calls=self._total_session_calls + self._llm_calls,
            director_notes=[self._director_note] if self._director_note else [],
            news_events=news_events,
            news_segments=news_segments,
            show_id=self._show_id,
        )

    # ═══════════════════════════════════════════════════════════════════
    # PROMPT BUILDERS
    # ═══════════════════════════════════════════════════════════════════

    def _build_anchor_prompt(self, anchor: AnchorConfig) -> str:
        """Build the system prompt for a news anchor."""
        config = self.config
        if not isinstance(config, NewsConfig):
            raise TypeError(f"Expected NewsConfig, got {type(config)}")

        return ANCHOR_SYSTEM_PROMPT.format(
            name=anchor.name,
            topic=config.topic or config.brief,
            style=anchor.style or "professional and authoritative",
            voice_fingerprint=anchor.voice_fingerprint or "(no specific voice style)",
            setting=config.setting or "news studio",
        )

    def _build_host_prompt(self) -> str:
        """Build the system prompt for the show host."""
        config = self.config
        if not isinstance(config, NewsConfig):
            raise TypeError(f"Expected NewsConfig, got {type(config)}")

        host = config.show_host
        if not host:
            return ""

        anchor_list = ", ".join(a.name for a in config.anchors) or "TBD"

        if host.role == "emcee":
            role_instructions = EMCEE_ROLE_INSTRUCTIONS
        else:
            role_instructions = NARRATOR_ROLE_INSTRUCTIONS

        return SHOW_HOST_SYSTEM_PROMPT.format(
            name=host.name,
            role=host.role,
            style=host.style or "polished and engaging",
            voice_fingerprint=host.voice_fingerprint or "(no specific voice style)",
            topic=config.topic or config.brief,
            setting=config.setting or "news studio",
            anchor_list=anchor_list,
            role_upper=host.role.upper(),
            role_instructions=role_instructions,
        )

    # ═══════════════════════════════════════════════════════════════════
    # SEGMENT GENERATION
    # ═══════════════════════════════════════════════════════════════════

    def _generate_segment(
        self,
        *,
        segment: SegmentConfig,
        anchor: AnchorConfig,
        segment_number: int,
        total_segments: int,
        global_order_start: int,
    ) -> list[dict]:
        """Generate audio events for one news segment.

        The anchor delivers the story in one or more beats:
          - 1 beat (default):  Single delivery — lead + body in one shot.
          - 2 beats:           Lead (hook + setup) then body (detail + context).
          - 3+ beats:          Lead → body sections → tag/wrap.

        The number of beats is controlled by segment.target_beats (set by
        the architect based on the segment's depth and the requested length).
        """
        config = self.config
        if not isinstance(config, NewsConfig):
            raise TypeError(f"Expected NewsConfig, got {type(config)}")

        seg_id = f"{self._show_id}_seg_{segment_number - 1}"
        system_prompt = self._anchor_prompts.get(anchor.name, "")

        # Build key facts block.
        if segment.key_facts:
            key_facts_block = "\n".join(f"  - {f}" for f in segment.key_facts)
        else:
            key_facts_block = "  (no specific facts provided — deliver based on the headline and angle)"

        # Build reference section.
        reference_section = ""
        if config.reference_material:
            reference_section = (
                f"REFERENCE MATERIAL (cite only from this):\n{config.reference_material}\n"
            )

        num_beats = max(1, segment.target_beats)
        events: list[dict] = []

        # Define beat labels for multi-beat segments.
        if num_beats == 1:
            beat_labels = ["full"]
        elif num_beats == 2:
            beat_labels = ["lead", "body"]
        else:
            beat_labels = ["lead"] + [f"body_{i}" for i in range(1, num_beats - 1)] + ["tag"]

        for beat_idx, beat_label in enumerate(beat_labels):
            # Duration per beat: distribute evenly across beats.
            min_s, max_s = DURATION_SENTENCES.get(segment.duration_hint, (5, 10))
            duration_instruction = f"Aim for {min_s}-{max_s} sentences."

            # Add word budget hint when target_words is set.
            remaining_segments = total_segments - segment_number + 1
            remaining_beats = remaining_segments * num_beats - beat_idx
            words_per_beat = 0
            if config.target_words and num_beats > 0:
                words_per_segment = config.target_words // total_segments
                words_per_beat = words_per_segment // num_beats
                words_per_beat = self._rebalanced_target(words_per_beat, remaining_beats)
                duration_instruction += "\n" + self._length_contract_block(
                    words_per_beat, tolerance_pct=12.0,
                )

            # Append runtime budget status for the LLM.
            budget_status = self._budget_status_for_prompt(
                remaining_beats, total_steps=total_segments * num_beats,
            )
            if budget_status:
                duration_instruction += f"\n{budget_status}"

            # Build beat-specific instruction.
            if beat_label == "full":
                beat_instruction = (
                    f"Deliver this news segment. Cover the key facts naturally, "
                    f"following the angle provided. Match the tone to the story content. "
                    f"{duration_instruction}"
                )
            elif beat_label == "lead":
                beat_instruction = (
                    f"Deliver the LEAD for this story — hook the listener, establish "
                    f"the headline, and set up the key question or conflict. "
                    f"{duration_instruction}"
                )
            elif beat_label == "tag":
                beat_instruction = (
                    f"Deliver the TAG for this story — wrap up the segment with "
                    f"context, implications, or a forward-looking note. "
                    f"{duration_instruction}"
                )
            else:
                beat_instruction = (
                    f"Continue the BODY of this story — go deeper into the facts, "
                    f"provide context, examples, or expert perspectives. "
                    f"Do NOT repeat what you already covered in earlier beats. "
                    f"{duration_instruction}"
                )

            user_message = SEGMENT_DELIVERY_PROMPT.format(
                segment_number=segment_number,
                total_segments=total_segments,
                headline=segment.headline,
                angle=segment.angle or "straight news delivery",
                segment_tone=segment.tone or config.tone or "authoritative",
                duration_hint=segment.duration_hint,
                key_facts=key_facts_block,
                reference_section=reference_section,
                duration_instruction=beat_instruction,
            )

            # Add director note if present.
            if self._director_note:
                user_message = f"DIRECTOR'S NOTE:\n{self._director_note}\n\n{user_message}"

            # Send via the anchor's AgentSession.
            session = self._anchor_sessions.get(anchor.name)
            if session is not None:
                raw_response = session.send(user_message)
            else:
                raw_response = self._client.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
                self._llm_calls += 1

            event_dict = self._parse_response(raw_response)
            event_dict["speaker"] = anchor.name
            event_dict["role"] = "anchor"
            event_dict["event_kind"] = "utterance"
            event_dict["segment_id"] = seg_id
            event_dict["global_order"] = global_order_start + beat_idx
            event_dict["local_order"] = beat_idx
            event_dict["beat_label"] = beat_label

            # ── Length contract validation (Phase 1+2) ──────────────
            if words_per_beat > 0:
                actual = len((event_dict.get("text") or "").split())
                warning = self._validate_word_count(
                    declared=event_dict.get("word_count"),
                    actual=actual,
                    target_words=words_per_beat,
                    tolerance_pct=12.0,
                )
                if warning:
                    self._length_warnings.append(
                        f"[seg {segment_number}/{total_segments} "
                        f"beat {beat_label}] {warning}"
                    )

            # ── Checkpoint render: measure THIS beat's audio length ──
            # Beats are individual TTS units, so each one has a clear
            # global index in the budget tracker.
            beat_global_index = global_order_start + beat_idx
            self._checkpoint_segment(
                beat_global_index, event_dict.get("text") or "",
            )

            # Pauses: short between beats, medium around the full segment.
            if beat_idx == 0:
                event_dict.setdefault("pause_before", "short")
            else:
                event_dict.setdefault("pause_before", "none")

            if beat_idx == num_beats - 1:
                event_dict.setdefault("pause_after", "medium")
            else:
                event_dict.setdefault("pause_after", "short")

            events.append(event_dict)

        return events

    def _generate_host_event(self, *, beat: str, context_hint: str) -> dict:
        """Generate one narration event from the show host."""
        config = self.config
        if not isinstance(config, NewsConfig):
            raise TypeError(f"Expected NewsConfig, got {type(config)}")

        host = config.show_host
        if not host:
            return {}

        user_message = f"NARRATION BEAT: {beat.upper()}\n\n{context_hint}"

        raw_response = self._client.chat(
            messages=[
                {"role": "system", "content": self._host_prompt},
                {"role": "user", "content": user_message},
            ]
        )
        self._llm_calls += 1

        event_dict = self._parse_response(raw_response)
        event_dict["speaker"] = host.name
        event_dict["role"] = "host"
        event_dict["event_kind"] = "narration"
        event_dict["narration_beat"] = beat
        event_dict["target"] = "audience"

        if beat == "intro":
            event_dict.setdefault("pause_after", "long")
        elif beat == "transition":
            event_dict.setdefault("pause_before", "medium")
            event_dict.setdefault("pause_after", "medium")
        elif beat == "outro":
            event_dict.setdefault("pause_before", "long")

        return event_dict

    # ═══════════════════════════════════════════════════════════════════
    # RESPONSE PARSING
    # ═══════════════════════════════════════════════════════════════════

    def _parse_response(self, raw: str) -> dict:
        """Parse an agent's JSON response into an event dict.

        Handles markdown fences, preamble text, and graceful fallback.
        """
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
            return data
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
    # CONTEXT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _compress_context(self) -> None:
        """Compress older transcript entries into a summary."""
        older = self._transcript[:-DEFAULT_RECENT_WINDOW]
        if len(older) < 2:
            return

        transcript_text = "\n".join(
            f"[{t.get('speaker', '?')}]: {t.get('text', '')}" for t in older
        )

        prompt = CONTEXT_SUMMARY_PROMPT.format(transcript=transcript_text)

        self._summary = self._client.chat(
            messages=[
                {"role": "system", "content": "You are a concise summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        self._llm_calls += 1

    def _director_check(self, segment_number: int, total_segments: int) -> dict:
        """Ask the Director to review the broadcast and issue a pacing note."""
        config = self.config
        if not isinstance(config, NewsConfig):
            return {}

        director = config.director_brief
        if not director:
            return {}

        # Current mood target.
        if director.mood_progression:
            progress = segment_number / max(total_segments, 1)
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
            segment_number=segment_number,
            total_segments=total_segments,
            show_arc=director.show_arc or "(no specific arc)",
            current_mood=current_mood,
            closing_strategy=director.closing_strategy or "(natural ending)",
            recent_content=recent_text or "(broadcast just started)",
        )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": "You are a broadcast director. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        self._llm_calls += 1

        result = self._parse_response(raw)
        mood = result.get("mood_note", "")
        nudge = result.get("nudge", "")
        self._director_note = f"{mood} {nudge}".strip() if (mood or nudge) else ""

        return result
