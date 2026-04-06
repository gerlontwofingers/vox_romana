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

# Duration hints map to approximate sentence counts for instruction.
DURATION_SENTENCES = {
    "short": (2, 4),
    "medium": (4, 7),
    "long": (7, 12),
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
    ) -> None:
        super().__init__(config)
        self._client = client or DeepSeekClient(api_key=api_key)

        # Populated by _build_agents():
        self._anchor_prompts: dict[str, str] = {}
        self._host_prompt: str | None = None

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
            total_llm_calls=self._llm_calls,
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

        The anchor gets the segment config (headline, facts, angle) and
        delivers the story. For longer segments, we may generate multiple
        beats (lead, body, tag).
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

        # Duration instruction.
        min_s, max_s = DURATION_SENTENCES.get(segment.duration_hint, (4, 7))
        duration_instruction = f"Aim for {min_s}-{max_s} sentences."

        # Build the user message.
        user_message = SEGMENT_DELIVERY_PROMPT.format(
            segment_number=segment_number,
            total_segments=total_segments,
            headline=segment.headline,
            angle=segment.angle or "straight news delivery",
            segment_tone=segment.tone or config.tone or "authoritative",
            duration_hint=segment.duration_hint,
            key_facts=key_facts_block,
            reference_section=reference_section,
            duration_instruction=duration_instruction,
        )

        # Add context summary if we have one.
        if self._summary:
            user_message = f"BROADCAST SO FAR (summary):\n{self._summary}\n\n{user_message}"

        # Add director note if present.
        if self._director_note:
            user_message = f"DIRECTOR'S NOTE:\n{self._director_note}\n\n{user_message}"

        # Call the LLM.
        raw_response = self._client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        )
        self._llm_calls += 1

        # Parse the response.
        event_dict = self._parse_response(raw_response)
        event_dict["speaker"] = anchor.name
        event_dict["role"] = "anchor"
        event_dict["event_kind"] = "utterance"
        event_dict["segment_id"] = seg_id
        event_dict["global_order"] = global_order_start
        event_dict["local_order"] = 0

        # Set segment-appropriate pauses.
        event_dict.setdefault("pause_before", "short")
        event_dict.setdefault("pause_after", "medium")

        return [event_dict]

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
            ]
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
            ]
        )
        self._llm_calls += 1

        result = self._parse_response(raw)
        mood = result.get("mood_note", "")
        nudge = result.get("nudge", "")
        self._director_note = f"{mood} {nudge}".strip() if (mood or nudge) else ""

        return result
