"""News show bootstrap — brief → fully populated NewsConfig.

The news bootstrap is simpler than the debate bootstrap because there's no
adversarial casting. Instead of participants with opposing sides, we have:
    1. An ARCHITECT call that plans the rundown (segments, anchor, setting)
    2. An optional DIRECTOR call that plans the broadcast arc

The anchor doesn't need the full "Profile Builder" treatment because they
don't argue — they deliver. We still give them a voice_fingerprint via a
lightweight LLM call.

CHAIN:
    brief → [Researcher] → Architect → [Anchor profiler] → Director → NewsConfig

EXAMPLE:
    bootstrap = NewsBootstrap()
    config = bootstrap.from_brief("Top 5 AI stories this week")
    # config has segments, anchor, show host, director brief — ready for NewsConductor
"""

from __future__ import annotations

import json
from typing import Any

from ...core.bootstrap import BaseShowBootstrap
from ...core.client import DeepSeekClient
from ...core.config import ParticipantProfile, ShowHostConfig
from ...core.search import BaseSearchProvider, build_research_provider
from .config import AnchorConfig, NewsConfig, NewsDirectorBrief, SegmentConfig


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECT SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

NEWS_ARCHITECT_PROMPT = """\
You are the Architect for a news show generation system. Your job is to take
a brief and produce a structured news show configuration: what stories to cover,
who anchors, and the overall broadcast plan.

You will receive:
  1. A brief — the user's natural language description of the news show they want.
  2. Optionally, reference material — factual source text about the topics.

You must produce a JSON object with EXACTLY this schema. Do NOT include any
text outside the JSON block.

{
  "title": "string — show title (e.g., 'The Evening Brief', 'Tech Roundup')",
  "topic": "string — the broadcast theme (e.g., 'This week in AI')",
  "tone": "string — overall broadcast tone (e.g., 'authoritative', 'conversational')",
  "setting": "string — where the broadcast happens (e.g., 'modern news studio')",
  "target_segments": "integer — how many segments/stories to cover (scale to match brief)",
  "anchors": [
    {
      "name": "string — anchor's full name",
      "style": "string — delivery style description"
    }
  ],
  "segments": [
    {
      "headline": "string — story headline (short, punchy)",
      "angle": "string — editorial angle or focus",
      "key_facts": ["array of verified fact strings for this story"],
      "tone": "string — segment-specific tone (e.g., 'somber', 'upbeat')",
      "duration_hint": "string — 'short', 'medium', or 'long'",
      "segment_type": "string — 'anchor_read' (default), 'package', 'breaking'",
      "target_beats": "integer — how many beats (1=simple read, 2=lead+body, 3+=deep dive). Default 1."
    }
  ],
  "show_host": {
    "name": "string — host name",
    "role": "string — 'narrator' or 'emcee'",
    "style": "string — hosting style"
  }
}

RULES:
- Usually 1-2 anchors. Solo anchor is most common.
- Order segments by editorial priority: lead story first, kicker last.
- Scale segment count to match the brief: a quick roundup might be 3-5
  segments, a deep-dive show could be 10-20+, a full-hour broadcast
  might be 15-30. There is NO hard cap — use as many segments as the
  content warrants. If the user specifies a word count or duration,
  plan enough segments to fill it.
- The lead story should be "medium" or "long". Kicker should be "short".
- show_host wraps the broadcast in narrative structure. ALWAYS include one.
- If given reference material, extract key_facts from it for each segment.
- If NOT given reference material, use plausible key_facts that are
  clearly labeled as illustrative (do NOT fabricate specific statistics).
- ALWAYS respond with valid JSON. Nothing else.
"""

# ═══════════════════════════════════════════════════════════════════════════
# DIRECTOR SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

NEWS_DIRECTOR_PROMPT = """\
You are the Director for a news broadcast. Your job is to plan the pacing
and dramatic arc of the broadcast — how energy flows from segment to segment.

You will receive details about the broadcast: topic, segments, tone, and length.

You must produce a JSON object with EXACTLY this schema:

{
  "show_arc": "string — narrative structure of the broadcast. Example: \
'Open with the lead (heaviest/most urgent story), middle segments alternate \
between hard news and lighter stories, close with an uplifting kicker.'",
  "mood_progression": ["array of strings — ordered moods across the broadcast. \
4-6 entries. Example: ['urgent', 'analytical', 'concerned', 'hopeful', 'warm']"],
  "segment_order_notes": "string — notes on why stories are ordered this way",
  "closing_strategy": "string — how to close the broadcast. Example: 'Anchor \
delivers a brief kicker story, host wraps with a recap of the top headlines \
and a warm sign-off.'"
}

RULES:
- A broadcast should have energy variation — don't keep the same intensity throughout.
- Lead story is always the most important/urgent.
- End on a lighter note when possible (the "kicker" tradition).
- mood_progression should have 4-6 entries mapping to the broadcast timeline.
- ALWAYS respond with valid JSON. Nothing else.
"""

# ═══════════════════════════════════════════════════════════════════════════
# ANCHOR PROFILER PROMPT
# ═══════════════════════════════════════════════════════════════════════════

ANCHOR_PROFILER_PROMPT = """\
You are creating a voice profile for a news anchor or show host. Given their
name and style, produce a voice profile — how they SOUND and how they TALK.

Anchor name: {name}
Anchor style: {style}
Broadcast topic: {topic}
Broadcast tone: {tone}

Respond with ONLY a JSON object:
{{
  "voice_gender": "string — 'male' or 'female'. The vocal gender of this \
person for TTS voice assignment. Base this on their name and role.",
  "voice_hint": "string — a short TTS casting note describing how this person \
SOUNDS. Accent, age range, energy, tone texture. Examples: 'warm authoritative \
baritone, 50s, network news cadence', 'clear alto, 30s, steady and precise', \
'bright tenor, 40s, BBC polish'.",
  "voice_fingerprint": "string — 2-3 sentences describing their vocal delivery, \
cadence, verbal habits, and on-air personality. Be specific and distinctive."
}}
"""


class NewsBootstrap(BaseShowBootstrap):
    """Orchestrates the brief-to-config bootstrap chain for news shows.

    Chain: brief → [Researcher] → Architect → Anchor profiler → Director → config
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
        memory: Any | None = None,
        target_words: int | None = None,
    ) -> NewsConfig:
        """Run the full bootstrap chain: brief → fully populated NewsConfig.

        Args:
            brief:              Natural language description of the news show.
            reference_material: Factual source text to ground the output.
            fact_check:         "off", "quick", or "thorough".
            overrides:          Dict of NewsConfig field names to override.
            auto_research:      If True, research the topic before generating.
            search_provider:    Search provider for web research. If None and
                                auto_research=True, falls back to LLM-only research.
            memory:             Optional MemoryRecall instance. When provided,
                                auto-research consults the local knowledge base
                                FIRST and only falls through to ``search_provider``
                                if memory doesn't have enough material.

        Returns:
            A fully populated NewsConfig.
        """
        overrides = overrides or {}

        # ── Stage 0: Auto-research (optional) ────────────────────────
        if auto_research and not reference_material:
            # Reuse the debate researcher — it works for any topic.
            from ..debates.researcher import DebateResearcher, NEWS_QUERY_FORMULATOR_PROMPT
            effective_provider = build_research_provider(search_provider, memory)
            if effective_provider is None:
                # No search provider AND no memory → LLM-only research
                # (DeepSeek compiles from training knowledge).
                researcher = DebateResearcher(self._client, query_formulator_prompt=NEWS_QUERY_FORMULATOR_PROMPT)
            else:
                researcher = DebateResearcher(self._client, effective_provider, query_formulator_prompt=NEWS_QUERY_FORMULATOR_PROMPT)
            reference_material = researcher.run(brief) or None

        # ── Stage 1: Architect ──────────────────────────────────────
        config = self._run_architect(brief, reference_material, fact_check, target_words=target_words)

        # ── Stage 2: Anchor profiling ────────────────────────────────
        for i, anchor in enumerate(config.anchors):
            config.anchors[i] = self._profile_anchor(
                anchor,
                topic=config.topic or brief,
                tone=config.tone or "authoritative",
            )

        # ── Stage 2.5: Show host profiling ───────────────────────────
        if config.show_host and config.show_host.name:
            self._profile_host(
                config.show_host,
                topic=config.topic or brief,
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
    ) -> NewsConfig:
        """Call the Architect LLM to produce a skeleton NewsConfig."""
        user_parts = [f"BRIEF:\n{brief}"]
        if reference_material:
            user_parts.append(f"\nREFERENCE MATERIAL:\n{reference_material}")
        if target_words:
            user_parts.append(
                f"\nTARGET LENGTH: {target_words:,} words. "
                f"Plan enough segments and beats to fill this — a typical "
                f"segment produces ~100-200 words per beat. For a {target_words:,}-word "
                f"show, plan roughly {max(3, target_words // 150)} segments with "
                f"multiple beats for the longer ones."
            )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": NEWS_ARCHITECT_PROMPT},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            temperature=0,
        )

        return self._parse_architect_response(
            raw, brief=brief, reference_material=reference_material, fact_check=fact_check,
        )

    def _parse_architect_response(
        self,
        raw_json: str,
        *,
        brief: str = "",
        reference_material: str | None = None,
        fact_check: str = "off",
    ) -> NewsConfig:
        """Parse the Architect's JSON into a NewsConfig."""
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"News Architect produced invalid JSON. First 200 chars: "
                f"{raw_json[:200]!r}\nParse error: {e}"
            ) from e

        # Build anchors.
        anchors = []
        for a in data.get("anchors", []):
            anchors.append(AnchorConfig(
                name=a.get("name", ""),
                style=a.get("style"),
            ))

        # Build segments.
        segments = []
        for s in data.get("segments", []):
            segments.append(SegmentConfig(
                headline=s.get("headline", ""),
                angle=s.get("angle"),
                key_facts=s.get("key_facts", []),
                tone=s.get("tone"),
                duration_hint=s.get("duration_hint", "medium"),
                segment_type=s.get("segment_type", "anchor_read"),
                target_beats=s.get("target_beats", 1),
            ))

        # Build show host.
        host_data = data.get("show_host")
        show_host = None
        if host_data:
            show_host = ShowHostConfig(
                name=host_data.get("name", ""),
                role=host_data.get("role", "narrator"),
                style=host_data.get("style"),
            )

        return NewsConfig(
            brief=brief,
            reference_material=reference_material,
            fact_check=fact_check,
            title=data.get("title"),
            topic=data.get("topic"),
            tone=data.get("tone"),
            setting=data.get("setting"),
            target_segments=data.get("target_segments"),
            anchors=anchors,
            segments=segments,
            show_host=show_host,
        )

    def _profile_anchor(
        self,
        anchor: AnchorConfig,
        *,
        topic: str,
        tone: str,
    ) -> AnchorConfig:
        """Give an anchor a voice_fingerprint via LLM."""
        prompt = ANCHOR_PROFILER_PROMPT.format(
            name=anchor.name,
            style=anchor.style or "professional",
            topic=topic,
            tone=tone,
        )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": "You produce JSON voice profiles."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(cleaned)
            anchor.voice_fingerprint = data.get("voice_fingerprint", "")
            if data.get("voice_hint"):
                anchor.voice_hint = data["voice_hint"]
            if data.get("voice_gender"):
                anchor.voice_gender = data["voice_gender"]
        except (json.JSONDecodeError, TypeError):
            anchor.voice_fingerprint = "(default authoritative news delivery)"

        return anchor

    def _profile_host(self, host: ShowHostConfig, *, topic: str) -> None:
        """Enrich the show host with voice_fingerprint, voice_hint, and voice_gender via LLM."""
        prompt = ANCHOR_PROFILER_PROMPT.format(
            name=host.name,
            style=host.style or "polished and engaging",
            topic=topic,
            tone="produced broadcast",
        )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": "You produce JSON voice profiles."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(cleaned)
            host.voice_fingerprint = data.get("voice_fingerprint", "")
            if data.get("voice_hint"):
                host.voice_hint = data["voice_hint"]
            if data.get("voice_gender"):
                host.voice_gender = data["voice_gender"]
        except (json.JSONDecodeError, TypeError):
            host.voice_fingerprint = "(warm, polished broadcast host delivery)"

    def _run_director(self, config: NewsConfig) -> NewsDirectorBrief:
        """Call the Director LLM to plan the broadcast arc."""
        segment_list = "\n".join(
            f"  {i+1}. {s.headline} ({s.duration_hint}, {s.tone or config.tone or 'neutral'})"
            for i, s in enumerate(config.segments)
        )

        user_message = (
            f"TOPIC: {config.topic or config.brief}\n"
            f"TONE: {config.tone or 'authoritative'}\n"
            f"TOTAL SEGMENTS: {len(config.segments)}\n"
            f"SEGMENTS:\n{segment_list}"
        )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": NEWS_DIRECTOR_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )

        return self._parse_director_response(raw)

    def _parse_director_response(self, raw_json: str) -> NewsDirectorBrief:
        """Parse the Director's JSON into a NewsDirectorBrief."""
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"News Director produced invalid JSON. First 200 chars: "
                f"{raw_json[:200]!r}\nParse error: {e}"
            ) from e

        return NewsDirectorBrief(
            show_arc=data.get("show_arc", ""),
            mood_progression=data.get("mood_progression", []),
            segment_order_notes=data.get("segment_order_notes"),
            closing_strategy=data.get("closing_strategy"),
        )
