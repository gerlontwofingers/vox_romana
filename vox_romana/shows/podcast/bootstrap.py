"""The Bootstrap orchestrator — chains Architect → Host Profiler → Director.

This is the conductor of the bootstrap assembly line for podcasts. It
coordinates three agents in sequence to turn a brief into a fully populated
PodcastConfig:

    1. ARCHITECT: brief → skeleton config (cast, segments, show_name)
    2. HOST PROFILER: stub host → full personality (one per host)
    3. DIRECTOR: config → episode arc (mood, pacing, per-segment notes)

After all three stages, any user-provided overrides are applied on top.
The result is a PodcastConfig ready for the PodcastConductor.

GENDER ENFORCEMENT:
    The Architect prompt makes gender a REQUIRED field in its JSON schema.
    This is the architectural fix for the "Trash Pandas" bug — when a brief
    says "3 women," all 3 MUST be female. The Architect schema enforces
    this, and PodcastConfig.validate() double-checks it.

OVERRIDE SYSTEM:
    Overrides let you auto-generate most of the config but tweak parts:

        config = bootstrap.from_brief(
            "Three women discuss soccer rivalries",
            overrides={
                "tone": "chaotic energy",
                "show_name": "Trash Pandas",
            },
        )

    Override keys must match PodcastConfig field names. Unknown keys are
    silently ignored.

EXAMPLE — full auto:
    from vox_romana.shows.podcast.bootstrap import PodcastBootstrap

    bootstrap = PodcastBootstrap()     # uses DEEPSEEK_API_KEY from env
    config = bootstrap.from_brief(
        "Three women discuss the greatest rivalries in women's soccer",
        target_words=5000,
    )
    # config is now fully populated and ready for PodcastConductor(config)

EXAMPLE — auto with overrides:
    config = bootstrap.from_brief(
        "Two friends argue about whether hot dogs are sandwiches",
        overrides={"tone": "unhinged banter", "show_name": "Food Fight"},
    )

EXAMPLE — mock client for testing:
    from unittest.mock import Mock
    mock_client = Mock()
    mock_client.chat.return_value = '{"title": "Test", ...}'
    bootstrap = PodcastBootstrap(client=mock_client)
"""

from __future__ import annotations

import json

from ...core.client import DeepSeekClient
from ...core.search import BaseSearchProvider
from .config import (
    PodcastConfig,
    PodcastDirectorBrief,
    PodcastHostProfile,
    PodcastSegmentConfig,
)


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECT PROMPT
# ═══════════════════════════════════════════════════════════════════════════
# The Architect reads a brief and produces the skeleton config: show name,
# hosts, segments. Think of this as the casting call + episode outline.
#
# CRITICAL: "gender" is REQUIRED for every host. The Architect MUST set it
# based on the brief. This prevents the Trash Pandas bug (brief says
# "3 women" → one gets cast as male).

PODCAST_ARCHITECT_PROMPT = """\
You are the Architect for a podcast generation system. Given a brief
describing the podcast, produce a structured episode plan.

You must produce a JSON object with EXACTLY this schema:

{{
  "title": "string — the episode title",
  "show_name": "string — the podcast show name",
  "topic": "string — the main topic of this episode",
  "tone": "string — overall episode tone (playful, serious, chaotic, etc.)",
  "hosts": [
    {{
      "name": "string — host display name (e.g., 'The Maven', 'Dr. Sarah Chen')",
      "gender": "string — REQUIRED. 'male' or 'female'. If the brief specifies \
gender (e.g., '3 women discuss'), you MUST follow it. For unspecified genders, \
make a creative choice based on the character's name and role.",
      "role": "string — their function: 'lead host', 'co-host', 'guest', 'expert'",
      "speech_style": "string — how they talk: 'rapid-fire with pop culture \
references', 'measured and analytical', 'enthusiastic storyteller', etc.",
      "knowledge_domains": ["array of strings — topics this host speaks \
authoritatively on"],
      "forbidden_knowledge": ["array of strings — topics this host must NOT \
claim expertise in"],
      "interruption_tendency": "string — 'low', 'medium', or 'high'"
    }}
  ],
  "segments": [
    {{
      "segment_type": "string — 'intro', 'banter', 'deep_dive', 'drama', or 'outro'",
      "topic": "string — what this segment covers",
      "lead_host": "string — name of the host who drives this segment",
      "duration_hint": "string — 'short', 'medium', or 'long'",
      "target_words": "integer — approximate word count for this segment"
    }}
  ]
}}

RULES:
- The HOST list MUST match the brief. If the brief says "3 women," all 3
  hosts MUST have gender: "female". If it says "2 men and a woman," follow
  that exactly. Gender is NOT optional.
- Create 2-6 hosts depending on the brief. Most podcasts have 2-4 hosts.
- Each host MUST have a distinct personality and speech style. No two hosts
  should sound the same when their dialogue is generated.
- Plan 3-8 segments. Most episodes have:
  1. An intro segment (short)
  2. 1-3 main content segments (medium-long)
  3. Maybe a banter or drama segment for variety
  4. An outro segment (short)
- If the user specifies a word count or duration, plan enough segments and
  target_words to fill it. Rough guide: 150 words ≈ 1 minute of speech.
- knowledge_domains and forbidden_knowledge should be complementary across
  hosts — spread expertise so each host has a REASON to be there.
- ALWAYS respond with valid JSON. Nothing else.
"""


# ═══════════════════════════════════════════════════════════════════════════
# HOST PROFILER PROMPT
# ═══════════════════════════════════════════════════════════════════════════
# Takes a host stub and produces the rich personality fields: voice
# fingerprint, behavioral triggers, emotional arc, etc. Identical in
# concept to the debate DebateProfileBuilder but tuned for podcast hosts.

PODCAST_HOST_PROFILER_PROMPT = """\
You are a Character Designer for a podcast generation system. Your job is
to take a host stub (name + role + speech_style) and create a rich
personality profile that makes this host sound authentic and distinct.

You will receive:
  1. A host name, role, and speech style.
  2. The podcast topic.
  3. Optionally, reference material — factual source text.

Produce a JSON object with EXACTLY this schema:

{{
  "voice_hint": "string — a TTS casting note. Describe the ideal SOUND: \
accent, age range, energy, warmth, texture. Examples: 'warm alto, late 20s, \
California laid-back energy', 'sharp Brooklyn accent, 40s, fast-talking.'",

  "tts_style": "string — a persistent TTS instruction prepended to every \
line this host delivers. 1-2 sentences. Example: 'Speak with infectious \
enthusiasm and quick pacing, like a sports commentator on game day.'",

  "voice_fingerprint": "string — HOW this host talks. Rhetorical habits, \
sentence patterns, verbal tics, favorite phrases, use of questions vs \
statements. Be SPECIFIC. Generic descriptions are useless.",

  "behavioral_triggers": "string — how they react when excited, annoyed, \
interrupted, proven wrong, or when they're dominating the conversation.",

  "emotional_arc": "string — how their energy evolves over the episode. \
Start → middle → end.",

  "key_positions": ["array of strings — 5-10 talking points or perspectives \
they bring to this topic. Mix of serious analysis, hot takes, and personal \
connections."],

  "tangent_topics": ["array of strings — 3-6 personality-driven digressions. \
Anecdotes, analogies, jokes, references from their background."]
}}

RULES:
- voice_fingerprint is the MOST IMPORTANT field. Put real effort into
  specificity. Describe quirks, habits, patterns.
- key_positions should have DEPTH. A host in a long podcast may speak
  dozens of times — they need material to draw from.
- tangent_topics make conversations feel human. Make them character-specific.
- ALWAYS respond with valid JSON. Nothing else.
"""


# ═══════════════════════════════════════════════════════════════════════════
# DIRECTOR PROMPT
# ═══════════════════════════════════════════════════════════════════════════
# The Director plans the episode arc — how energy and mood flow across
# segments.

PODCAST_DIRECTOR_PROMPT = """\
You are the Director for a podcast episode. Your job is to plan the pacing
and emotional arc — how the energy flows from segment to segment.

You will receive details about the podcast: title, hosts, segments, and tone.

Produce a JSON object with EXACTLY this schema:

{{
  "arc": "string — the overall episode arc. Example: 'Open with high energy \
and humor, settle into focused analysis in the middle, ramp up with a \
controversial take in segment 4, then wind down warmly.'",
  "mood_progression": ["array of strings — ordered moods across the episode. \
4-8 entries. Example: ['excited', 'curious', 'focused', 'heated', 'reflective', 'warm']"],
  "segment_notes": {{
    "0": "optional per-segment note for segment 0",
    "2": "optional note for segment 2, key segments only"
  }},
  "closing_strategy": "string — how to end the episode. Example: 'Each host \
gives a one-sentence takeaway, the lead host thanks the listener, and they \
sign off with the show tagline.'"
}}

RULES:
- A podcast should have emotional variation — don't stay at one energy level.
- Build toward peaks and release tension afterward.
- mood_progression should have 4-8 entries mapping to the episode timeline.
- segment_notes are optional — only include them for segments that need
  specific directorial guidance.
- ALWAYS respond with valid JSON. Nothing else.
"""


class PodcastBootstrap:
    """Orchestrates the brief-to-config bootstrap chain for podcasts.

    Chain: brief → [Researcher] → Architect → Host profiler(s) → Director → config

    Attributes:
        _client: Shared DeepSeekClient passed to all bootstrap agents.
    """

    def __init__(
        self,
        client: DeepSeekClient | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        """Initialize the bootstrap orchestrator.

        Args:
            client:  An existing DeepSeekClient to use. In tests, inject a mock.
            api_key: API key for DeepSeekClient. Only used if client is None.
        """
        self._client = client or DeepSeekClient(api_key=api_key)

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
    ) -> PodcastConfig:
        """Run the full bootstrap chain: brief → fully populated PodcastConfig.

        Args:
            brief:              Natural language description of the podcast.
            reference_material: Source text or research material.
            fact_check:         "off", "quick", or "thorough".
            overrides:          Dict of PodcastConfig field names to override.
            auto_research:      If True, research the topic before generating.
            search_provider:    Search provider for web research.
            target_words:       Target word count for the episode.

        Returns:
            A fully populated PodcastConfig ready for PodcastConductor.
        """
        overrides = overrides or {}

        # ══════════════════════════════════════════════════════════════
        # STAGE 0: AUTO-RESEARCH (optional)
        # ══════════════════════════════════════════════════════════════
        # If the caller wants auto-research AND didn't provide their own
        # reference material, fire up the DebateResearcher to gather facts.
        # The researcher is domain-agnostic — it works for any topic.
        if auto_research and not reference_material:
            from ..debates.researcher import DebateResearcher
            if search_provider is None:
                researcher = DebateResearcher(self._client)
            else:
                researcher = DebateResearcher(self._client, search_provider)
            reference_material = researcher.run(brief) or None

        # ══════════════════════════════════════════════════════════════
        # STAGE 1: ARCHITECT
        # ══════════════════════════════════════════════════════════════
        # Reads the brief and produces the skeleton: show_name, hosts
        # (with gender!), segments, topic, tone.
        config = self._run_architect(
            brief, reference_material, fact_check, target_words=target_words,
        )

        # ══════════════════════════════════════════════════════════════
        # STAGE 2: HOST PROFILER (one per host)
        # ══════════════════════════════════════════════════════════════
        # Each host stub gets fleshed out: voice_fingerprint, behavioral
        # triggers, emotional_arc, key_positions, tangent_topics, etc.
        # Sequential for now — async later if needed.
        for i, host in enumerate(config.hosts):
            config.hosts[i] = self._profile_host(
                host,
                topic=config.topic or brief,
                reference_material=reference_material,
            )

        # ══════════════════════════════════════════════════════════════
        # STAGE 3: DIRECTOR
        # ══════════════════════════════════════════════════════════════
        # Plans the episode arc: mood progression, per-segment notes,
        # closing strategy.
        config.director_brief = self._run_director(config)

        # ══════════════════════════════════════════════════════════════
        # STAGE 4: APPLY OVERRIDES
        # ══════════════════════════════════════════════════════════════
        # User gets the last word. Any field in overrides replaces what
        # the agents produced.
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    # ═══════════════════════════════════════════════════════════════════
    # ARCHITECT
    # ═══════════════════════════════════════════════════════════════════

    def _run_architect(
        self,
        brief: str,
        reference_material: str | None,
        fact_check: str,
        *,
        target_words: int | None = None,
    ) -> PodcastConfig:
        """Call the Architect LLM to produce a skeleton PodcastConfig."""
        user_parts = [f"BRIEF:\n{brief}"]
        if reference_material:
            user_parts.append(f"\nREFERENCE MATERIAL:\n{reference_material}")
        if target_words:
            user_parts.append(
                f"\nTARGET LENGTH: {target_words:,} words total. "
                f"Plan enough segments to fill this — rough guide: "
                f"150 words ≈ 1 minute of speech. Distribute target_words "
                f"across segments proportionally."
            )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": PODCAST_ARCHITECT_PROMPT},
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
    ) -> PodcastConfig:
        """Parse the Architect's JSON into a PodcastConfig.

        Testable without the LLM — pass a canned JSON string.
        """
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        # Find the JSON object.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Podcast Architect produced invalid JSON. First 200 chars: "
                f"{raw_json[:200]!r}\nParse error: {e}"
            ) from e

        # ── Build hosts ──────────────────────────────────────────────
        hosts = []
        for h in data.get("hosts", []):
            hosts.append(PodcastHostProfile(
                name=h.get("name", ""),
                gender=h.get("gender"),  # REQUIRED — validate() will catch if missing
                role=h.get("role", "host"),
                speech_style=h.get("speech_style"),
                knowledge_domains=h.get("knowledge_domains", []),
                forbidden_knowledge=h.get("forbidden_knowledge", []),
                interruption_tendency=h.get("interruption_tendency", "medium"),
            ))

        # ── Build segments ───────────────────────────────────────────
        segments = []
        for s in data.get("segments", []):
            segments.append(PodcastSegmentConfig(
                segment_type=s.get("segment_type", "banter"),
                topic=s.get("topic", ""),
                lead_host=s.get("lead_host"),
                target_words=s.get("target_words"),
                duration_hint=s.get("duration_hint", "medium"),
            ))

        return PodcastConfig(
            brief=brief,
            reference_material=reference_material,
            fact_check=fact_check,
            title=data.get("title"),
            show_name=data.get("show_name"),
            topic=data.get("topic"),
            tone=data.get("tone"),
            hosts=hosts,
            segments=segments,
        )

    # ═══════════════════════════════════════════════════════════════════
    # HOST PROFILER
    # ═══════════════════════════════════════════════════════════════════

    def _profile_host(
        self,
        host: PodcastHostProfile,
        *,
        topic: str,
        reference_material: str | None = None,
    ) -> PodcastHostProfile:
        """Flesh out a host stub into a full personality profile.

        The original host's identity fields (name, gender, role, speech_style,
        knowledge_domains, forbidden_knowledge, interruption_tendency) are
        preserved. The profiler fills in: voice_fingerprint, behavioral_triggers,
        emotional_arc, key_positions, tangent_topics, voice_hint, tts_style.
        """
        user_parts = [
            f"HOST: {host.name}",
            f"ROLE: {host.role}",
            f"SPEECH STYLE: {host.speech_style or 'engaging and conversational'}",
            f"KNOWLEDGE DOMAINS: {', '.join(host.knowledge_domains) or 'general'}",
            f"TOPIC: {topic}",
        ]
        if reference_material:
            user_parts.append(f"\nREFERENCE MATERIAL:\n{reference_material}")

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": PODCAST_HOST_PROFILER_PROMPT},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            temperature=0,
        )

        return self._parse_host_profile(raw, host=host)

    def _parse_host_profile(
        self,
        raw_json: str,
        *,
        host: PodcastHostProfile,
    ) -> PodcastHostProfile:
        """Parse the profiler's JSON into an updated PodcastHostProfile.

        Preserves the host's identity fields, fills in personality fields.
        Testable without the LLM — pass a canned JSON string.
        """
        cleaned = raw_json.strip()
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
        except (json.JSONDecodeError, TypeError):
            # Fallback: keep the host as-is with a generic fingerprint.
            host.voice_fingerprint = "(natural, conversational podcast host)"
            return host

        # Fill in personality fields while preserving identity fields.
        host.voice_hint = data.get("voice_hint", host.voice_hint)
        host.tts_style = data.get("tts_style", host.tts_style)
        host.voice_fingerprint = data.get("voice_fingerprint", host.voice_fingerprint)
        host.behavioral_triggers = data.get("behavioral_triggers", host.behavioral_triggers)
        host.emotional_arc = data.get("emotional_arc", host.emotional_arc)
        host.key_positions = data.get("key_positions", host.key_positions)
        host.tangent_topics = data.get("tangent_topics", host.tangent_topics)

        return host

    # ═══════════════════════════════════════════════════════════════════
    # DIRECTOR
    # ═══════════════════════════════════════════════════════════════════

    def _run_director(self, config: PodcastConfig) -> PodcastDirectorBrief:
        """Call the Director LLM to plan the episode arc."""
        host_list = ", ".join(h.name for h in config.hosts)
        segment_list = "\n".join(
            f"  {i+1}. [{seg.segment_type}] {seg.topic} "
            f"(lead: {seg.lead_host or 'rotating'}, {seg.duration_hint})"
            for i, seg in enumerate(config.segments)
        )

        user_message = (
            f"TITLE: {config.title or 'Untitled'}\n"
            f"SHOW: {config.show_name or 'Untitled Podcast'}\n"
            f"TONE: {config.tone or 'engaging'}\n"
            f"HOSTS: {host_list}\n"
            f"TOTAL SEGMENTS: {len(config.segments)}\n"
            f"SEGMENTS:\n{segment_list}"
        )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": PODCAST_DIRECTOR_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )

        return self._parse_director_response(raw)

    def _parse_director_response(self, raw_json: str) -> PodcastDirectorBrief:
        """Parse the Director's JSON into a PodcastDirectorBrief.

        Testable without the LLM — pass a canned JSON string.
        """
        cleaned = raw_json.strip()
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
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Podcast Director produced invalid JSON. First 200 chars: "
                f"{raw_json[:200]!r}\nParse error: {e}"
            ) from e

        # Parse segment_notes — the Director returns string keys ("0", "2"),
        # we convert them to int keys for PodcastDirectorBrief.
        raw_notes = data.get("segment_notes", {})
        segment_notes: dict[int, str] = {}
        for key, value in raw_notes.items():
            try:
                segment_notes[int(key)] = str(value)
            except (ValueError, TypeError):
                pass  # Skip malformed keys

        return PodcastDirectorBrief(
            arc=data.get("arc", ""),
            mood_progression=data.get("mood_progression", []),
            segment_notes=segment_notes,
            closing_strategy=data.get("closing_strategy"),
        )
