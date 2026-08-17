"""Podcast configuration models.

A podcast is a REACTIVE multi-host show — two or more hosts discuss topics
across structured segments. Unlike debates (adversarial, turn-level) or
audiobooks (scripted, chapter-level), podcasts are conversational and
generate at the segment level: one LLM call produces multiple speakers'
lines for an entire segment.

Config hierarchy:
    ContentConfig (core)
        └── PodcastConfig
                ├── PodcastHostProfile[] — the cast of hosts
                ├── PodcastSegmentConfig[] — ordered segment outline
                └── PodcastDirectorBrief — pacing/mood arc

GENDER ENFORCEMENT:
    PodcastHostProfile.gender is a REQUIRED field (Literal["male", "female"]).
    This fixes a known bug from the debate pipeline where the Architect
    could ignore gender specifications in the prompt (the "Trash Pandas"
    bug: prompt said "3 women" but output had 2 women + 1 man). The
    podcast Architect schema makes gender mandatory, and validate() warns
    if any host is missing one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ...core.config import ContentConfig, VoiceProfile


# ─────────────────────────────────────────────────────────────────────────────
# PodcastHostProfile
# ─────────────────────────────────────────────────────────────────────────────
# Like ParticipantProfile but with podcast-specific fields: knowledge
# domains, forbidden knowledge, interruption tendency, and REQUIRED gender.

@dataclass
class PodcastHostProfile(VoiceProfile):
    """Complete personality profile for a podcast host.

    This extends the concept of ParticipantProfile with podcast-specific
    fields. Hosts aren't adversarial debaters — they're conversational
    collaborators with distinct domains of expertise and personality quirks.

    EXAMPLES:
        # A sports-savvy host
        PodcastHostProfile(
            name="The Maven",
            gender="female",
            role="lead host",
            knowledge_domains=["women's soccer", "Olympic history"],
            forbidden_knowledge=["cricket rules", "F1 regulations"],
            interruption_tendency="medium",
        )

        # A laid-back co-host
        PodcastHostProfile(
            name="Chill Carlos",
            gender="male",
            role="co-host",
            knowledge_domains=["music production", "pop culture"],
            interruption_tendency="low",
        )

    Attributes:
        name:                   Display name (e.g., "The Maven").
        gender:                 Required. "male" or "female". Drives TTS
                                voice selection. NOT optional — this is the
                                fix for the Trash Pandas casting bug.
        role:                   Function in the show: "lead host", "co-host",
                                "guest", "expert", etc.
        speech_style:           Free-form style description. The Architect
                                fills this in (e.g., "rapid-fire with lots
                                of pop culture references").
        knowledge_domains:      Topics this host speaks authoritatively on.
                                Drives segment assignment and expertise routing.
        forbidden_knowledge:    Topics this host must NEVER speak as an expert
                                (e.g., "baseball rules"). They can REACT to
                                these topics but shouldn't claim authority.
        interruption_tendency:  How often this host interrupts others.
                                "low" = polite, waits for pauses.
                                "medium" = normal conversational overlap.
                                "high" = frequently jumps in, overlaps.
        voice_fingerprint:      HOW this host talks — cadence, verbal tics,
                                sentence patterns. Filled by the Host Profiler.
        behavioral_triggers:    How they react under pressure or excitement.
        emotional_arc:          How their energy evolves over the episode.
        key_positions:          Main talking points or perspectives they bring.
        tangent_topics:         Personality-driven digressions — anecdotes,
                                analogies they naturally reach for.
        voice_hint:             TTS casting note (e.g., "warm alto, 30s").
        tts_style:              Persistent TTS instruction for vocal identity.
    """

    name: str = ""
    gender: Literal["male", "female"] | None = None
    role: str = "host"
    speech_style: str | None = None
    knowledge_domains: list[str] = field(default_factory=list)
    forbidden_knowledge: list[str] = field(default_factory=list)
    interruption_tendency: Literal["low", "medium", "high"] = "medium"
    behavioral_triggers: str | None = None
    emotional_arc: str | None = None
    key_positions: list[str] = field(default_factory=list)
    tangent_topics: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# PodcastSegmentConfig
# ─────────────────────────────────────────────────────────────────────────────
# Each segment is a self-contained conversational block. The conductor
# generates one segment per LLM call, producing all speakers' lines at once.

@dataclass
class PodcastSegmentConfig:
    """Configuration for a single podcast segment.

    Segments are the structural unit of a podcast — like chapters in an
    audiobook or turns in a debate, but at a higher level. One segment
    produces 3-15 speaker turns in a single LLM call.

    EXAMPLES:
        # Cold open
        PodcastSegmentConfig(
            segment_type="intro",
            topic="Welcome to Trash Pandas — tonight's topic: soccer rivalries",
            lead_host="The Maven",
            duration_hint="short",
        )

        # Deep dive on a specific subtopic
        PodcastSegmentConfig(
            segment_type="deep_dive",
            topic="The 1999 Women's World Cup final",
            lead_host="The Maven",
            target_words=800,
            duration_hint="long",
        )

    Attributes:
        segment_type:   The kind of segment. Controls tone and pacing:
                        "intro"     — cold open, topic setup, host intros
                        "banter"    — light back-and-forth, personality-driven
                        "deep_dive" — focused analysis on a specific topic
                        "drama"     — high-energy debate or heated discussion
                        "outro"     — wrap-up, takeaways, sign-off
        topic:          What this segment is about. The LLM uses this as
                        the primary generation prompt for the segment.
        lead_host:      Name of the host who drives this segment. If None,
                        the conductor rotates hosts.
        target_words:   Approximate word count for this segment. If None,
                        the conductor uses duration_hint to estimate.
        duration_hint:  Rough length guide: "short" (~200 words),
                        "medium" (~400 words), "long" (~800+ words).
    """

    segment_type: Literal["intro", "banter", "deep_dive", "drama", "outro"] = "banter"
    topic: str = ""
    lead_host: str | None = None
    target_words: int | None = None
    duration_hint: Literal["short", "medium", "long"] = "medium"


# ─────────────────────────────────────────────────────────────────────────────
# PodcastDirectorBrief
# ─────────────────────────────────────────────────────────────────────────────
# The Director's plan for how the episode arc should feel.

@dataclass
class PodcastDirectorBrief:
    """Pacing and arc plan for the podcast episode.

    Produced by the Director agent during bootstrap. The conductor uses
    this to inject per-segment mood notes and decide when to compress
    context.

    Attributes:
        arc:                   Overall episode arc description.
        mood_progression:      Ordered moods across the episode.
        segment_notes:         Per-segment director notes, keyed by segment
                               index (0-based). Only segments with specific
                               notes are included.
        closing_strategy:      How to end the episode.
        compression_interval:  Compress the context summary every N segments.
                               Lower = more frequent compression = smaller
                               context but more LLM calls.

                               .. note::
                                   Unlike the ``check_interval`` field on
                                   other shows' ``DirectorBrief`` classes,
                                   this field drives **context compression**
                                   (collapsing older turns into a summary),
                                   *not* director interjections. The podcast
                                   conductor also uses it as the director
                                   check-in cadence as a convenience, but the
                                   primary purpose is context management.
    """

    arc: str = ""
    mood_progression: list[str] = field(default_factory=list)
    segment_notes: dict[int, str] = field(default_factory=dict)
    closing_strategy: str | None = None
    compression_interval: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# PodcastConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PodcastConfig(ContentConfig):
    """Full configuration for podcast generation.

    Extends ContentConfig with podcast-specific fields: hosts (the cast),
    segments (the episode outline), and director brief (pacing plan).

    USAGE:
        config = PodcastConfig(
            brief="Three women discuss soccer rivalries",
            show_name="Trash Pandas",
            hosts=[
                PodcastHostProfile(name="The Maven", gender="female", ...),
                PodcastHostProfile(name="The Rookie", gender="female", ...),
                PodcastHostProfile(name="The Historian", gender="female", ...),
            ],
            segments=[
                PodcastSegmentConfig(segment_type="intro", topic="Welcome to Trash Pandas"),
                PodcastSegmentConfig(segment_type="deep_dive", topic="1999 WWC Final"),
                PodcastSegmentConfig(segment_type="outro", topic="Wrap-up"),
            ],
        )
    """

    content_type: str = "podcast"
    flexibility_score: float = 0.7
    show_name: str | None = None  # e.g., "Trash Pandas", "The Deep Dive"
    topic: str | None = None  # Main episode topic
    tone: str | None = None  # e.g., "playful", "serious", "chaotic energy"
    hosts: list[PodcastHostProfile] = field(default_factory=list)
    segments: list[PodcastSegmentConfig] = field(default_factory=list)
    director_brief: PodcastDirectorBrief | None = None
    music_style: str = ""

    def validate(self) -> list[str]:
        """Return a list of validation problems (empty = valid).

        Checks podcast-specific requirements on top of base ContentConfig
        validation. Warns (but doesn't hard-fail) on missing gender — this
        is the Trash Pandas bug fix: gender should be set, and we make it
        very visible if it isn't.
        """
        problems = super().validate()

        if not self.hosts:
            problems.append("PodcastConfig requires at least one host.")

        if not self.segments:
            problems.append("PodcastConfig requires at least one segment.")

        # Gender enforcement: warn loudly if any host is missing gender.
        # The Architect schema makes it required, but we check here too
        # in case someone builds a config manually.
        for host in self.hosts:
            if not host.name:
                problems.append("Every host must have a name.")
            if not host.gender:
                problems.append(
                    f"Host '{host.name or '(unnamed)'}' is missing gender. "
                    f"Gender is required for podcast hosts to ensure correct "
                    f"TTS voice casting."
                )

        # Check segment lead_host references actually exist in the cast.
        host_names = {h.name for h in self.hosts}
        for i, seg in enumerate(self.segments):
            if seg.lead_host and seg.lead_host not in host_names:
                problems.append(
                    f"Segment {i} lead_host '{seg.lead_host}' is not in the "
                    f"host list. Known hosts: {', '.join(host_names)}."
                )

        return problems
