"""News show configuration models.

A news show is structurally different from a debate:

    DEBATE:  Two+ people argue a single topic for the entire show.
             The structure emerges from the conversation itself.

    NEWS:    An anchor (or anchors) presents a SEQUENCE OF SEGMENTS,
             each covering a different story. Between segments, the
             show host provides transitions. Some segments might be
             anchor reads (scripted), others might be reporter packages
             or panel discussions (reactive).

THE CAST:

    NewsConfig          The whole blueprint — everything the conductor needs.
    AnchorConfig        The news anchor(s) — the voice of the broadcast.
    SegmentConfig       One story in the rundown — what to cover and how.
    DirectorBrief       Pacing and dramatic arc for the full broadcast.

RE-EXPORTED FROM CORE (shared across all show types):

    ParticipantProfile  For guests, panelists, field reporters, etc.
    ShowHostConfig      The narrative wrapper (intro/transitions/outro).

HOW THESE GET BUILT:

    Option A — Full auto ("just give me a newscast"):
        bootstrap = NewsBootstrap()
        config = bootstrap.from_brief("Top tech and AI stories this week")

    Option B — Auto with tweaks:
        config = bootstrap.from_brief(
            "Breaking: major climate summit results",
            overrides={"target_segments": 5, "tone": "urgent"},
        )

    Option C — Fully manual:
        config = NewsConfig(
            brief="...",
            title="The Evening Brief",
            segments=[SegmentConfig(headline="...", angle="...")],
            anchors=[AnchorConfig(name="Jordan Lee", style="authoritative")],
        )
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.config import ContentConfig, ParticipantProfile, ShowHostConfig


@dataclass
class AnchorConfig:
    """Configuration for a news anchor.

    Attributes:
        name:       Display name (e.g., "Jordan Lee").
        style:      Free-form delivery style description.
                    Example: "Warm, authoritative NPR-style delivery."
        voice_fingerprint:
                    HOW this anchor talks — cadence, patterns, verbal habits.
        voice_gender:
                    The vocal gender for TTS voice assignment.
                    "male", "female", or None (auto-detect).
        voice_hint:
                    Casting note for the renderer's voice catalog — what
                    this anchor should sound like. E.g., "clear, warm,
                    authoritative, American male, 40s".
        tts_style:
                    Persistent TTS instruction for this anchor's vocal
                    identity. Prepended to every utterance's emotion hint.
    """

    name: str = ""
    style: str | None = None
    voice_fingerprint: str | None = None
    voice_gender: str | None = None
    voice_hint: str | None = None
    tts_style: str | None = None


@dataclass
class SegmentConfig:
    """One story in the rundown — the blueprint for a single news segment.

    The conductor generates content for each segment in order. Some segments
    are simple anchor reads (the anchor narrates facts), others might include
    reporter soundbites or expert quotes woven into the anchor's delivery.

    Attributes:
        headline:       The story headline (short, punchy).
        angle:          The editorial angle or focus for this story.
        key_facts:      Bullet points of verified facts for this story.
        tone:           Segment-specific tone override (e.g., "somber" for
                        tragedy, "upbeat" for human interest).
        duration_hint:  Rough guide — "short" (~30s), "medium" (~60s),
                        "long" (~90-120s). Defaults to "medium".
        segment_type:   "anchor_read", "package", "panel", "breaking".
                        Defaults to "anchor_read" — the anchor narrates.
    """

    headline: str = ""
    angle: str | None = None
    key_facts: list[str] = field(default_factory=list)
    tone: str | None = None
    duration_hint: str = "medium"
    segment_type: str = "anchor_read"
    target_beats: int = 1  # How many beats to generate (lead, body, tag, etc.)


@dataclass
class DirectorBrief:
    """Pacing plan for the full broadcast.

    Attributes:
        show_arc:           High-level narrative structure.
                            Example: "Open with the lead story (heaviest news),
                            middle segments softer, close with kicker."
        mood_progression:   Ordered list of moods for the broadcast arc.
        segment_order_notes: Any notes on story ordering strategy.
        closing_strategy:   How to end the broadcast.
        check_interval:     How many segments between director check-ins.
    """

    show_arc: str = ""
    mood_progression: list[str] = field(default_factory=list)
    segment_order_notes: str | None = None
    closing_strategy: str | None = None
    check_interval: int = 3
    outline_sections: list | None = None  # list of OutlineSection dicts


@dataclass
class NewsConfig(ContentConfig):
    """Complete configuration for generating a news show.

    Extends ContentConfig with news-specific fields: anchors, segments
    (the rundown), director brief, and show host.

    TYPICAL LIFECYCLE:
        1. User creates with just a brief:
               config = NewsConfig(brief="Top tech stories this week")

        2. Bootstrap agents fill in the rest:
               config = NewsBootstrap().from_brief(config.brief)

        3. Conductor generates the show:
               conductor = NewsConductor(config)
               result = conductor.run()

    Attributes:
        topic:          The broadcast topic or theme.
        tone:           Overall broadcast tone ("authoritative", "conversational").
        setting:        Where the broadcast takes place ("studio", "field").
        anchors:        The anchor(s) delivering the news.
        segments:       The rundown — ordered list of stories to cover.
        show_host:      Narrative wrapper (intro/transitions/outro).
        director_brief: Pacing plan for the broadcast.
        target_segments: How many segments to generate.
        guests:         Optional guest profiles (for panel segments).
    """

    content_type: str = "news"
    flexibility_score: float = 0.5
    topic: str | None = None
    tone: str | None = None
    setting: str | None = None
    anchors: list[AnchorConfig] = field(default_factory=list)
    segments: list[SegmentConfig] = field(default_factory=list)
    show_host: ShowHostConfig | None = None
    director_brief: DirectorBrief | None = None
    target_segments: int | None = None
    guests: list[ParticipantProfile] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Check this config for problems."""
        problems = super().validate()

        if not self.anchors:
            problems.append("No anchors defined. A news show needs at least one anchor.")

        for i, a in enumerate(self.anchors):
            if not a.name.strip():
                problems.append(f"Anchor {i} has no name.")

        if not self.segments:
            problems.append("No segments defined. A news show needs at least one story.")

        for i, s in enumerate(self.segments):
            if not s.headline.strip():
                problems.append(f"Segment {i} has no headline.")

        names = [a.name for a in self.anchors]
        seen: set[str] = set()
        for name in names:
            if name in seen:
                problems.append(f"Duplicate anchor name: '{name}'.")
            seen.add(name)

        return problems
