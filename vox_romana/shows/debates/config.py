"""Debate-specific configuration models.

This is where "I want a debate about X" turns into a fully specified blueprint
that the DebateConductor can actually execute.

THE CAST OF CHARACTERS (dataclasses in this file):

    DebateConfig          The whole enchilada. Everything the conductor needs.
    ModeratorConfig       Who runs the show and how.
    AudienceConfig        Is there a crowd? Do they participate?
    DirectorBrief         The dramatic arc — tension, pacing, resolution.

RE-EXPORTED FROM CORE (shared across all show types):

    ParticipantProfile    A participant's complete personality file.
    ShowHostConfig        The narrative host / emcee configuration.

HOW THESE GET BUILT:

    Option A — Full auto ("just give me a debate"):
        bootstrap = DebateBootstrap()
        config = bootstrap.from_brief("Should cities ban cars downtown?")
        # Architect, Profile Builder, and Director agents fill everything in.

    Option B — Auto with tweaks:
        config = bootstrap.from_brief(
            "Should cities ban cars downtown?",
            overrides={"debate_format": "oxford", "target_turns": 40},
        )

    Option C — Fully manual (you fill in everything yourself):
        config = DebateConfig(
            brief="...",
            title="Cars in the City",
            topic="urban transit policy",
            debate_format="public_forum",
            participants=[
                ParticipantProfile(name="Dr. Chen", side="pro-ban", ...),
                ParticipantProfile(name="Mayor Ortiz", side="anti-ban", ...),
            ],
            moderator=ModeratorConfig(name="Jordan Lee"),
            ...
        )

MUTABILITY NOTE:
    All config dataclasses here are mutable (no frozen=True). The bootstrap
    process fills them in incrementally — the Architect creates the skeleton,
    then the Profile Builder fleshes out each participant, then the Director
    adds the dramatic arc. Freezing would require creating new objects at
    every step, which is pointlessly awkward for config data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.config import ContentConfig, ParticipantProfile, ShowHostConfig


# ─────────────────────────────────────────────────────────────────────────────
# ModeratorConfig
# ─────────────────────────────────────────────────────────────────────────────
# The referee. Not every debate needs one (crossfire debates are free-for-alls),
# but when present, the moderator controls turn order, enforces rules, and
# keeps things from going off the rails.

@dataclass
class ModeratorConfig:
    """Configuration for the debate moderator.

    Attributes:
        name:   The moderator's display name.

        mode:   How actively they intervene:
                "active"  — Asks follow-up questions, pushes back, redirects.
                "passive" — Only manages turn order and time.
                "none"    — No moderator. Participants self-organize.
                (Default: "active")

        style:  Free-form description of their moderation style.
                Example: "Socratic — asks probing questions rather than
                making statements. Slightly favors the underdog."
    """

    name: str = ""
    mode: str = "active"
    style: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# AudienceConfig
# ─────────────────────────────────────────────────────────────────────────────
# The crowd. Optional. When present, they add reactions (applause, laughter,
# murmurs) and can even ask questions.

@dataclass
class AudienceConfig:
    """Configuration for the debate audience.

    Attributes:
        present:          Is there an audience at all?
        participation:    Can they ask questions / interject?
        size_description: Vibes, not numbers.
                          Example: "Small academic seminar, ~30 people"
                          or "Packed town hall, standing room only"
    """

    present: bool = False
    participation: bool = False
    size_description: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# DirectorBrief
# ─────────────────────────────────────────────────────────────────────────────
# The dramatic arc. This is the Director agent's game plan — it tells the
# conductor *when* to crank up the tension, *when* to let things breathe,
# and *how* to land the ending.
#
# Think of it like a film director's shot list, but for conversational
# dynamics instead of camera angles.

@dataclass
class DirectorBrief:
    """The dramatic arc plan for the debate.

    The Director agent runs every `check_interval` turns during generation.
    It reads the current state of the debate and nudges things toward the
    plan laid out here. It's not a rigid script — it's more like a jazz
    chart. The players improvise, but within a structure.

    Attributes:
        dramatic_arc:
                    High-level narrative structure.
                    Example: "Three-act structure. Act 1 (turns 1-10):
                    establish positions, polite disagreement. Act 2
                    (turns 11-25): escalation, specific rebuttals, one
                    interruption. Act 3 (turns 26-35): peak tension,
                    then gradual de-escalation and closing statements."

        mood_progression:
                    Ordered list of moods the debate should move through.
                    The Director uses these as waypoints.
                    Example: ["academic", "engaged", "heated",
                              "personal", "reconciliatory"]

        key_moments:
                    Specific dramatic beats the Director should try to
                    engineer. Not guaranteed — the agents are autonomous —
                    but the Director will steer toward these.
                    Example: ["Moderator challenges the frontrunner's
                    strongest claim", "Audience member asks an
                    uncomfortable question", "One participant makes
                    a surprising concession"]

        closing_strategy:
                    How to end it. The Director signals "wrap it up"
                    to the moderator when the debate approaches its
                    target length.
                    Example: "Each participant gets a 2-turn closing
                    statement. Moderator summarizes key disagreements."

        check_interval:
                    How many turns between Director check-ins. Lower =
                    more control, higher = more organic. 7 is a good
                    default. Setting it to 1 makes the Director review
                    every single turn (expensive, micromanaging).
    """

    dramatic_arc: str = ""
    mood_progression: list[str] = field(default_factory=list)
    key_moments: list[str] = field(default_factory=list)
    closing_strategy: str | None = None
    check_interval: int = 7


# ─────────────────────────────────────────────────────────────────────────────
# DebateConfig
# ─────────────────────────────────────────────────────────────────────────────
# The main event. This bundles everything above into one object that the
# DebateConductor consumes.

@dataclass
class DebateConfig(ContentConfig):
    """Complete configuration for generating a debate.

    This is the single object that the DebateConductor needs to generate
    a full debate. It extends ContentConfig (which has brief, fact_check,
    reference_material) with debate-specific fields.

    TYPICAL LIFECYCLE:
        1. User creates a DebateConfig with just a brief:
               config = DebateConfig(brief="Should AI art be copyrightable?")

        2. Bootstrap agents fill in the rest:
               config = DebateBootstrap().from_brief(config.brief)
           Now config.participants, config.director_brief, etc. are populated.

        3. User optionally tweaks anything:
               config.participants[0].side = "pro-copyright"
               config.target_turns = 50

        4. Conductor generates the debate:
               conductor = DebateConductor(config)
               result = conductor.run()

    Attributes:
        topic:          The debate topic, distilled to a clear statement.
                        Example: "Should universal basic income replace
                        existing welfare programs?"

        debate_format:  The structure of the debate. Affects turn order,
                        interruption rules, and moderator behavior.
                        Examples: "public_forum", "crossfire", "oxford",
                        "panel", "town_hall", "roundtable"

        setting:        Where this debate is happening. Affects audience
                        behavior and general tone.
                        Example: "University lecture hall", "Cable news
                        studio", "Community town hall"

        tone:           Overall vibe the Director should aim for.
                        Example: "academic but accessible", "cable news
                        energy", "TED talk civility"

        participants:   List of ParticipantProfile objects. Usually 2-4
                        debaters. Filled in by the Architect and Profile
                        Builder during bootstrap.

        moderator:      ModeratorConfig, or None for unmoderated debates.

        audience:       AudienceConfig, or None for no audience.

        show_host:      ShowHostConfig, or None for no narrative host.
                        When present, the conductor generates intro, transition,
                        and outro events spoken by this host character.

        director_brief: The dramatic arc plan. Filled in by the Director
                        Brief agent during bootstrap.

        target_turns:   How many speaking turns to generate before wrapping
                        up. A rough guide, not a hard limit — the conductor
                        will let a natural ending happen rather than cutting
                        off mid-sentence at exactly turn 40.
                        Rule of thumb: ~1 turn per 30-45 seconds of audio.
                        So target_turns=40 ≈ a 20-30 minute debate.
    """

    # ── Override the content_type from the base class ──────────────────────
    content_type: str = "debate"

    # ── Debate-specific fields ─────────────────────────────────────────────
    topic: str | None = None
    debate_format: str | None = None
    setting: str | None = None
    tone: str | None = None
    participants: list[ParticipantProfile] = field(default_factory=list)
    moderator: ModeratorConfig | None = None
    audience: AudienceConfig | None = None
    show_host: ShowHostConfig | None = None
    director_brief: DirectorBrief | None = None
    target_turns: int | None = None

    def validate(self) -> list[str]:
        """Check this config for problems. Returns a list of warning strings.

        Extends the base ContentConfig.validate() with debate-specific checks.

        Example:
            config = DebateConfig(brief="test")
            problems = config.validate()
            for p in problems:
                print(f"  ⚠ {p}")
            # Output:
            #   ⚠ No participants defined. Need at least 2 for a debate.
        """
        # Start with the base class checks (brief, fact_check, etc.)
        problems = super().validate()

        # ── Participant checks ─────────────────────────────────────────
        if len(self.participants) < 2:
            problems.append(
                f"Need at least 2 participants for a debate, got {len(self.participants)}. "
                "A debate with fewer than 2 people is just a monologue."
            )

        # Check for duplicate names — the conductor uses names to route
        # turns, so duplicates would cause chaos.
        names = [p.name for p in self.participants]
        seen: set[str] = set()
        for name in names:
            if name in seen:
                problems.append(f"Duplicate participant name: '{name}'.")
            seen.add(name)

        # Every participant should have a name
        for i, p in enumerate(self.participants):
            if not p.name.strip():
                problems.append(f"Participant {i} has no name.")

        # ── Moderator checks ──────────────────────────────────────────
        if self.moderator and not self.moderator.name.strip():
            problems.append("Moderator is configured but has no name.")

        # ── Target turns sanity check ─────────────────────────────────
        if self.target_turns is not None and self.target_turns < 4:
            problems.append(
                f"target_turns={self.target_turns} is too few. Even a short "
                "debate needs at least 4 turns to say hello and goodbye."
            )

        return problems
