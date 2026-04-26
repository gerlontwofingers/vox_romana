"""Base configuration for all vox_romana content types.

Think of ContentConfig as a "job order" you hand to the factory floor.
You (the human) fill in what you know — maybe just a one-sentence idea,
maybe a detailed spec — and the system's bootstrap agents fill in the rest.

This base class is deliberately thin. It only holds what's common to EVERY
content type: debates, news shows, audiobooks, whatever comes next. Each
content type subclasses this and adds its own domain-specific fields.

WHY A BASE CLASS?
    Because the bootstrap pattern is the same everywhere:
    1. User provides a brief  (natural language, any length)
    2. System generates a full config  (via LLM agents)
    3. Config feeds into a conductor or renderer  (generates the content)
    4. Output goes through pipeline → TTS

    The fields here (brief, reference_material, fact_check) are needed at
    every step regardless of content type. Debate-specific stuff like
    "participants" lives in DebateConfig.

USAGE:
    # You never instantiate ContentConfig directly. Use a subclass:
    from vox_romana.shows.debates.config import DebateConfig

    config = DebateConfig(
        brief="Two economists debate universal basic income",
        reference_material="Finland ran a UBI pilot in 2017-2018...",
        fact_check="thorough",
    )

MUTABILITY:
    Unlike our DebateEvent / DebateSegment models (which are frozen — immutable
    output data), config objects are MUTABLE. That's because the bootstrap
    process fills them in progressively:
        1. User sets brief + reference_material
        2. Architect agent fills in title, topic, participants
        3. Profile Builder fills in personality details
        4. Director fills in dramatic arc
    Freezing them would make this incremental build-up a nightmare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── The valid fact-check modes ─────────────────────────────────────────────


# ── The valid fact-check modes ─────────────────────────────────────────────
# These are the three speeds on the fact-checking dial:
#
#   "off"      — No checking. Fast. Use during development or when you trust
#                your reference material to fully constrain the output.
#
#   "quick"    — LLM-only verification. The system reads each claim and checks
#                it against your reference_material. Catches obvious fabrications.
#                ~$0.02 per debate, ~15 seconds.
#                Think of it as a spell-checker for facts.
#
#   "thorough" — LLM check PLUS web search for flagged claims. Anything the
#                LLM can't verify from reference_material gets a real web search.
#                ~$0.10 per debate, ~2-4 minutes.
#                Think of it as a human fact-checker with a search engine.

FactCheckMode = Literal["off", "quick", "thorough"]


# ── Content modes for fact-checking behaviour ──────────────────────────────
# These tell the FactChecker WHAT KIND of truth a given show is committing to:
#
#   "factual"             — Real-world content. Every concrete claim must be
#                            verifiable against reality (or reference_material).
#                            Fabrications BLOCK publication and trigger rewrites.
#                            News and debates are always this mode.
#
#   "fictional-universe"  — Content set in an established fictional world
#                            (Star Wars, Marvel, Tolkien, Harry Potter, …).
#                            The check is against established CANON, not
#                            real-world facts. Luke Skywalker cannot be from
#                            Kansas — he is from Tatooine. Canon violations
#                            are WARNINGS (CANON_WARN) rather than blockers,
#                            because storytelling sometimes bends canon.
#
#   "pure-fiction"        — Wholly original fiction with no real-world or
#                            established-canon claims. The FactChecker SKIPS
#                            entirely (no LLM calls). Use this for original
#                            short stories, made-up worlds, etc.

ContentMode = Literal["factual", "fictional-universe", "pure-fiction"]


@dataclass
class ContentConfig:
    """Base configuration that every content type shares.

    Attributes:
        content_type:  A short string identifying the kind of content.
                       Subclasses set this automatically (e.g., "debate").

        brief:         The human's description of what they want. Can be one
                       sentence ("two people argue about pineapple on pizza")
                       or a multi-paragraph spec. The bootstrap agents will
                       work with whatever you give them.

        reference_material:
                       Optional block of factual text that grounds the content.
                       Think of it as the "source documents" a journalist would
                       use. Every agent gets this injected into its prompt so
                       it can only cite real information, not hallucinated junk.

                       For debates: include key statistics, study names, real
                       quotes, timelines. The more you give, the more factual
                       the output.

        fact_check:    How aggressively to verify claims after generation.
                       See FactCheckMode above for the three options.

        title:         Human-readable title. Usually filled in by the Architect
                       agent during bootstrap, but you can set it manually.

        estimated_duration_minutes:
                       Rough target length. The conductor uses this to decide
                       how many turns to generate before wrapping up.

        target_duration_minutes:
                       Precise target length in minutes used by the adaptive
                       length system.  When set, the conductor estimates a
                       word budget and enforces ±tolerance_percent.

        target_words:  Explicit word-count target. When set, takes priority
                       over ``target_duration_minutes`` (the conductor uses
                       this count directly instead of converting minutes to
                       words). Only one of the two should be set; setting
                       both is allowed but ``target_words`` wins.

        tolerance_percent:
                       How much deviation (in %) from the word budget is
                       acceptable.  Default: 10.0 (±10%).

        flexibility_score:
                       0.0–1.0 hint to the conductor about how strictly to
                       enforce the target.  0.0 = rigid (audiobooks), 1.0 =
                       loose (podcasts).  Content-type configs override this.
    """

    content_type: str = "generic"
    brief: str = ""
    reference_material: str | None = None
    fact_check: FactCheckMode = "off"
    content_mode: ContentMode = "factual"
    title: str | None = None
    language: str = "en"
    estimated_duration_minutes: int | None = None
    target_duration_minutes: float | None = None
    target_words: int | None = None
    tolerance_percent: float = 10.0
    flexibility_score: float = 0.5
    max_retries: int = 3
    allow_bootstrap_fabrications: bool = False

    def validate(self) -> list[str]:
        """Return a list of problems with this config. Empty list = all good.

        This is a soft check — it returns strings, not exceptions. That way
        you can inspect what's wrong before deciding whether to bail out.

        Example:
            problems = config.validate()
            if problems:
                for p in problems:
                    print(f"  ⚠ {p}")
        """
        problems: list[str] = []

        if not self.brief.strip():
            problems.append("brief is empty — the system needs SOMETHING to work with.")

        if self.fact_check not in ("off", "quick", "thorough"):
            problems.append(
                f"fact_check must be 'off', 'quick', or 'thorough', got '{self.fact_check}'."
            )

        if self.content_mode not in ("factual", "fictional-universe", "pure-fiction"):
            problems.append(
                "content_mode must be 'factual', 'fictional-universe', or "
                f"'pure-fiction', got '{self.content_mode}'."
            )

        # If you're asking for fact-checking but gave no reference material,
        # that's like asking a librarian to verify a book with an empty library.
        if self.fact_check != "off" and not self.reference_material:
            problems.append(
                "fact_check is enabled but reference_material is empty. "
                "The fact-checker needs source material to verify against."
            )

        return problems


# ─────────────────────────────────────────────────────────────────────────────
# VoiceProfile  (shared mixin for any voiced role)
# ─────────────────────────────────────────────────────────────────────────────
# Every show-type role that gets a TTS voice — debater, anchor, narrator,
# podcast host, show host — needs the same four pieces of voice metadata.
# Centralising them here removes duplicate field declarations from each
# show package's config module.
#
#   voice_gender:       "male" | "female" | None (auto-detect)
#   voice_hint:         Casting note for the renderer's voice catalog
#                       (e.g., "deep authoritative baritone, 50s").
#   tts_style:          Persistent TTS instruction prepended to every
#                       utterance (engines that support it use it; others
#                       ignore gracefully).
#   voice_fingerprint:  Description of HOW this person talks — cadence,
#                       rhetorical habits, verbal tics.
#
# Concrete profile classes inherit from VoiceProfile and add their own
# domain-specific fields.

@dataclass
class VoiceProfile:
    """Common voice-casting fields shared by every voiced role.

    Inherited by :class:`ParticipantProfile`, :class:`ShowHostConfig`,
    and the show-specific profile classes (AnchorConfig, NarratorConfig,
    PodcastHostProfile). Carries only the fields the TTS layer cares
    about — content/personality fields live on the subclasses.
    """

    voice_gender: str | None = None
    voice_hint: str | None = None
    tts_style: str | None = None
    voice_fingerprint: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# ParticipantProfile
# ─────────────────────────────────────────────────────────────────────────────
# The "character sheet" for any voiced participant in a show — debaters,
# anchors, reporters, panelists, guests. Every show type reuses this.

@dataclass
class ParticipantProfile(VoiceProfile):
    """Complete personality profile for a show participant.

    Used by debates (debaters), news shows (anchors, reporters, panelists),
    podcasts (hosts, guests), and any future content type that has people
    talking.

    Attributes:
        name:       Full display name (e.g., "Dr. Sarah Chen").

        side:       Which position they're arguing. Free-form text —
                    "pro-ban", "anti-regulation", "moderate", whatever
                    fits. None if not applicable (anchors, narrators).

        role:       Their function in the show. Examples: "participant",
                    "moderator", "anchor", "reporter", "panelist", "guest".
                    (Default: "participant")

        voice_fingerprint:
                    HOW this person talks, not WHAT they believe.
                    Rhetorical habits, sentence patterns, verbal tics.

        behavioral_triggers:
                    How they react to specific situations.

        emotional_arc:
                    How their demeanor should evolve over the show.

        key_positions:
                    The main arguments or talking points.

        factual_claims:
                    Specific, verifiable claims grounded in reference material.

        reference_corpus:
                    Real quotes or excerpts for few-shot style matching.

        tangent_topics:
                    Personality-driven digressions this person naturally
                    reaches for — anecdotes, analogies from other fields,
                    personal stories, rhetorical diversions. Used when core
                    arguments are exhausted to keep dialogue fresh.

        voice_gender:
                    The vocal gender for TTS voice assignment.
                    "male", "female", or None (auto-detect from name/context).
                    Set by the profile builder based on character identity.

        voice_hint:
                    A casting note describing what this character's voice
                    should SOUND like — age, warmth, accent, vocal quality.
                    Used by the renderer's voice catalog to pick the best
                    TTS voice. E.g., "deep authoritative baritone, 50s,
                    British accent, world-weary." Set by the profile builder.

        tts_style:
                    A persistent TTS instruction prepended to every
                    utterance's per-line emotion hint. Keeps the character's
                    vocal identity stable across the whole show. E.g.,
                    "Speak with a gruff, world-weary gravitas." Engines
                    that support natural-language instructions (Kokoro,
                    OpenAI) will use this; others ignore it gracefully.
    """

    name: str = ""
    side: str | None = None
    role: str = "participant"
    behavioral_triggers: str | None = None
    emotional_arc: str | None = None
    key_positions: list[str] = field(default_factory=list)
    factual_claims: list[str] = field(default_factory=list)
    reference_corpus: list[str] = field(default_factory=list)
    tangent_topics: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# ShowHostConfig
# ─────────────────────────────────────────────────────────────────────────────
# The "voice of the show" — the narrator or emcee who wraps content in a
# narrative structure. Reused across debates, news shows, podcasts, etc.

@dataclass
class ShowHostConfig(VoiceProfile):
    """Configuration for a show's narrative host.

    The voice that bookends content — intros, transitions, outros.
    Distinct from in-content roles like moderators or anchors.

    Attributes:
        name:       The host's display name (e.g., "Narrator", "Marcus Webb").

        role:       What kind of host:
                    "narrator" — Third-person omniscient, documentary-style.
                    "emcee"    — First-person, warm, direct-to-audience.
                    (Default: "narrator")

        style:      Free-form description of their hosting style.

        voice_fingerprint:
                    HOW the host talks — same concept as ParticipantProfile.

        voice_gender:
                    The vocal gender for TTS voice assignment.
                    "male", "female", or None (auto-detect).

        voice_hint:
                    Casting note for TTS voice selection — what the host
                    should sound like. See ParticipantProfile.voice_hint.

        tts_style:
                    Persistent TTS instruction for this host's vocal
                    identity. See ParticipantProfile.tts_style.
    """

    name: str = ""
    role: str = "narrator"
    style: str | None = None
