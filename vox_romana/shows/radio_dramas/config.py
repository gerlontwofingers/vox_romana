"""Radio drama configuration models.

A radio drama is a SCRIPTED show type — a multi-voice narrative fiction
episode in the tradition of 1930s–50s "golden age of radio" serials
(Sam Spade, Lights Out, Suspense, X Minus One). Unlike debates (reactive,
turn-by-turn) or podcasts (segment-level conversation), radio dramas are
scene-by-scene scripted fiction with a cold-open announcer, era-styled
music cues, and a cast of characters.

Structurally this is the closest cousin to the audiobook package: the
AudiobookEvent / AudioEvent shape carries exactly the per-line delivery
fields a script needs (tone, pace, pause_before, pause_after, nonverbal).
Radio drama differs in having a MULTI-VOICE cast and a scene loop instead
of a single narrator's chapter loop.

Config hierarchy:
    ContentConfig (core)
        └── RadioDramaConfig
                ├── CharacterConfig[]    — the voiced cast (incl. announcer)
                ├── SceneConfig[]        — ordered scene outline
                └── RadioDramaDirectorBrief — pacing/arc plan
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.config import ContentConfig, VoiceProfile

# Valid era-dial settings. The era knob maps to announcer register,
# music-style seed, and vocabulary — see prompts.py (the data-driven
# ERA table built in shot 5.1-b).
ERAS = ("1940s", "1970s", "present", "future")

# Pilot vs serial planner modes. "pilot" = self-contained single episode;
# "serial" = episode with a series_id, recurring cast, and a cliffhanger.
SERIES_MODES = ("pilot", "serial")


@dataclass
class CharacterConfig(VoiceProfile):
    """One voiced character (or the announcer) in a radio drama.

    Voice fields (``voice_gender``, ``voice_hint``, ``tts_style``,
    ``voice_fingerprint``) are inherited from :class:`VoiceProfile`.

    NOTE (5.1-a): the ``genre`` field on RadioDramaConfig is a SINGLE knob
    for now — per-genre character-archetype overlays (noir, sci-fi, horror,
    sitcom, western) are future work, not built here.
    """

    name: str = ""
    role: str = "character"  # "announcer", "protagonist", "detective", ...
    archetype: str | None = None  # e.g. "hardboiled detective"
    description: str | None = None  # Who they are in the story
    speech_style: str | None = None  # How they talk
    personality_triggers: str | None = None  # How they react


@dataclass
class SceneConfig:
    """Configuration for a single scene in the episode outline."""

    title: str = ""
    synopsis: str | None = None  # What happens in this scene
    location: str | None = None  # Where it happens (audio-only setting cue)
    characters: list[str] = field(default_factory=list)  # Names present
    duration_hint: str = "medium"  # "short", "medium", "long"
    target_words: int | None = None
    music_hint: str | None = None  # e.g. "dramatic 1940s orchestral sting"
    # NOTE (5.1-a): SFX is DEFERRED for the MVP. sfx_hints holds the cues
    # we'll emit as [sfx: ...] bracket text (which the TTS renderer already
    # strips) — no discrete SFX audio synthesis is built yet.
    sfx_hints: list[str] = field(default_factory=list)
    cliffhanger: bool = False  # Scene ends on a hook (serial episodes)


@dataclass
class RadioDramaDirectorBrief:
    """Pacing and arc plan for the radio drama.

    Attributes:
        narrative_arc:     High-level arc from cold open to sign-off.
        mood_progression:  Ordered moods across scenes
                           (e.g. ["tense", "foreboding", "resolved"]).
        scene_notes:       Director notes keyed by scene index.
        closing_strategy:  How to close the final scene — e.g. "full
                           resolution" (pilot) or "cliffhanger hook"
                           (serial).
        check_interval:    How many scenes between Director check-ins.
                           Default: every 3 scenes.
    """

    narrative_arc: str = ""
    mood_progression: list[str] = field(default_factory=list)
    scene_notes: dict[int, str] = field(default_factory=dict)
    closing_strategy: str | None = None
    check_interval: int = 3


@dataclass
class RadioDramaConfig(ContentConfig):
    """Full configuration for radio drama generation.

    Extends ContentConfig with drama-specific fields: genre, era, cast,
    scenes, and the pilot-vs-serial planner fields.
    """

    content_type: str = "radio_drama"
    flexibility_score: float = 0.2  # Scripted → rigid length control (like audiobooks)
    genre: str | None = None  # SINGLE knob for now (see module docstring note)
    era: str = "1940s"  # Era dial: 1940s | 1970s | present | future
    setting: str | None = None  # World/setting description
    tone: str | None = None  # Overall tone
    announcer: CharacterConfig | None = None  # Cold-open / sign-off voice
    characters: list[CharacterConfig] = field(default_factory=list)  # The cast
    scenes: list[SceneConfig] = field(default_factory=list)
    director_brief: RadioDramaDirectorBrief | None = None
    language: str = "en"
    music_style: str = ""

    # ── Pilot vs serial planner ──────────────────────────────────────
    # Filled in by the bootstrap planner (shot 5.1-c). "pilot" produces a
    # self-contained episode; "serial" adds series continuity fields.
    series_mode: str = "pilot"  # "pilot" | "serial"
    series_id: str | None = None
    episode_number: int | None = None
    recurring_cast: list[str] = field(default_factory=list)  # Names carried across episodes

    def validate(self) -> list[str]:
        """Return a list of validation problems (empty = valid)."""
        problems = super().validate()
        if not self.characters and not self.announcer:
            problems.append(
                "RadioDramaConfig requires at least one character or an announcer."
            )
        if not self.scenes:
            problems.append("RadioDramaConfig requires at least one scene.")
        if self.era not in ERAS:
            problems.append(
                f"era must be one of {ERAS}, got {self.era!r}."
            )
        if self.series_mode not in SERIES_MODES:
            problems.append(
                f"series_mode must be one of {SERIES_MODES}, "
                f"got {self.series_mode!r}."
            )
        if self.series_mode == "serial" and not self.series_id:
            problems.append(
                "series_mode='serial' requires a series_id."
            )
        return problems
