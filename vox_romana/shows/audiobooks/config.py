"""Audiobook configuration models.

An audiobook is a SCRIPTED show type — a single narrator (or small cast)
reads through a structured text, chapter by chapter. Unlike debates (reactive,
multi-agent) or news (segment-based, anchor-driven), audiobooks are
outline-driven narration with optional dialogue scenes.

Config hierarchy:
    ContentConfig (core)
        └── AudiobookConfig
                ├── NarratorConfig  — the primary voice
                ├── ChapterConfig[] — ordered chapter outline
                └── AudiobookDirectorBrief   — pacing/mood arc
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.config import ContentConfig, ParticipantProfile, ShowHostConfig, VoiceProfile


@dataclass
class NarratorConfig(VoiceProfile):
    """The primary narrator voice for the audiobook."""

    name: str = ""
    style: str | None = None  # e.g., "Warm, measured, literary"


@dataclass
class ChapterConfig:
    """Configuration for a single audiobook chapter."""

    title: str = ""
    synopsis: str | None = None  # What happens in this chapter
    key_points: list[str] = field(default_factory=list)  # Ordered narrative beats
    tone: str | None = None  # Chapter-specific tone override
    duration_hint: str = "medium"  # "short", "medium", "long"
    target_beats: int = 1  # How many generation beats (like news segments)
    characters: list[str] = field(default_factory=list)  # Named characters in this chapter


@dataclass
class AudiobookDirectorBrief:
    """Pacing and arc plan for the audiobook.

    Attributes:
        narrative_arc:     High-level narrative structure describing how the
                           story should feel from opening chapter to close.
        mood_progression:  Ordered list of moods the narration should move
                           through (e.g. ["anticipatory", "tense", "resolved"]).
        chapter_notes:     Director notes for specific chapters or transition
                           points. Free text.
        closing_strategy:  How to close the final chapter — e.g. "Leave on
                           an ambiguous note" or "Full resolution".
        check_interval:    How many *chapters* between Director check-ins.
                           At each check-in the Director reads the narration
                           so far and injects a mood/pacing note for the next
                           chapter. Lower = tighter arc control, higher =
                           more organic flow. Default: every 3 chapters.
    """

    narrative_arc: str = ""
    mood_progression: list[str] = field(default_factory=list)
    chapter_notes: str | None = None
    closing_strategy: str | None = None
    check_interval: int = 3  # Director check-in every N chapters


@dataclass
class AudiobookConfig(ContentConfig):
    """Full configuration for audiobook generation.

    Extends ContentConfig with audiobook-specific fields: narrator,
    chapters, genre, and character cast.
    """

    content_type: str = "audiobook"
    flexibility_score: float = 0.2
    topic: str | None = None  # Subject/theme
    genre: str | None = None  # e.g., "science fiction", "history", "self-help"
    tone: str | None = None  # Overall tone
    setting: str | None = None  # World/setting description
    narrator: NarratorConfig | None = None
    chapters: list[ChapterConfig] = field(default_factory=list)
    characters: list[ParticipantProfile] = field(default_factory=list)  # Cast for dialogue
    director_brief: AudiobookDirectorBrief | None = None
    language: str = "en"
    music_style: str = ""

    def validate(self) -> list[str]:
        """Return a list of validation problems (empty = valid)."""
        problems = super().validate()
        if not self.chapters:
            problems.append("AudiobookConfig requires at least one chapter.")
        if not self.narrator:
            problems.append("AudiobookConfig requires a narrator.")
        elif not self.narrator.name:
            problems.append("Narrator must have a name.")
        return problems


# Backwards-compatibility alias. ``DirectorBrief`` was renamed to
# ``AudiobookDirectorBrief`` to disambiguate the three former DirectorBrief
# classes that lived in debates/news/audiobooks (each with different fields).
# External callers that imported ``DirectorBrief`` continue to work via this
# alias; new code should use the explicit ``AudiobookDirectorBrief``.
DirectorBrief = AudiobookDirectorBrief
