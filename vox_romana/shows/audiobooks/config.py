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
                └── DirectorBrief   — pacing/mood arc
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.config import ContentConfig, ParticipantProfile, ShowHostConfig


@dataclass
class NarratorConfig:
    """The primary narrator voice for the audiobook."""

    name: str = ""
    style: str | None = None  # e.g., "Warm, measured, literary"
    voice_fingerprint: str | None = None
    voice_gender: str | None = None
    voice_hint: str | None = None  # TTS casting note
    tts_style: str | None = None


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
class DirectorBrief:
    """Pacing and arc plan for the audiobook."""

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
    director_brief: DirectorBrief | None = None

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
