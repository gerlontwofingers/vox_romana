"""Runsheet — the episode schedule for broadcast-style content.

A runsheet is a producer's script: an ordered list of entries that define
what happens in a show and when. News shows, radio programs, and talk shows
use runsheets. Simpler content types (debates, audiobooks) don't need one —
the runsheet field on ContentConfig defaults to None.

USAGE:
    from vox_romana.core.runsheet import Runsheet, RunsheetEntry, SegmentEntry, BreakEntry

    runsheet = Runsheet(entries=[
        SegmentEntry(segment_type="cold_open", topic="Hook", target_duration_seconds=20),
        SegmentEntry(segment_type="headlines", topic="Top stories", target_duration_seconds=90),
        BreakEntry(break_type="commercial", duration_seconds=30),
        SegmentEntry(segment_type="deep_dive", topic="AI regulation", target_duration_seconds=300),
        SegmentEntry(segment_type="signoff", topic="Wrap", target_duration_seconds=30),
    ])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# Entry types — the building blocks of a runsheet
# ─────────────────────────────────────────────────────────────────────────────

EntryType = Literal["segment", "break", "bumper", "transition"]


@dataclass
class RunsheetEntry:
    """Base class for any entry in a runsheet.

    Attributes:
        entry_type:   What kind of entry ("segment", "break", "bumper", "transition").
        label:        Human-readable label for logging/display.
        preemptable:  Can this entry be bumped by a higher-priority event
                      (e.g., breaking news)? Default False.
    """

    entry_type: EntryType
    label: str = ""
    preemptable: bool = False


@dataclass
class SegmentEntry(RunsheetEntry):
    """A content segment — the LLM-generated parts of the show.

    Attributes:
        segment_type:              What kind of segment (e.g., "cold_open",
                                   "headlines", "deep_dive", "panel", "signoff").
        topic:                     What this segment covers.
        participants:              Names from the cast pool (resolved at runtime).
        target_duration_seconds:   Rough length guide.
        force_mode:                "scripted" or "reactive" to override the
                                   default for this segment type. None = auto.
        director_notes:            Segment-specific pacing/mood hints.
        safety_notes:              Extra guardrails for satirical content.
    """

    entry_type: EntryType = "segment"
    segment_type: str = ""
    topic: str = ""
    participants: list[str] = field(default_factory=list)
    target_duration_seconds: int = 120
    force_mode: str | None = None
    director_notes: str | None = None
    safety_notes: str | None = None


@dataclass
class BreakEntry(RunsheetEntry):
    """A non-content break — commercial, station ID, sponsor read.

    The conductor doesn't generate LLM content for breaks. The audio
    mixer splices in pre-recorded assets or inserts silence.

    Attributes:
        break_type:       "commercial", "station_id", "sponsor_read", "silence".
        duration_seconds: How long the break lasts in the final audio.
        asset_path:       Path to a pre-recorded audio file (optional).
    """

    entry_type: EntryType = "break"
    break_type: str = "silence"
    duration_seconds: int = 30
    asset_path: str | None = None


@dataclass
class BumperEntry(RunsheetEntry):
    """A short audio cue — stinger, whoosh, transition sound.

    Attributes:
        bumper_type:  "stinger", "whoosh", "theme", "news_alert", etc.
        asset_path:   Path to the audio file.
    """

    entry_type: EntryType = "bumper"
    bumper_type: str = "stinger"
    asset_path: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# AudioAssetLibrary — paths to reusable audio assets
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioAssetLibrary:
    """Paths to audio assets used by the mixer during rendering.

    All fields are optional. No assets = clean audio, no music.

    Attributes:
        theme_music:           Show open/close music file.
        bumper_stinger:        Short "we'll be right back" sound.
        transition_bed:        Music under segment transitions.
        breaking_news_sting:   Dramatic alert sound.
        custom_assets:         Named extras (e.g., {"applause": "path/to/applause.mp3"}).
    """

    theme_music: str | None = None
    bumper_stinger: str | None = None
    transition_bed: str | None = None
    breaking_news_sting: str | None = None
    custom_assets: dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Runsheet — the full episode schedule
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Runsheet:
    """An ordered schedule for a broadcast-style episode.

    The conductor iterates entries in order. For each segment entry,
    it dispatches to scripted or reactive generation. For breaks and
    bumpers, it emits marker events that the audio mixer handles.

    Attributes:
        entries:                Ordered list of runsheet entries.
        total_runtime_target:   Target episode length in seconds (advisory).
        break_pattern:          Pattern hint for auto-inserting breaks.
                                "none", "every_8_min", "after_headlines", etc.
    """

    entries: list[RunsheetEntry] = field(default_factory=list)
    total_runtime_target: int | None = None
    break_pattern: str = "none"
