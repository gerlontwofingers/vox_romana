"""vox_romana.shows.debates — multi-participant debate generation.

This is the first and most mature show type. It generates structured debates
from a plain-text brief, producing AudioEvents ready for TTS rendering.

The pipeline flows through three phases:

    BOOTSTRAP (pre-production):
        architect.py       → Reads the brief, decides cast size, format, tone.
        profile_builder.py → Builds full personality profiles for each participant.
        director.py        → Plans the dramatic arc (rising tension, pivots, close).
        researcher.py      → Optionally grounds the debate in web-sourced facts.

    CONDUCTOR (live recording):
        conductor.py       → Runs the multi-agent turn loop. Each participant,
                             moderator, and director is an LLM agent. The conductor
                             manages turn order, context compression, and pacing.

    POST-PRODUCTION:
        parsers.py         → Extracts structured DebateEvents from raw LLM output.
        merge.py           → Combines chunks into a single MergedDebate document.
        validator.py       → Checks the output for structural issues.
        serializer.py      → Saves to JSON.
        pipeline.py        → Convenience wrapper that chains parse → merge → validate.

USAGE:
    from vox_romana.shows.debates import DebateBootstrap, DebateConductor
    from vox_romana.core.client import DeepSeekClient

    client = DeepSeekClient()
    config = DebateBootstrap(client).from_brief("Two economists debate UBI")
    result = DebateConductor(config, client=client).run()
"""

from .architect import DebateArchitect
from .bootstrap import DebateBootstrap
from .conductor import DebateConductor, DebateConductorResult
from .config import (
    AudienceConfig,
    DebateConfig,
    DebateDirectorBrief,
    DirectorBrief,  # backwards-compat alias for DebateDirectorBrief
    ModeratorConfig,
    ParticipantProfile,
    ShowHostConfig,
)
from .director import DebateDirector
from .merge import DebateDocumentMerger, DebateMergeError
from .models import DebateChunk, DebateEvent, DebateSegment, MergedDebate, ParsedDebateDocument
from .parsers import (
    DebateParserError,
    DebateParserRegistry,
    DebateTemplate01Parser,
    DebateTemplate02Parser,
)
from .pipeline import DebatePipeline, PipelineResult
from .profile_builder import DebateProfileBuilder
from .researcher import DebateResearcher
from .serializer import DebateSerializer
from .validator import DebateValidator, ValidationReport

__all__ = [
    "AudienceConfig",
    "DebateArchitect",
    "DebateBootstrap",
    "DebateConductor",
    "DebateConductorResult",
    "DebateChunk",  # backwards-compat alias for DebateSegment
    "DebateSegment",
    "DebateConfig",
    "DebateDirector",
    "DebateDocumentMerger",
    "DebateEvent",
    "DebateMergeError",
    "DebateParserError",
    "DebateParserRegistry",
    "DebatePipeline",
    "DebateProfileBuilder",
    "DebateResearcher",
    "DebateSerializer",
    "DebateTemplate01Parser",
    "DebateTemplate02Parser",
    "DebateValidator",
    "DebateDirectorBrief",
    "DirectorBrief",  # backwards-compat alias for DebateDirectorBrief
    "MergedDebate",
    "ModeratorConfig",
    "ParsedDebateDocument",
    "ParticipantProfile",
    "PipelineResult",
    "ShowHostConfig",
    "ValidationReport",
]
