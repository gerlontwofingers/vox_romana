"""vox_romana.core — shared infrastructure for all show types.

This package contains everything that's content-type agnostic: the LLM
client, agent wrappers, base configuration, search providers, the catalog,
the editing pipeline, and the notes archive. Show-specific code lives under
vox_romana.shows (debates, news, etc.).

Architecture at a glance:

    client.py        — DeepSeekClient: thin API wrapper (one per pipeline run).
    agent.py         — DeepSeekAgent: simple stateful agent (no compression).
    session.py       — AgentSession: production agent with sliding-window
                       compression. Used by all persistent broadcast agents.
    config.py        — ContentConfig base class + ParticipantProfile.
    conductor.py     — BaseConductor: abstract lifecycle for reactive shows.
    router.py        — ShowRouter: classifies briefs → show type.
    catalog.py       — Catalog: SQLite content log with FTS5 search.
    search.py        — BaseSearchProvider + GoogleSearchProvider.
    data_researcher.py — DataResearcher: grounded statistical analysis.
    models.py        — AudioEvent, AgentResponse, FactCard.
    "archive.py"       — NoteArchive + AgentNoteStore + OutlineArchive (JSONL).
    librarian.py     — Librarian: ephemeral search agent for the archive.
    editing/         — ChunkEditor, DirectorOutline, RetryLoop (quality gating).
    planner.py       — ContentPlanner (placeholder for episodic splitting).
    runsheet.py      — Runsheet entries for broadcast-style shows.

All public symbols are re-exported here for convenience:
    from vox_romana.core import DeepSeekClient, AgentSession, ChunkEditor
"""

from .agent import DeepSeekAgent
from .archive import AgentNote, AgentNoteStore, EditorNote, NoteArchive, OutlineArchive, OutlineRecord
from .catalog import Catalog, RunRecord, StudyRecord
from .client import DeepSeekClient
from .conductor import BaseConductor, ConductorResult
from .config import ContentConfig, FactCheckMode, ParticipantProfile, ShowHostConfig
from .data_researcher import DataResearcher, StudyResult
from .duration import (
    SpeechRateAdvisor,
    WPM_DEFAULTS,
    estimate_total_words,
    get_wpm,
    parse_duration_from_text,
    seconds_to_mmss,
    seconds_to_words,
    tolerance_window,
    words_to_seconds,
)
from .editing import (
    EDITOR_SYSTEM_PROMPT,
    ChunkEditor,
    ChunkResult,
    DirectorOutline,
    EditorVerdict,
    OutlineSection,
    OutlineSnapshot,
    RetryLoop,
)
from .json_retry import JsonParseError, json_retry, parse_json_response
from .librarian import Librarian, LibrarianResult
from .models import AgentResponse, AudioEvent, FactCard
from .router import RouterResult, ShowRouter, parse_router_response
from .search import BaseSearchProvider, GoogleSearchProvider, SearchResult
from .session import AgentSession

__all__ = [
    "AgentNote",
    "AgentNoteStore",
    "AgentResponse",
    "AgentSession",
    "AudioEvent",
    "BaseConductor",
    "BaseSearchProvider",
    "Catalog",
    "ChunkEditor",
    "ChunkResult",
    "ConductorResult",
    "ContentConfig",
    "DataResearcher",
    "DeepSeekAgent",
    "DeepSeekClient",
    "DirectorOutline",
    "EDITOR_SYSTEM_PROMPT",
    "EditorNote",
    "EditorVerdict",
    "FactCard",
    "FactCheckMode",
    "GoogleSearchProvider",
    "JsonParseError",
    "Librarian",
    "LibrarianResult",
    "NoteArchive",
    "OutlineArchive",
    "OutlineRecord",
    "OutlineSection",
    "OutlineSnapshot",
    "ParticipantProfile",
    "RetryLoop",
    "RouterResult",
    "RunRecord",
    "SearchResult",
    "ShowRouter",
    "ShowHostConfig",
    "StudyRecord",
    "StudyResult",
    "SpeechRateAdvisor",
    "WPM_DEFAULTS",
    "estimate_total_words",
    "get_wpm",
    "json_retry",
    "parse_duration_from_text",
    "parse_json_response",
    "parse_router_response",
    "seconds_to_mmss",
    "seconds_to_words",
    "tolerance_window",
    "words_to_seconds",
]
