"""vox_romana — LLM-driven audio content generation.

This package generates scripted audio content (debates, news shows, audiobooks,
podcasts) using LLM agents, then feeds the structured output into text-to-speech.

Architecture:
    core/       — Generic LLM client, agent, base models, and shared config.
    shows/      — Content-type implementations (debates, news, audiobooks, podcasts).
    tts/        — Text-to-speech engines, mixer, and renderer.

Quick start (debates):
    from vox_romana import DebatePipeline

    pipeline = DebatePipeline()
    result = pipeline.run([raw_llm_chunk_1, raw_llm_chunk_2])
    if result.ok:
        pipeline.save(result.merged[0], "output/debate.json")

Core symbols are re-exported here. For show-specific symbols, import from
the show's subpackage (e.g., vox_romana.shows.debates).
"""

# ── Core (shared across all content types) ─────────────────────────────────
from .core import (
    AgentResponse,
    AudioEvent,
    BaseConductor,
    BaseSearchProvider,
    Catalog,
    ConductorResult,
    ContentConfig,
    DataResearcher,
    DeepSeekAgent,
    DeepSeekClient,
    FactCard,
    FactCheckMode,
    GoogleSearchProvider,
    ParticipantProfile,
    RouterResult,
    RunRecord,
    SearchResult,
    ShowRouter,
    ShowHostConfig,
    StudyRecord,
    StudyResult,
    parse_router_response,
)

# ── Debates (the first and most mature show type) ─────────────────────────
from .shows.debates import (
    AudienceConfig,
    DebateArchitect,
    DebateBootstrap,
    DebateChunk,
    DebateConductor,
    DebateConductorResult,
    DebateConfig,
    DebateDirector,
    DebateDocumentMerger,
    DebateEvent,
    DebateMergeError,
    DebateParserError,
    DebateParserRegistry,
    DebatePipeline,
    DebateProfileBuilder,
    DebateResearcher,
    DebateDirectorBrief,
    DebateSegment,
    DebateSerializer,
    DebateTemplate01Parser,
    DebateTemplate02Parser,
    DebateValidator,
    DirectorBrief,  # backwards-compat alias for DebateDirectorBrief
    MergedDebate,
    ModeratorConfig,
    ParsedDebateDocument,
    PipelineResult,
    ValidationReport,
)

# ── News (the second show type) ───────────────────────────────────────────
from .shows.news import (
    AnchorConfig,
    MergedNewsShow,
    NewsBootstrap,
    NewsConductor,
    NewsConductorResult,
    NewsConfig,
    NewsDirectorBrief,
    NewsEvent,
    NewsSegment,
    NewsSerializer,
    SegmentConfig,
)

# ── Audiobooks (the third show type) ──────────────────────────────────────
from .shows.audiobooks import (
    AudiobookBootstrap,
    AudiobookChapter,
    AudiobookConductor,
    AudiobookConductorResult,
    AudiobookConfig,
    AudiobookDirectorBrief,
    AudiobookEvent,
    AudiobookSerializer,
    MergedAudiobook,
    NarratorConfig,
)
from .shows.audiobooks.config import ChapterConfig as AudiobookChapterConfig

# ── Podcasts (the fourth show type) ───────────────────────────────────────
from .shows.podcast import (
    MergedPodcast,
    PodcastConfig,
    PodcastDirectorBrief,
    PodcastEvent,
    PodcastHostProfile,
    PodcastSegment,
    PodcastSegmentConfig,
)
from .shows.podcast.bootstrap import PodcastBootstrap
from .shows.podcast.conductor import PodcastConductor, PodcastConductorResult
from .shows.podcast.serializer import PodcastSerializer

__all__ = [
    # Core
    "AgentResponse",
    "AudioEvent",
    "BaseConductor",
    "BaseSearchProvider",
    "Catalog",
    "ConductorResult",
    "ContentConfig",
    "DataResearcher",
    "DeepSeekAgent",
    "DeepSeekClient",
    "FactCard",
    "FactCheckMode",
    "GoogleSearchProvider",
    "ParticipantProfile",
    "RouterResult",
    "RunRecord",
    "SearchResult",
    "ShowRouter",
    "ShowHostConfig",
    "parse_router_response",
    "StudyRecord",
    "StudyResult",
    # Debates
    "AudienceConfig",
    "DebateArchitect",
    "DebateBootstrap",
    "DebateChunk",  # backwards-compat alias for DebateSegment
    "DebateSegment",
    "DebateConductor",
    "DebateConductorResult",
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
    "PipelineResult",
    "ValidationReport",
    # News
    "AnchorConfig",
    "MergedNewsShow",
    "NewsBootstrap",
    "NewsConductor",
    "NewsConductorResult",
    "NewsConfig",
    "NewsDirectorBrief",
    "NewsEvent",
    "NewsSegment",
    "NewsSerializer",
    "SegmentConfig",
    # Audiobooks
    "AudiobookBootstrap",
    "AudiobookChapter",
    "AudiobookChapterConfig",
    "AudiobookConductor",
    "AudiobookConductorResult",
    "AudiobookConfig",
    "AudiobookDirectorBrief",
    "AudiobookEvent",
    "AudiobookSerializer",
    "MergedAudiobook",
    "NarratorConfig",
    # Podcasts
    "MergedPodcast",
    "PodcastBootstrap",
    "PodcastConductor",
    "PodcastConductorResult",
    "PodcastConfig",
    "PodcastDirectorBrief",
    "PodcastEvent",
    "PodcastHostProfile",
    "PodcastSegment",
    "PodcastSegmentConfig",
    "PodcastSerializer",
]
