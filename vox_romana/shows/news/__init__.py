"""vox_romana.shows.news — news broadcast generation.

Generates structured news shows from a plain-text brief. The output is a
sequence of AudioEvents organized into segments (headlines, deep dives,
interviews, closings, etc.), ready for TTS rendering.

The pipeline mirrors the debate pattern:

    BOOTSTRAP:
        bootstrap.py  → Reads the brief, plans the rundown, profiles anchors.

    CONDUCTOR:
        conductor.py  → Generates the broadcast segment by segment. Each anchor
                        and reporter is an LLM agent. The conductor manages the
                        show host, director check-ins, and context compression.

    POST-PRODUCTION:
        serializer.py → Saves to JSON.

USAGE:
    from vox_romana.shows.news import NewsBootstrap, NewsConductor
    from vox_romana.core.client import DeepSeekClient

    client = DeepSeekClient()
    config = NewsBootstrap(client).from_brief("Top 5 AI stories this week")
    result = NewsConductor(config, client=client).run()
"""

from .bootstrap import NewsBootstrap
from .conductor import NewsConductor, NewsConductorResult
from .config import (
    AnchorConfig,
    DirectorBrief,  # backwards-compat alias for NewsDirectorBrief
    NewsConfig,
    NewsDirectorBrief,
    SegmentConfig,
)
from .models import MergedNewsShow, NewsEvent, NewsSegment
from .prompts import (
    ANCHOR_SYSTEM_PROMPT,
    CONTEXT_SUMMARY_PROMPT,
    DIRECTOR_CHECK_PROMPT,
    SHOW_HOST_SYSTEM_PROMPT,
)
from .serializer import NewsSerializer

__all__ = [
    "AnchorConfig",
    "DirectorBrief",  # backwards-compat alias for NewsDirectorBrief
    "NewsDirectorBrief",
    "MergedNewsShow",
    "NewsBootstrap",
    "NewsConductor",
    "NewsConductorResult",
    "NewsConfig",
    "NewsEvent",
    "NewsSegment",
    "NewsSerializer",
    "SegmentConfig",
    # Prompts
    "ANCHOR_SYSTEM_PROMPT",
    "CONTEXT_SUMMARY_PROMPT",
    "DIRECTOR_CHECK_PROMPT",
    "SHOW_HOST_SYSTEM_PROMPT",
]
