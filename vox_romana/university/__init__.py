"""vox_romana.university — local deep research system.

The University is a local deep research subsystem powered by a local LLM
(via Ollama). A ``Scholar`` agent runs a time-budgeted, interruptable
read/synthesize loop over knowledge sources (Wikipedia + local memory) and
builds structured notes plus a running synthesis. Sessions persist forever
and auto-join the existing memory system.

Public API (built incrementally across chunks):

    from vox_romana.university import (
        # Chunk B (sources):
        Article, KnowledgeSource, WikipediaSource, LocalSource,
        # Chunk C (session + config):
        ScholarNote, ScholarSession, UniversityConfig,
        # Chunk D (agent + orchestrator stub):
        Scholar, University,
        # Chunk F (search provider):
        UniversitySearchProvider,
    )
"""

from vox_romana.university.config import UniversityConfig
from vox_romana.university.scholar import Scholar
from vox_romana.university.seminar_plan import SeminarPlan
from vox_romana.university.session import ScholarNote, ScholarSession
from vox_romana.university.sources import (
    Article,
    KnowledgeSource,
    LocalSource,
    WikipediaSource,
)
from vox_romana.university.search_provider import UniversitySearchProvider
from vox_romana.university.university import University

__all__ = [
    "Article",
    "KnowledgeSource",
    "LocalSource",
    "Scholar",
    "ScholarNote",
    "ScholarSession",
    "SeminarPlan",
    "UniversitySearchProvider",
    "University",
    "UniversityConfig",
    "WikipediaSource",
]
