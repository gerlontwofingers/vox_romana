"""ContentPlanner — episodic chunking for long-form content.

This is a PLACEHOLDER module. The ContentPlanner will be implemented when
we have long-form content types (audiobooks) that need splitting into
episodes. For now, this defines the interfaces so other code can reference
them without importing unfinished implementations.

The ContentPlanner sits between the VoxRouter and the format-specific
Bootstrap. It examines the content and decides whether to split it into
multiple episodes:

    VoxRouter → ContentPlanner → Bootstrap → Conductor → Renderer

For short content, the planner is a pass-through. For long content
(e.g., a 500k-word novel), it produces a list of EpisodeSpec objects,
each fed independently into the format's bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeSpec:
    """Specification for one episode of a multi-episode series.

    The ContentPlanner produces these. Each one feeds into a format-specific
    Bootstrap.from_brief() independently.

    Attributes:
        episode_number:         1-indexed episode number.
        series_id:              Groups episodes together.
        series_title:           Human-readable series name.
        episode_brief:          Episode-specific brief for the bootstrap.
        source_range:           For text-based content: which slice of source material.
        continuation_context:   Summary of previous episodes (for LLM context).
    """

    episode_number: int = 1
    series_id: str = ""
    series_title: str = ""
    episode_brief: str = ""
    source_range: str | None = None
    continuation_context: str | None = None
