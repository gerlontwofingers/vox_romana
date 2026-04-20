"""Podcast show type — multi-host conversational content generation.

A podcast is a REACTIVE show type where multiple hosts discuss topics
across structured segments. Unlike debates (adversarial, turn-by-turn) or
audiobooks (scripted narration), podcasts generate an entire segment per
LLM call — producing natural multi-speaker banter in one shot.

Key architectural difference from debates:
    - Debates: one LLM call per speaker turn (reactive, sequential)
    - Podcasts: one LLM call per segment (multi-speaker, segment-level)

This is more cost-efficient and produces more natural conversation flow.

Module structure (mirrors audiobooks/debates):
    config.py      — PodcastHostProfile, PodcastSegmentConfig, PodcastConfig
    models.py      — PodcastEvent (alias), PodcastSegment, MergedPodcast
    prompts.py     — System prompts and segment prompt templates
    bootstrap.py   — PodcastBootstrap (Architect → Host Profiler → Director)
    conductor.py   — PodcastConductor (segment-level generation loop)
    serializer.py  — PodcastSerializer (save/load JSON)

USAGE:
    # One-shot from brief:
    conductor = PodcastConductor.from_brief(
        "Three women discuss the greatest rivalries in women's soccer",
        target_words=5000,
    )
    result = conductor.run()
    merged = result.to_merged_podcast()
    PodcastSerializer.save(merged, "output/text/podcasts/soccer_rivalries.json")

    # From existing config:
    config = PodcastBootstrap().from_brief("...")
    conductor = PodcastConductor(config)
    result = conductor.run()
"""

from .config import (
    PodcastConfig,
    PodcastDirectorBrief,
    PodcastHostProfile,
    PodcastSegmentConfig,
)
from .models import MergedPodcast, PodcastEvent, PodcastSegment

__all__ = [
    "PodcastConfig",
    "PodcastDirectorBrief",
    "PodcastHostProfile",
    "PodcastSegmentConfig",
    "PodcastEvent",
    "PodcastSegment",
    "MergedPodcast",
]
