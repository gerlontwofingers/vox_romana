"""vox_romana.planning — agents that decide WHAT to make next.

The planning package sits in front of the show pipeline.  Where the
``shows/`` package answers "given a brief, how do we make the show?",
``planning/`` answers "given a series, what should the next show be?".

Modules:
    episode_planner   Generates the brief for the next episode in a
                      series, using the writers'-bible and prior
                      episode summaries from the catalog.
"""

from vox_romana.planning.episode_planner import EpisodeBrief, EpisodePlanner

__all__ = ["EpisodeBrief", "EpisodePlanner"]
