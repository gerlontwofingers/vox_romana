"""Podcast serializer \u2014 thin wrapper over :class:`BaseShowSerializer`."""

from __future__ import annotations

from ...core.serializer import BaseShowSerializer
from .models import MergedPodcast


class PodcastSerializer(BaseShowSerializer[MergedPodcast]):
    """Saves and loads MergedPodcast objects as JSON files."""
