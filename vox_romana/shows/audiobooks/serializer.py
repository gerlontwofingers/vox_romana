"""Audiobook serializer \u2014 thin wrapper over :class:`BaseShowSerializer`."""

from __future__ import annotations

from ...core.serializer import BaseShowSerializer
from .models import MergedAudiobook


class AudiobookSerializer(BaseShowSerializer[MergedAudiobook]):
    """Saves and loads MergedAudiobook objects as JSON files."""
