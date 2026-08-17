"""Lecture serializer — thin wrapper over :class:`BaseShowSerializer`."""

from __future__ import annotations

from ...core.serializer import BaseShowSerializer
from .models import MergedLecture


class LectureSerializer(BaseShowSerializer[MergedLecture]):
    """Saves and loads MergedLecture objects as JSON files."""
