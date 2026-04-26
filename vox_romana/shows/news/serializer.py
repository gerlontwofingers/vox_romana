"""News show serializer — thin wrapper over :class:`BaseShowSerializer`."""

from __future__ import annotations

from ...core.serializer import BaseShowSerializer
from .models import MergedNewsShow


class NewsSerializer(BaseShowSerializer[MergedNewsShow]):
    """Saves and loads news show objects as human-readable JSON."""
