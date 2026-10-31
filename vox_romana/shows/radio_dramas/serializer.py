"""Radio drama serializer — thin wrapper over :class:`BaseShowSerializer`."""

from __future__ import annotations

from ...core.serializer import BaseShowSerializer
from .models import MergedRadioDrama


class RadioDramaSerializer(BaseShowSerializer[MergedRadioDrama]):
    """Saves and loads MergedRadioDrama objects as JSON files."""
