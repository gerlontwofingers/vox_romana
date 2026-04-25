"""Read and write debate data as JSON.

Thin wrapper over :class:`vox_romana.core.serializer.BaseShowSerializer`
that adds the parsed/merged distinction unique to debates (debates have
a pre-merge ``ParsedDebateDocument`` step that other show types skip).
"""

from __future__ import annotations

from pathlib import Path

from ...core.serializer import BaseShowSerializer
from .models import MergedDebate, ParsedDebateDocument


class DebateSerializer(BaseShowSerializer[MergedDebate]):
    """Saves and loads debate objects as human-readable JSON."""

    def save_parsed(self, document: ParsedDebateDocument, path: str | Path) -> Path:
        """Write a parsed document (pre-merge) to a JSON file."""
        return self._write(document.to_dict(), path)

    def save_merged(self, debate: MergedDebate, path: str | Path) -> Path:
        """Write a merged debate to a JSON file. Alias of :meth:`save`."""
        return self.save(debate, path)

    def save_merged_many(self, debates: tuple[MergedDebate, ...], path: str | Path) -> Path:
        """Write multiple merged debates into one JSON file as a list."""
        return self.save_many(debates, path)
