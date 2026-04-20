"""Podcast serializer — save/load MergedPodcast as JSON.

Same pattern as DebateSerializer, NewsSerializer, and AudiobookSerializer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import MergedPodcast


class PodcastSerializer:
    """Saves and loads MergedPodcast objects as JSON files.

    USAGE:
        serializer = PodcastSerializer()

        # Save:
        serializer.save(merged_podcast, "output/text/podcasts/episode.json")

        # Load (returns raw dict):
        data = serializer.load("output/text/podcasts/episode.json")

        # To JSON string:
        json_str = serializer.to_json_string(merged_podcast)
    """

    def save(self, podcast: MergedPodcast, path: str | Path) -> Path:
        """Save a MergedPodcast to a JSON file.

        Args:
            podcast: The MergedPodcast to save.
            path:    File path to write to. Parent dirs are created if needed.

        Returns:
            The resolved Path of the saved file.
        """
        return self._write(podcast.to_dict(), path)

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a JSON file and return the raw dict.

        Args:
            path: File path to read from.

        Returns:
            The parsed JSON as a dict.
        """
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def to_json_string(self, podcast: MergedPodcast) -> str:
        """Convert a MergedPodcast to a JSON string.

        Args:
            podcast: The MergedPodcast to convert.

        Returns:
            A JSON string with 2-space indentation.
        """
        return json.dumps(podcast.to_dict(), indent=2, ensure_ascii=False)

    def _write(self, data: Any, path: str | Path) -> Path:
        """Write JSON data to a file, creating parent dirs if needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return p
