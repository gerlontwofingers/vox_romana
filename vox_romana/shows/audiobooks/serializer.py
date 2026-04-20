"""Audiobook serializer — save/load MergedAudiobook as JSON.

Same pattern as DebateSerializer and NewsSerializer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import MergedAudiobook


class AudiobookSerializer:
    """Saves and loads MergedAudiobook objects as JSON files."""

    def save(self, audiobook: MergedAudiobook, path: str | Path) -> Path:
        """Save a MergedAudiobook to a JSON file."""
        return self._write(audiobook.to_dict(), path)

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a JSON file and return the raw dict."""
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def to_json_string(self, audiobook: MergedAudiobook) -> str:
        """Convert a MergedAudiobook to a JSON string."""
        return json.dumps(audiobook.to_dict(), indent=2, ensure_ascii=False)

    def _write(self, data: Any, path: str | Path) -> Path:
        """Write JSON data to a file, creating parent dirs if needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return p
