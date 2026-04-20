"""Read and write news show data as JSON.

Same pattern as DebateSerializer — bridges Python objects and the filesystem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import MergedNewsShow


class NewsSerializer:
    """Saves and loads news show objects as human-readable JSON."""

    def save(self, show: MergedNewsShow, path: str | Path) -> Path:
        """Write a merged news show to a JSON file."""
        return self._write(show.to_dict(), path)

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a previously saved JSON news show file."""
        filepath = Path(path)
        with filepath.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def to_json_string(self, show: MergedNewsShow) -> str:
        """Return a pretty-printed JSON string without writing to disk."""
        return json.dumps(show.to_dict(), indent=2, ensure_ascii=False)

    def _write(self, data: Any, path: str | Path) -> Path:
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        return filepath
