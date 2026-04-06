"""Read and write debate data as JSON.

This is the bridge between your internal Python objects and the filesystem.
Parsed or merged debates go in as objects, come out as clean JSON files.
JSON files come in as paths, come out as dicts you can inspect or reload.

The JSON is always pretty-printed so you can actually read it. Performance
is not a concern here — these files are small.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import MergedDebate, ParsedDebateDocument


class DebateSerializer:
    """Saves and loads debate objects as human-readable JSON."""

    # ── Writing ────────────────────────────────────────────────────────────

    def save_parsed(self, document: ParsedDebateDocument, path: str | Path) -> Path:
        """Write a parsed document (pre-merge) to a JSON file."""
        return self._write(document.to_dict(), path)

    def save_merged(self, debate: MergedDebate, path: str | Path) -> Path:
        """Write a merged debate to a JSON file."""
        return self._write(debate.to_dict(), path)

    def save_merged_many(self, debates: tuple[MergedDebate, ...], path: str | Path) -> Path:
        """Write multiple merged debates into one JSON file as a list."""
        return self._write([d.to_dict() for d in debates], path)

    # ── Reading ────────────────────────────────────────────────────────────

    def load(self, path: str | Path) -> dict[str, Any] | list[dict[str, Any]]:
        """Load a previously saved JSON debate file back into a plain dict.

        This gives you the raw dict, not a reconstituted dataclass. Good for
        inspection, downstream pipelines, or feeding into speech-to-text code
        that just needs the JSON structure.
        """
        filepath = Path(path)
        with filepath.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # ── Convenience ────────────────────────────────────────────────────────

    def to_json_string(self, obj: ParsedDebateDocument | MergedDebate) -> str:
        """Return a pretty-printed JSON string without writing to disk."""
        return json.dumps(obj.to_dict(), indent=2, ensure_ascii=False)

    # ── Internals ──────────────────────────────────────────────────────────

    def _write(self, data: Any, path: str | Path) -> Path:
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")  # trailing newline keeps git happy
        return filepath
