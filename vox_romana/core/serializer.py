"""Generic JSON serializer base for show-level output models.

Every show type (debates, news, podcasts, audiobooks) ends up writing a
"merged" output object to JSON for downstream tools and the renderer.
The pattern is identical:

    1. Call obj.to_dict()
    2. json.dump it pretty-printed, UTF-8, with a trailing newline so git
       doesn't grumble.
    3. Read back as a raw dict (no reconstitution).

``BaseShowSerializer`` captures that pattern once. Show-specific
serializers become thin subclasses that exist only for naming/typing.

USAGE (subclass):

    from vox_romana.core.serializer import BaseShowSerializer
    from .models import MergedFoo

    class FooSerializer(BaseShowSerializer[MergedFoo]):
        pass

USAGE (call):

    s = FooSerializer()
    s.save(merged, "out/foo.json")
    data = s.load("out/foo.json")
    text = s.to_json_string(merged)

The base methods accept any object exposing a ``to_dict()`` method, so
domain models stay decoupled from the serializer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generic, Iterable, Protocol, TypeVar


class _SupportsToDict(Protocol):
    def to_dict(self) -> Any: ...  # noqa: D401, E704


T = TypeVar("T", bound=_SupportsToDict)


class BaseShowSerializer(Generic[T]):
    """Generic JSON serializer for any show-output model with ``to_dict()``.

    Subclasses get ``save``, ``load``, ``to_json_string``, and
    ``save_many`` for free. The generic type parameter ``T`` is purely
    for static-typing clarity — no runtime check is performed.
    """

    # ── Writing ────────────────────────────────────────────────────────

    def save(self, obj: T, path: str | Path) -> Path:
        """Write a single show object to a pretty-printed JSON file."""
        return self._write(obj.to_dict(), path)

    def save_many(self, objs: Iterable[T], path: str | Path) -> Path:
        """Write multiple show objects to one JSON file as an array."""
        return self._write([o.to_dict() for o in objs], path)

    # ── Reading ────────────────────────────────────────────────────────

    def load(self, path: str | Path) -> dict[str, Any] | list[dict[str, Any]]:
        """Load a previously saved JSON file as a raw dict (or list)."""
        filepath = Path(path)
        with filepath.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # ── Convenience ────────────────────────────────────────────────────

    def to_json_string(self, obj: T) -> str:
        """Return a pretty-printed JSON string without writing to disk."""
        return json.dumps(obj.to_dict(), indent=2, ensure_ascii=False)

    # ── Internals ──────────────────────────────────────────────────────

    def _write(self, data: Any, path: str | Path) -> Path:
        """Write ``data`` as pretty JSON to ``path``, with a trailing newline."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")  # keeps git happy
        return filepath
