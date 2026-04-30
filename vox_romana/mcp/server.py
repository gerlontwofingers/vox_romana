"""vox_romana MCP server — read-only tool surface (Chunk 3).

Exposes six tools to MCP clients:

* ``vox_search_memory``  — unified memory recall (LM + vector + FTS).
* ``vox_list_episodes``  — recent episodes from the catalog.
* ``vox_get_episode``    — one episode by run-id.
* ``vox_search_corpus``  — Chroma vector search over ingested docs.
* ``vox_list_voices``    — Kokoro + Piper voice catalogs.
* ``vox_list_lenses``    — configured LLM lens registry.

Heavy backends (Chroma, Catalog, Lens rack) are constructed lazily on
first call so an MCP client connecting to an unconfigured workspace
gets a useful error rather than a startup crash.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Backend cache — one instance per process, built on first use.
# ─────────────────────────────────────────────────────────────────────


class _Backends:
    """Lazy holders for vox_romana subsystems used by the MCP tools.

    Instances are constructed on first access and reused for the life
    of the server.  Constructors that require config or external
    services (Chroma, the catalog DB) raise from inside the property,
    where the FastMCP layer turns the exception into a tool error
    rather than a server crash.
    """

    def __init__(self) -> None:
        self._memory = None
        self._vector = None
        self._catalog = None
        self._lens_rack = None

    @property
    def memory(self):
        if self._memory is None:
            from vox_romana.memory.recall import MemoryRecall
            self._memory = MemoryRecall()
        return self._memory

    @property
    def vector(self):
        if self._vector is None:
            from vox_romana.memory.vector_index import ChromaVectorIndex
            self._vector = ChromaVectorIndex()
        return self._vector

    @property
    def catalog(self):
        if self._catalog is None:
            from vox_romana.core.catalog import Catalog
            self._catalog = Catalog()
        return self._catalog

    @property
    def lens_rack(self):
        if self._lens_rack is None:
            from vox_romana.core.lens import build_default_rack
            from vox_romana.core.llm_config import load_llm_config
            from vox_romana.core.lens import LLMLensRack
            try:
                self._lens_rack = LLMLensRack.from_config(load_llm_config())
            except Exception as exc:  # config missing or malformed
                logger.debug("Falling back to default rack: %s", exc)
                self._lens_rack = build_default_rack()
        return self._lens_rack


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _serialise(obj: Any) -> Any:
    """Convert dataclasses / Path / arbitrary objects into JSON-safe data."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialise(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _serialise(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return _serialise(asdict(obj))
    # pathlib.Path, datetime, etc.
    return str(obj)


# ─────────────────────────────────────────────────────────────────────
# Server factory
# ─────────────────────────────────────────────────────────────────────


def build_server(backends: _Backends | None = None) -> FastMCP:
    """Construct (but do not run) a FastMCP server with all read-only tools.

    Splitting construction from ``run`` makes the server testable: a
    test can inject a fake ``_Backends`` instance whose properties
    return mocks.

    Args:
        backends: Optional pre-built ``_Backends`` instance.  When
                  omitted, a fresh one is created with lazy real
                  backends.

    Returns:
        A configured :class:`FastMCP` instance.  Call ``.run()`` to
        serve over stdio.
    """
    backends = backends or _Backends()
    server = FastMCP("vox_romana")

    # ── 1. vox_search_memory ────────────────────────────────────────

    @server.tool(
        name="vox_search_memory",
        description=(
            "Unified semantic search over vox_romana's memory store.  "
            "Combines LM-synthesised answers, vector similarity, and "
            "FTS5 keyword fallback.  Use this for natural-language "
            "questions like 'what have we said about UBI?' or "
            "'episodes that mention cooperative game theory'."
        ),
    )
    def vox_search_memory(query: str, top_k: int = 5, synthesize: bool = True) -> dict[str, Any]:
        """Search across episodes, transcripts, and ingested documents.

        Args:
            query:       Natural-language question or keyword phrase.
            top_k:       Maximum number of source hits to return (1–50).
            synthesize:  When True, also include an LM-synthesised answer.
        """
        top_k = max(1, min(int(top_k), 50))
        result = backends.memory.query(query, top_k=top_k, synthesize=synthesize)
        return _serialise(result)

    # ── 2. vox_list_episodes ────────────────────────────────────────

    @server.tool(
        name="vox_list_episodes",
        description=(
            "List the most recent episodes recorded in the catalog.  "
            "Optionally filter by series_id (e.g. 'trash_pandas') or by "
            "content_type ('debate', 'news', 'podcast', 'audiobook').  "
            "Returns run_id, title, brief, content_type, series_id, "
            "episode_number, created_at."
        ),
    )
    def vox_list_episodes(
        series_id: str | None = None,
        content_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        rows = backends.catalog.recent(
            content_type=content_type,
            series_id=series_id,
            limit=limit,
        )
        return [_serialise(r) for r in rows]

    # ── 3. vox_get_episode ──────────────────────────────────────────

    @server.tool(
        name="vox_get_episode",
        description=(
            "Fetch one episode's catalog record by run_id (the value "
            "returned by vox_list_episodes).  Returns null if no such "
            "episode exists."
        ),
    )
    def vox_get_episode(run_id: str) -> dict[str, Any] | None:
        rec = backends.catalog.get_run(run_id)
        return _serialise(rec) if rec is not None else None

    # ── 4. vox_search_corpus ────────────────────────────────────────

    @server.tool(
        name="vox_search_corpus",
        description=(
            "Vector-similarity search over ingested documents (the "
            "Chroma corpus).  Use this when you want raw passage hits "
            "rather than synthesised answers — e.g. quoting source "
            "material verbatim.  Returns a list of "
            "{text, score, source_id} objects sorted by similarity."
        ),
    )
    def vox_search_corpus(
        query: str,
        top_k: int = 5,
        source_type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        top_k = max(1, min(int(top_k), 50))
        hits = backends.vector.search(
            query, top_k=top_k, source_type_filter=source_type_filter
        )
        return [
            {"text": text, "score": float(score), "source_id": source_id}
            for text, score, source_id in hits
        ]

    # ── 5. vox_list_voices ──────────────────────────────────────────

    @server.tool(
        name="vox_list_voices",
        description=(
            "List all TTS voices available to vox_romana.  Optionally "
            "filter by engine ('kokoro' or 'piper').  Returns a list of "
            "voice descriptors with gender, accent, age, tone, best_for "
            "and engine — useful for picking a voice for a character."
        ),
    )
    def vox_list_voices(engine: str | None = None) -> list[dict[str, Any]]:
        from vox_romana.tts.voice_catalog import (
            KOKORO_VOICE_CATALOG,
            PIPER_VOICE_CATALOG,
        )
        sources: list[dict] = []
        if engine in (None, "kokoro"):
            sources.extend(KOKORO_VOICE_CATALOG.values())
        if engine in (None, "piper"):
            sources.extend(PIPER_VOICE_CATALOG.values())
        return [_serialise(v) for v in sources]

    # ── 6. vox_list_lenses ──────────────────────────────────────────

    @server.tool(
        name="vox_list_lenses",
        description=(
            "List the configured LLM lens registry.  Each lens is a "
            "named LLM client with creative metadata: personality_hint, "
            "cost_tier, language_strengths.  Use this to suggest "
            "casting for a character — e.g. a small/brittle lens for "
            "an ingenue, a polyglot lens for a translator."
        ),
    )
    def vox_list_lenses() -> dict[str, Any]:
        rack = backends.lens_rack
        return {
            "default": rack.default_lens_name,
            "routes": rack.routes(),
            "lenses": rack.list_lenses(),
        }

    return server


def main() -> None:  # pragma: no cover
    """Entry point for ``python -m vox_romana.mcp``."""
    logging.basicConfig(level=logging.INFO)
    server = build_server()
    server.run()  # FastMCP defaults to stdio transport
