"""Tests for vox_romana.mcp.server.

The strategy: build the server with a hand-rolled fake ``_Backends``
whose properties return ``MagicMock`` instances.  Then invoke each
tool via the FastMCP introspection API and assert it returns
JSON-safe data plus that the right backend method was called.

We avoid spinning up the actual stdio transport — that's manual smoke
territory.  Tool functions are reachable through ``server._tool_manager``
which is FastMCP's tool registry.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from vox_romana.mcp.server import _Backends, _serialise, build_server


# ── Fakes ────────────────────────────────────────────────────────────


@dataclass
class _FakeRunRecord:
    run_id: str
    title: str
    content_type: str
    series_id: str | None = None
    episode_number: int | None = None


class _FakeBackends(_Backends):
    """Drop-in _Backends with mockable subsystems."""

    def __init__(self) -> None:
        super().__init__()
        self._memory = MagicMock()
        self._vector = MagicMock()
        self._catalog = MagicMock()
        # Real lens rack so vox_list_lenses returns sensible structure.
        from vox_romana.core.lens import build_default_rack
        self._lens_rack = build_default_rack()


@pytest.fixture
def backends() -> _FakeBackends:
    return _FakeBackends()


@pytest.fixture
def server(backends: _FakeBackends):
    return build_server(backends)


# ── Helpers ──────────────────────────────────────────────────────────


def _call_tool(server, name: str, **kwargs):
    """Invoke an MCP tool by name through FastMCP's tool manager."""
    return asyncio.run(server.call_tool(name, kwargs))


# ── _serialise ───────────────────────────────────────────────────────


def test_serialise_handles_primitives_and_dataclasses() -> None:
    rec = _FakeRunRecord(run_id="r1", title="t", content_type="podcast")
    out = _serialise({"a": 1, "rec": rec, "list": [rec]})
    assert out["a"] == 1
    assert out["rec"]["run_id"] == "r1"
    assert out["list"][0]["title"] == "t"


def test_serialise_stringifies_paths() -> None:
    from pathlib import Path
    out = _serialise(Path("/tmp/x"))
    assert isinstance(out, str)


# ── Server construction ──────────────────────────────────────────────


def test_build_server_registers_six_tools(server) -> None:
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    # Chunk 3 read-only tools
    assert names >= {
        "vox_search_memory",
        "vox_list_episodes",
        "vox_get_episode",
        "vox_search_corpus",
        "vox_list_voices",
        "vox_list_lenses",
    }
    # Chunk 4 write/action tools also present
    assert names >= {
        "vox_ingest_document",
        "vox_kick_off_generate",
        "vox_get_job_status",
        "vox_render_existing",
        "vox_publish_existing",
        "vox_probe_character",
    }


# ── vox_search_memory ────────────────────────────────────────────────


def test_search_memory_invokes_memory_query(server, backends) -> None:
    backends._memory.query.return_value = {"answer": "stub", "hits": []}
    _call_tool(server, "vox_search_memory", query="ubi", top_k=3)
    backends._memory.query.assert_called_once_with("ubi", top_k=3, synthesize=True)


def test_search_memory_clamps_top_k(server, backends) -> None:
    backends._memory.query.return_value = {}
    _call_tool(server, "vox_search_memory", query="x", top_k=999)
    assert backends._memory.query.call_args.kwargs["top_k"] == 50


# ── vox_list_episodes / vox_get_episode ──────────────────────────────


def test_list_episodes_passes_filters(server, backends) -> None:
    backends._catalog.recent.return_value = [
        _FakeRunRecord(run_id="a", title="A", content_type="news"),
    ]
    _call_tool(server, "vox_list_episodes", series_id="s1", limit=5)
    backends._catalog.recent.assert_called_once_with(
        content_type=None, series_id="s1", limit=5
    )


def test_get_episode_returns_none_for_missing(server, backends) -> None:
    backends._catalog.get_run.return_value = None
    result = _call_tool(server, "vox_get_episode", run_id="missing")
    # FastMCP returns (content_blocks, structured_content) tuple from call_tool.
    # Drill into the structured payload.
    structured = result[1] if isinstance(result, tuple) else result
    assert structured in (None, {"result": None})


def test_get_episode_returns_record(server, backends) -> None:
    backends._catalog.get_run.return_value = _FakeRunRecord(
        run_id="abc", title="The Title", content_type="debate"
    )
    result = _call_tool(server, "vox_get_episode", run_id="abc")
    structured = result[1] if isinstance(result, tuple) else result
    payload = structured.get("result", structured) if isinstance(structured, dict) else structured
    assert payload["run_id"] == "abc"
    assert payload["title"] == "The Title"


# ── vox_search_corpus ────────────────────────────────────────────────


def test_search_corpus_normalises_hits(server, backends) -> None:
    backends._vector.search.return_value = [
        ("hello world", 0.91, "doc#1"),
        ("again", 0.5, "doc#2"),
    ]
    result = _call_tool(server, "vox_search_corpus", query="hi", top_k=2)
    structured = result[1] if isinstance(result, tuple) else result
    items = structured.get("result", structured) if isinstance(structured, dict) else structured
    assert items[0] == {"text": "hello world", "score": 0.91, "source_id": "doc#1"}


# ── vox_list_voices ──────────────────────────────────────────────────


def test_list_voices_filters_by_engine(server) -> None:
    result = _call_tool(server, "vox_list_voices", engine="kokoro")
    structured = result[1] if isinstance(result, tuple) else result
    items = structured.get("result", structured) if isinstance(structured, dict) else structured
    assert isinstance(items, list)
    assert items, "expected at least one Kokoro voice"
    assert all(v["engine"] == "kokoro" for v in items)


def test_list_voices_returns_both_when_unfiltered(server) -> None:
    result = _call_tool(server, "vox_list_voices")
    structured = result[1] if isinstance(result, tuple) else result
    items = structured.get("result", structured) if isinstance(structured, dict) else structured
    engines = {v["engine"] for v in items}
    assert engines >= {"kokoro"}  # piper may be empty if no models installed


# ── vox_list_lenses ──────────────────────────────────────────────────


def test_list_lenses_returns_default_routes_and_catalog(server) -> None:
    result = _call_tool(server, "vox_list_lenses")
    structured = result[1] if isinstance(result, tuple) else result
    payload = structured.get("result", structured) if isinstance(structured, dict) else structured
    assert "default" in payload
    assert "routes" in payload
    assert isinstance(payload["lenses"], list)
    names = {l["name"] for l in payload["lenses"]}
    assert "cloud-deepseek" in names
    assert "local-llama-dim" in names
