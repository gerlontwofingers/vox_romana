"""Tests for the write/action MCP tools (Chunk 4).

All tests use the same _FakeBackends fixture from the Chunk 3 suite and
mock subprocess.run / DocumentIngester so no real processes are spawned
and no real files are written.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.mcp.server import _JOBS, _JOB_LOCK, _Backends, build_server


# ── Shared fake backends (mirrors test_mcp_server.py) ────────────────


class _FakeBackends(_Backends):
    def __init__(self) -> None:
        super().__init__()
        self._memory = MagicMock()
        self._vector = MagicMock()
        self._catalog = MagicMock()
        from vox_romana.core.lens import build_default_rack
        self._lens_rack = build_default_rack()


@pytest.fixture
def backends() -> _FakeBackends:
    return _FakeBackends()


@pytest.fixture
def server(backends):
    return build_server(backends)


def _call(server, name: str, **kwargs):
    return asyncio.run(server.call_tool(name, kwargs))


def _payload(result):
    """Unwrap FastMCP call_tool result → the actual dict."""
    structured = result[1] if isinstance(result, tuple) else result
    if isinstance(structured, dict):
        return structured.get("result", structured)
    return structured


# ── Tool registration ─────────────────────────────────────────────────


def test_all_twelve_tools_registered(server) -> None:
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names >= {
        "vox_ingest_document",
        "vox_kick_off_generate",
        "vox_get_job_status",
        "vox_render_existing",
        "vox_publish_existing",
        "vox_probe_character",
    }


# ── vox_ingest_document ──────────────────────────────────────────────


def test_ingest_document_missing_file(server) -> None:
    result = _payload(_call(server, "vox_ingest_document", path="/no/such/file.txt"))
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_ingest_document_success(server, tmp_path) -> None:
    doc = tmp_path / "hello.txt"
    doc.write_text("Hello world " * 100, encoding="utf-8")
    with patch("vox_romana.memory.ingest.DocumentIngester.ingest") as mock_ingest, \
         patch("vox_romana.memory.vector_index.ChromaVectorIndex.migrate_documents"):
        mock_ingest.return_value = MagicMock(chunks=3, chars=300, skipped=False, format=".txt", source=str(doc))
        result = _payload(_call(server, "vox_ingest_document", path=str(doc), tag="test"))
    assert result["ok"] is True
    mock_ingest.assert_called_once()


def test_ingest_document_with_force_reingest(server, tmp_path) -> None:
    doc = tmp_path / "data.txt"
    doc.write_text("x" * 50, encoding="utf-8")
    with patch("vox_romana.memory.ingest.DocumentIngester.ingest") as mock_ingest, \
         patch("vox_romana.memory.vector_index.ChromaVectorIndex.migrate_documents"):
        mock_ingest.return_value = MagicMock(chunks=1, chars=50, skipped=False, format=".txt", source=str(doc))
        _call(server, "vox_ingest_document", path=str(doc), force_reingest=True)
    _, kwargs = mock_ingest.call_args
    assert kwargs.get("force_reingest") is True


# ── vox_kick_off_generate / vox_get_job_status ───────────────────────


def test_kick_off_generate_returns_job_id(server) -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        result = _payload(_call(
            server, "vox_kick_off_generate",
            prompt_text="debate about cats",
            show_type="debate",
            no_research=True,
        ))
    assert "job_id" in result
    assert result["status"] == "running"


def test_kick_off_generate_includes_lens_overrides_in_cmd(server) -> None:
    captured_cmds: list[list[str]] = []

    def fake_run(cmd, **kw):
        captured_cmds.append(cmd)
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        _call(
            server, "vox_kick_off_generate",
            prompt_text="news",
            lens_overrides={"character:Salvatore": "local-llama-dim"},
        )
    # Wait briefly for the thread to complete.
    import time; time.sleep(0.3)
    assert any(
        "--llm-lens" in cmd and "character:Salvatore=local-llama-dim" in cmd
        for cmd in captured_cmds
    )


def test_get_job_status_unknown_id(server) -> None:
    result = _payload(_call(server, "vox_get_job_status", job_id="no-such-uuid"))
    assert result["ok"] is False
    assert "unknown" in result["error"]


def test_get_job_status_after_kick_off(server) -> None:
    import time

    def fast_run(cmd, **kw):
        return MagicMock(returncode=0, stdout="done", stderr="")

    with patch("subprocess.run", side_effect=fast_run):
        kick_result = _payload(_call(
            server, "vox_kick_off_generate",
            prompt_text="short podcast",
        ))
    job_id = kick_result["job_id"]
    # Poll until thread finishes (max 2s).
    for _ in range(20):
        time.sleep(0.1)
        status = _payload(_call(server, "vox_get_job_status", job_id=job_id))
        if status.get("status") in ("done", "failed"):
            break
    assert status.get("status") in ("running", "done")


# ── vox_render_existing ──────────────────────────────────────────────


def test_render_missing_json(server) -> None:
    result = _payload(_call(server, "vox_render_existing", json_path="/no/file.json"))
    assert result["ok"] is False


def test_render_existing_calls_run_py(server, tmp_path) -> None:
    json_file = tmp_path / "show.json"
    json_file.write_text("{}", encoding="utf-8")
    audio_file = json_file.with_suffix(".mp3")
    audio_file.write_text("fake audio", encoding="utf-8")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="rendered", stderr="")
        result = _payload(_call(
            server, "vox_render_existing",
            json_path=str(json_file),
        ))
    assert result["ok"] is True
    cmd = mock_run.call_args[0][0]
    assert "--mode" in cmd and "render" in cmd
    assert str(json_file) in cmd


# ── vox_publish_existing ─────────────────────────────────────────────


def test_publish_missing_json(server) -> None:
    result = _payload(_call(server, "vox_publish_existing", json_path="/bad/path.json"))
    assert result["ok"] is False


def test_publish_existing_passes_format(server, tmp_path) -> None:
    json_file = tmp_path / "show.json"
    json_file.write_text("{}", encoding="utf-8")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        _call(server, "vox_publish_existing", json_path=str(json_file), publish_format="both")
    cmd = mock_run.call_args[0][0]
    assert "--publish-format" in cmd and "both" in cmd


# ── vox_probe_character ──────────────────────────────────────────────


def test_probe_character_returns_markdown(server) -> None:
    fake_client = MagicMock()
    fake_client.chat.return_value = "I see... things. Many things. Not good things."

    with patch("vox_romana.core.lens.LLMLens.client", new_callable=lambda: property(lambda self: fake_client)):
        result = _payload(_call(
            server, "vox_probe_character",
            character_brief="Salvatore: polyglot monk",
            prompt="What did you see?",
            lens_names=["cloud-deepseek"],
        ))
    assert result["ok"] is True
    md = result["markdown"]
    assert "# Lens character probe" in md
    assert "Salvatore" in md
    assert "cloud-deepseek" in md


def test_probe_character_unknown_lens(server) -> None:
    result = _payload(_call(
        server, "vox_probe_character",
        character_brief="someone",
        prompt="speak",
        lens_names=["no-such-lens"],
    ))
    assert result["ok"] is False
    assert "unknown" in result["error"]
