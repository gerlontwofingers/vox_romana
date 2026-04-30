"""vox_romana MCP server — read-only + write tool surface (Chunks 3–4).

**Read-only tools (Chunk 3):**

* ``vox_search_memory``  — unified memory recall (LM + vector + FTS).
* ``vox_list_episodes``  — recent episodes from the catalog.
* ``vox_get_episode``    — one episode by run-id.
* ``vox_search_corpus``  — Chroma vector search over ingested docs.
* ``vox_list_voices``    — Kokoro + Piper voice catalogs.
* ``vox_list_lenses``    — configured LLM lens registry.

**Write / action tools (Chunk 4):**

* ``vox_ingest_document``    — ingest a file into the knowledge store.
* ``vox_kick_off_generate``  — launch a generate job (non-blocking); returns job_id.
* ``vox_get_job_status``     — poll a job launched by vox_kick_off_generate.
* ``vox_render_existing``    — render an existing show JSON → audio (blocking).
* ``vox_publish_existing``   — format a show JSON as a human-readable document.
* ``vox_probe_character``    — cast a character across every lens; returns markdown.

All write tools carry ``"destructive": false/true`` metadata for MCP
clients that want to gate confirmation dialogs.  Heavy backends
(Chroma, Catalog, Lens rack) are constructed lazily on first call.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# In-memory job registry (vox_kick_off_generate / vox_get_job_status)
# ─────────────────────────────────────────────────────────────────────

_JOB_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
# Job record schema:
#   job_id:      str
#   status:      "running" | "done" | "failed"
#   returncode:  int | None
#   stdout:      str  (accumulated)
#   stderr:      str  (accumulated)
#   cmd:         list[str]  (for debugging)


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

    # ── 7. vox_ingest_document ───────────────────────────────────────

    @server.tool(
        name="vox_ingest_document",
        description=(
            "Ingest a document (PDF, DOCX, TXT, HTML, CSV, …) into the "
            "vox_romana knowledge store so future --research runs and "
            "vox_search_corpus queries can find it.  The file must "
            "already exist on disk.  Returns an IngestStats summary.  "
            "Note: this is non-destructive (adds data; nothing deleted)."
        ),
    )
    def vox_ingest_document(
        path: str,
        tag: str = "",
        force_reingest: bool = False,
    ) -> dict[str, Any]:
        """Ingest a single file into the corpus.

        Args:
            path:           Absolute or workspace-relative path to the file.
            tag:            Optional label embedded in every chunk.
            force_reingest: Re-ingest even if already processed.
        """
        from vox_romana.memory.ingest import DocumentIngester

        abs_path = Path(path).expanduser().resolve()
        if not abs_path.exists():
            return {"ok": False, "error": f"file not found: {abs_path}"}

        ingester = DocumentIngester()
        try:
            stats = ingester.ingest(abs_path, tag=tag, force_reingest=force_reingest)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        # Eagerly migrate the newly ingested chunks into Chroma so
        # vox_search_corpus returns them without a server restart.
        try:
            from vox_romana.memory.vector_index import ChromaVectorIndex
            idx = ChromaVectorIndex()
            idx.migrate_documents(force=False)
        except Exception as exc:
            logger.debug("Vector index migrate (non-fatal): %s", exc)

        return {"ok": True, "stats": _serialise(stats)}

    # ── 8. vox_kick_off_generate ────────────────────────────────────

    @server.tool(
        name="vox_kick_off_generate",
        description=(
            "Launch a vox_romana generate job in the background and "
            "return a job_id immediately.  Poll vox_get_job_status with "
            "that id to check completion.  The show JSON will appear in "
            "output/text/{type}/ when done.  "
            "lens_overrides is a dict like "
            "{\"character:Salvatore\": \"local-llama-dim\"} — keys are "
            "role names, values are lens names from vox_list_lenses."
        ),
    )
    def vox_kick_off_generate(
        prompt_text: str,
        show_type: str | None = None,
        duration: float | None = None,
        llm_backend: str = "auto",
        lens_overrides: dict[str, str] | None = None,
        no_research: bool = True,
    ) -> dict[str, Any]:
        """Start a generate job (non-blocking).

        Args:
            prompt_text:    The brief / topic for the show.
            show_type:      'debate', 'news', 'podcast', 'audiobook', or None.
            duration:       Target audio duration in minutes.
            llm_backend:    'cloud', 'local', or 'auto'.
            lens_overrides: Per-role lens assignments.
            no_research:    Skip web research (default: True for speed).
        """
        import tempfile

        # Write the prompt to a temp file so run.py can read it.
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="vox_mcp_", delete=False,
            encoding="utf-8",
        )
        try:
            tmp.write(prompt_text)
            tmp.flush()
            prompt_path = tmp.name
        finally:
            tmp.close()

        # Build the run.py command.
        repo_root = Path(__file__).resolve().parent.parent.parent
        cmd = [sys.executable, str(repo_root / "run.py"), prompt_path,
               "--mode", "generate", "--llm-backend", llm_backend]
        if show_type:
            cmd += ["--type", show_type]
        if duration is not None:
            cmd += ["--duration", str(duration)]
        if no_research:
            cmd.append("--no-research")
        for role, lens in (lens_overrides or {}).items():
            cmd += ["--llm-lens", f"{role}={lens}"]

        job_id = str(uuid.uuid4())
        with _JOB_LOCK:
            _JOBS[job_id] = {
                "job_id": job_id,
                "status": "running",
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "cmd": cmd,
                "_prompt_tmp": prompt_path,
            }

        def _run(job_id: str, cmd: list[str], prompt_path: str) -> None:
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=str(repo_root),
                )
                with _JOB_LOCK:
                    _JOBS[job_id].update({
                        "status": "done" if proc.returncode == 0 else "failed",
                        "returncode": proc.returncode,
                        "stdout": proc.stdout[-8000:],  # cap to avoid huge payloads
                        "stderr": proc.stderr[-4000:],
                    })
            except Exception as exc:
                with _JOB_LOCK:
                    _JOBS[job_id].update({
                        "status": "failed",
                        "returncode": -1,
                        "stderr": str(exc),
                    })
            finally:
                try:
                    os.unlink(prompt_path)
                except OSError:
                    pass

        t = threading.Thread(target=_run, args=(job_id, cmd, prompt_path), daemon=True)
        t.start()

        return {"job_id": job_id, "status": "running", "cmd": cmd}

    # ── 9. vox_get_job_status ────────────────────────────────────────

    @server.tool(
        name="vox_get_job_status",
        description=(
            "Poll the status of a generate (or other) job launched by "
            "vox_kick_off_generate.  Returns status ('running', 'done', "
            "'failed'), returncode, and the last 8 KB of stdout/stderr."
        ),
    )
    def vox_get_job_status(job_id: str) -> dict[str, Any]:
        with _JOB_LOCK:
            job = _JOBS.get(job_id)
        if job is None:
            return {"ok": False, "error": f"unknown job_id: {job_id}"}
        # Return a safe copy (drop internal keys prefixed with _).
        return {k: v for k, v in job.items() if not k.startswith("_")}

    # ── 10. vox_render_existing ──────────────────────────────────────

    @server.tool(
        name="vox_render_existing",
        description=(
            "Render an existing show JSON to audio (MP3 by default).  "
            "Blocking: waits for TTS synthesis to complete before "
            "returning.  Returns the path to the audio file and a "
            "duration estimate.  The JSON file must already exist."
        ),
    )
    def vox_render_existing(
        json_path: str,
        engine: str = "both",
        audio_format: str = "mp3",
    ) -> dict[str, Any]:
        """Synthesise audio from an existing show JSON.

        Args:
            json_path:    Absolute or relative path to the show JSON.
            engine:       'kokoro', 'piper', or 'both'.
            audio_format: 'mp3', 'wav', 'flac', or 'ogg'.
        """
        repo_root = Path(__file__).resolve().parent.parent.parent
        abs_json = Path(json_path).expanduser().resolve()
        if not abs_json.exists():
            return {"ok": False, "error": f"file not found: {abs_json}"}

        cmd = [
            sys.executable, str(repo_root / "run.py"),
            "--mode", "render",
            "--script-input", str(abs_json),
            "--engine", engine,
            "--format", audio_format,
            "--no-log-audio-duration",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(repo_root)
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        audio_path = abs_json.with_suffix(f".{audio_format}")
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "audio_path": str(audio_path) if audio_path.exists() else None,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-2000:],
        }

    # ── 11. vox_publish_existing ─────────────────────────────────────

    @server.tool(
        name="vox_publish_existing",
        description=(
            "Format an existing show JSON as a human-readable document "
            "(TXT and/or PDF).  The output is written next to the JSON "
            "unless publish_output_dir is given.  Returns the paths of "
            "created files.  Non-destructive: the source JSON is untouched."
        ),
    )
    def vox_publish_existing(
        json_path: str,
        publish_format: str = "txt",
        publish_output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Produce a human-readable document from a show JSON.

        Args:
            json_path:          Absolute or relative path to the show JSON.
            publish_format:     'txt', 'pdf', or 'both'.
            publish_output_dir: Optional output directory (defaults to JSON dir).
        """
        repo_root = Path(__file__).resolve().parent.parent.parent
        abs_json = Path(json_path).expanduser().resolve()
        if not abs_json.exists():
            return {"ok": False, "error": f"file not found: {abs_json}"}

        cmd = [
            sys.executable, str(repo_root / "run.py"),
            "--mode", "publish",
            "--script-input", str(abs_json),
            "--publish-format", publish_format,
        ]
        if publish_output_dir:
            cmd += ["--publish-output", str(Path(publish_output_dir).expanduser())]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(repo_root)
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-2000:],
        }

    # ── 12. vox_probe_character ──────────────────────────────────────

    @server.tool(
        name="vox_probe_character",
        description=(
            "Cast a character across every configured LLM lens and "
            "return a side-by-side Markdown report.  Use this to audition "
            "which lens gives the most authentic voice for a character "
            "before committing to a casting decision.  "
            "Optionally restrict to specific lenses with lens_names.  "
            "The report includes personality_hint, latency, and the "
            "verbatim model response per lens."
        ),
    )
    def vox_probe_character(
        character_brief: str,
        prompt: str,
        lens_names: list[str] | None = None,
        max_tokens: int = 400,
    ) -> dict[str, Any]:
        """Run the lens character probe and return markdown.

        Args:
            character_brief: 1–3 sentence description of the character.
            prompt:          What the character should say/do.
            lens_names:      Restrict to these lenses (default: all).
            max_tokens:      Max tokens per lens response.
        """
        import time

        from vox_romana.core.lens import build_default_rack
        from vox_romana.core.llm_config import load_llm_config
        from vox_romana.core.lens import LLMLensRack

        try:
            rack = LLMLensRack.from_config(load_llm_config())
        except Exception:
            rack = build_default_rack()

        all_lenses = list(rack._lenses.values())
        if lens_names:
            unknown = [n for n in lens_names if n not in rack._lenses]
            if unknown:
                return {"ok": False, "error": f"unknown lenses: {', '.join(unknown)}"}
            all_lenses = [rack._lenses[n] for n in lens_names]

        SYSTEM = (
            "You are voicing a single character in a creative work. "
            "Stay strictly in character. Do not narrate or break the fourth wall. "
            "Do not preface with 'Sure' or 'Here is'. Just speak.\n\n"
            "CHARACTER BRIEF:\n{brief}\n"
        )

        results: list[tuple] = []
        for lens in all_lenses:
            msgs = [
                {"role": "system", "content": SYSTEM.format(brief=character_brief.strip())},
                {"role": "user", "content": prompt.strip()},
            ]
            t0 = time.perf_counter()
            try:
                text = lens.client.chat(msgs, temperature=0.8, max_tokens=max_tokens)
                text = text.strip()
            except Exception as exc:
                text = f"[unavailable: {type(exc).__name__}: {exc}]"
            dt = time.perf_counter() - t0
            results.append((lens, text, dt))

        # Build markdown report.
        lines = [
            "# Lens character probe", "",
            "## Character brief", "",
            "> " + character_brief.strip().replace("\n", "\n> "), "",
            "## Prompt", "",
            "> " + prompt.strip().replace("\n", "\n> "), "",
        ]
        for lens, text, dt in results:
            lines += [
                f"## Lens — `{lens.name}` ({lens.backend} · {lens.model})", "",
            ]
            if lens.personality_hint:
                lines += [f"*Hint:* {lens.personality_hint}", ""]
            lines += [
                f"*Latency:* {dt:.2f}s · *Cost tier:* {lens.cost_tier}", "",
                "> " + text.replace("\n", "\n> "), "",
            ]

        return {"ok": True, "markdown": "\n".join(lines)}

    return server


def main() -> None:  # pragma: no cover
    """Entry point for ``python -m vox_romana.mcp``."""
    logging.basicConfig(level=logging.INFO)
    server = build_server()
    server.run()  # FastMCP defaults to stdio transport
