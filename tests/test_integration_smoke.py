"""Smoke tests for the three key no-LLM pipelines.

All tests call ``run.main(argv_list)`` directly — no subprocess, no API keys,
no model downloads, no network I/O.

Pipelines covered:
  publish  — minimal debate/news/podcast JSON → TXT document
  ingest   — --dry-run: reports without writing JSONL
  render   — mocked TTS engine returns silent WAV → output WAV file created
"""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import run  # repo-root run.py; exposes run.main(argv_list)


# ─────────────────────────────────────────────────────────────────────────────
# Shared silent-audio helper
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_RATE = 22050  # Hz — matches Kokoro default


def _silent_wav(duration_ms: int = 200) -> bytes:
    """Return minimal valid WAV bytes containing ``duration_ms`` of silence."""
    num_samples = int(_SAMPLE_RATE * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)           # 16-bit PCM
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(b"\x00\x00" * num_samples)
    return buf.getvalue()


_SILENT_WAV = _silent_wav()


# ─────────────────────────────────────────────────────────────────────────────
# JSON fixture builders
# ─────────────────────────────────────────────────────────────────────────────
# Each builder writes a minimal but valid show JSON to tmp_path and returns
# the path.  The type-detection key (debate_id / show_id / podcast_id) is
# what publish and render use to route to the right formatter/renderer.


def _make_debate_json(tmp_path: Path) -> Path:
    data = {
        "debate_id": "smoke-debate-01",
        "title": "Test Debate",
        "events": [
            {"event_kind": "utterance", "speaker_name": "Alice", "text": "Hello world."},
            {"event_kind": "utterance", "speaker_name": "Bob",   "text": "Good morning."},
        ],
    }
    p = tmp_path / "debate.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _make_news_json(tmp_path: Path) -> Path:
    data = {
        "show_id": "smoke-news-01",
        "title": "Test News Show",
        "events": [
            {"event_kind": "utterance", "speaker_name": "Anchor", "text": "Good evening."},
        ],
    }
    p = tmp_path / "news.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _make_podcast_json(tmp_path: Path) -> Path:
    data = {
        "podcast_id": "smoke-podcast-01",
        "title": "Test Podcast",
        "events": [
            {"event_kind": "utterance", "speaker_name": "Host", "text": "Welcome to the show."},
        ],
    }
    p = tmp_path / "podcast.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Publish tests
# ─────────────────────────────────────────────────────────────────────────────


def test_publish_debate_json(tmp_path):
    """publish mode on a minimal debate JSON creates a non-empty .txt document."""
    fixture = _make_debate_json(tmp_path)
    run.main([
        "--mode", "publish",
        "--script-input", str(fixture),
        "--publish-output", str(tmp_path),
        "--quiet",
    ])
    txts = list(tmp_path.glob("*.txt"))
    assert txts, "Expected at least one .txt file to be created"
    assert txts[0].stat().st_size > 0, "Published .txt must not be empty"


def test_publish_news_json(tmp_path):
    """publish mode on a minimal news JSON creates a non-empty .txt document."""
    fixture = _make_news_json(tmp_path)
    run.main([
        "--mode", "publish",
        "--script-input", str(fixture),
        "--publish-output", str(tmp_path),
        "--quiet",
    ])
    txts = list(tmp_path.glob("*.txt"))
    assert txts, "Expected at least one .txt file to be created"
    assert txts[0].stat().st_size > 0, "Published .txt must not be empty"


def test_publish_podcast_json(tmp_path):
    """publish mode on a minimal podcast JSON creates a non-empty .txt document."""
    fixture = _make_podcast_json(tmp_path)
    run.main([
        "--mode", "publish",
        "--script-input", str(fixture),
        "--publish-output", str(tmp_path),
        "--quiet",
    ])
    txts = list(tmp_path.glob("*.txt"))
    assert txts, "Expected at least one .txt file to be created"
    assert txts[0].stat().st_size > 0, "Published .txt must not be empty"


# ─────────────────────────────────────────────────────────────────────────────
# Ingest dry-run test
# ─────────────────────────────────────────────────────────────────────────────


def test_ingest_dry_run(tmp_path):
    """--mode ingest --dry-run completes without writing JSONL to the document store."""
    from vox_romana.core.paths import DOCUMENT_DIR

    (tmp_path / "doc.txt").write_text(
        "This is test content for the smoke ingest test.\n",
        encoding="utf-8",
    )

    before = set(DOCUMENT_DIR.glob("*.jsonl")) if DOCUMENT_DIR.exists() else set()

    run.main([
        "--mode", "ingest",
        "--ingest-path", str(tmp_path),
        "--dry-run",
        "--quiet",
    ])

    after = set(DOCUMENT_DIR.glob("*.jsonl")) if DOCUMENT_DIR.exists() else set()
    assert before == after, "dry-run should not write any JSONL to the document store"


# ─────────────────────────────────────────────────────────────────────────────
# Render test — mocked TTS engine
# ─────────────────────────────────────────────────────────────────────────────


class _FakeTTSEngine:
    """Silent-audio stub.  Returns valid WAV bytes without loading any model.

    Patched into vox_romana.cli.runners.make_engine so the render pipeline
    runs end-to-end through TTSRenderer → AudioMixer → export without
    touching Kokoro, Piper, or any ML framework.
    """

    def synthesize(
        self,
        text,
        *,
        voice="default",
        emotion="",
        speed=1.0,
        language="en",
        **kwargs,
    ):
        from vox_romana.tts.engine import SynthesizedAudio

        return SynthesizedAudio(audio=_SILENT_WAV, sample_rate=_SAMPLE_RATE)

    def available_voices(self):
        # Small set of Kokoro-style IDs so the renderer's voice-assignment
        # logic has candidates to work with.
        return ["af_heart", "af_alloy", "am_adam", "am_echo"]


def test_render_debate_mocked_tts(tmp_path, monkeypatch):
    """render mode with mocked TTS engine creates a non-empty WAV file."""
    fixture = _make_debate_json(tmp_path)
    audio_out = tmp_path / "out.wav"

    fake_engine = _FakeTTSEngine()
    monkeypatch.setattr(
        "vox_romana.cli.runners.make_engine",
        lambda *args, **kwargs: fake_engine,
    )

    run.main([
        "--mode", "render",
        "--script-input", str(fixture),
        "--audio-output", str(audio_out),
        "--format", "wav",
        "--no-log-audio-duration",   # skip catalog write
        "--no-music",                # avoid loading MusicGen model in tests
        "--quiet",
    ])

    assert audio_out.exists(), "Expected audio file to be created"
    assert audio_out.stat().st_size > 0, "Expected non-empty audio file"
