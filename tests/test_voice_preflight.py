"""Tests for Chunk 8 — voice preflight.

Covers check_piper_voices() in vox_romana.tts.voice_preflight:
  - skips when engine is not piper-related
  - skips for English-only shows
  - warns when a non-English language is missing from model_dir
  - silent when the required voice is already installed
  - handles character_langs overrides
  - reads from a dict or a file path
  - handles missing/corrupt show JSON gracefully
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def _check(**kwargs):
    from vox_romana.tts.voice_preflight import check_piper_voices
    return check_piper_voices(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Engine filtering
# ─────────────────────────────────────────────────────────────────────────────


class TestEngineFilter:
    def test_kokoro_engine_skips_check(self):
        show = {"language": "es", "title": "T"}
        result = _check(show_json=show, model_dir="/nonexistent", engine="kokoro")
        assert result == []

    def test_vc_engine_skips_check(self):
        show = {"language": "es", "title": "T"}
        result = _check(show_json=show, model_dir="/nonexistent", engine="vc")
        assert result == []

    def test_piper_engine_runs_check(self):
        show = {"language": "es", "title": "T"}
        result = _check(show_json=show, model_dir="/nonexistent", engine="piper")
        assert "es" in result

    def test_both_engine_runs_check(self):
        show = {"language": "fr", "title": "T"}
        result = _check(show_json=show, model_dir="/nonexistent", engine="both")
        assert "fr" in result


# ─────────────────────────────────────────────────────────────────────────────
# Language extraction
# ─────────────────────────────────────────────────────────────────────────────


class TestLanguageExtraction:
    def test_english_only_no_warning(self):
        show = {"language": "en", "title": "T"}
        result = _check(show_json=show, model_dir="/nonexistent", engine="both")
        assert result == []

    def test_no_language_field_no_warning(self):
        show = {"title": "T"}
        result = _check(show_json=show, model_dir="/nonexistent", engine="both")
        assert result == []

    def test_show_language_es_reported_missing(self):
        show = {"language": "es", "title": "T"}
        result = _check(show_json=show, model_dir="/nonexistent", engine="both")
        assert result == ["es"]

    def test_character_langs_override_detected(self):
        show = {
            "language": "en",
            "character_langs": {"Carlos Mendoza": "es"},
            "title": "T",
        }
        result = _check(show_json=show, model_dir="/nonexistent", engine="both")
        assert "es" in result

    def test_english_character_lang_not_reported(self):
        show = {
            "language": "en",
            "character_langs": {"Alice": "en"},
            "title": "T",
        }
        result = _check(show_json=show, model_dir="/nonexistent", engine="both")
        assert result == []

    def test_multiple_missing_languages_all_reported(self):
        show = {
            "language": "es",
            "character_langs": {"Giulia": "it"},
            "title": "T",
        }
        result = _check(show_json=show, model_dir="/nonexistent", engine="both")
        assert "es" in result
        assert "it" in result


# ─────────────────────────────────────────────────────────────────────────────
# Installed voice detection
# ─────────────────────────────────────────────────────────────────────────────


class TestInstalledVoiceDetection:
    def test_installed_voice_silences_warning(self, tmp_path):
        """If an es_* .onnx file exists, no warning for 'es'."""
        voice_dir = tmp_path / "piper-voices"
        voice_dir.mkdir()
        (voice_dir / "es_ES-davefx-medium.onnx").write_bytes(b"")
        show = {"language": "es", "title": "T"}
        result = _check(show_json=show, model_dir=voice_dir, engine="both")
        assert result == []

    def test_missing_onnx_reports_language(self, tmp_path):
        """Empty voices dir → language reported as missing."""
        voice_dir = tmp_path / "piper-voices"
        voice_dir.mkdir()
        show = {"language": "de", "title": "T"}
        result = _check(show_json=show, model_dir=voice_dir, engine="both")
        assert "de" in result

    def test_json_sidecar_not_counted_as_voice(self, tmp_path):
        """A .onnx.json sidecar file should NOT be treated as a voice model."""
        voice_dir = tmp_path / "piper-voices"
        voice_dir.mkdir()
        (voice_dir / "es_ES-davefx-medium.onnx.json").write_bytes(b"{}")
        show = {"language": "es", "title": "T"}
        result = _check(show_json=show, model_dir=voice_dir, engine="both")
        assert "es" in result  # .onnx.json alone should not satisfy the check


# ─────────────────────────────────────────────────────────────────────────────
# File path input
# ─────────────────────────────────────────────────────────────────────────────


class TestFilePathInput:
    def test_reads_from_json_file(self, tmp_path):
        show_file = tmp_path / "show.json"
        show_file.write_text(json.dumps({"language": "es", "title": "T"}))
        result = _check(show_json=show_file, model_dir="/nonexistent", engine="both")
        assert "es" in result

    def test_missing_file_returns_empty(self, tmp_path):
        result = _check(
            show_json=tmp_path / "does_not_exist.json",
            model_dir="/nonexistent",
            engine="both",
        )
        assert result == []

    def test_corrupt_json_returns_empty(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")
        result = _check(show_json=bad_file, model_dir="/nonexistent", engine="both")
        assert result == []
