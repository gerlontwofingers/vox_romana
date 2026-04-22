"""Tests for Shot 3 — Language-Aware TTS.

Covers:
  * `_lang_from_voice_id` parsing of Piper voice ids.
  * KokoroEngine: rejects non-English `language=` requests.
  * PiperEngine: validates voice/language match, picks language-appropriate
    default voice, exposes `available_languages()` /
    `available_voices_for_language()`.
  * TTSRenderer: extracts `language` from show data, threads it through to
    `engine.synthesize()`, pre-filters Piper voice pool when language != en.
  * `vox_romana.core.languages.supported_languages()`: scans Piper directory
    and always includes "en".
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.tts.engine import BaseTTSEngine, SynthesizedAudio
from vox_romana.tts.engines.kokoro_engine import KOKORO_LANGUAGES, KokoroEngine
from vox_romana.tts.engines.piper_engine import PiperEngine, _lang_from_voice_id
from vox_romana.tts.renderer import TTSRenderer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _wav_bytes(duration_ms: int = 100) -> bytes:
    sample_rate = 22050
    num_samples = int(sample_rate * duration_ms / 1000)
    raw = b"\x00" * (num_samples * 2)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)
    return buf.getvalue()


class _RecordingEngine(BaseTTSEngine):
    """Test double that records every call and returns silent WAVs."""

    def __init__(self, voices: list[str] | None = None):
        self.voices = voices or ["voice_a", "voice_b"]
        self.calls: list[dict] = []

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "default",
        emotion: str = "",
        speed: float = 1.0,
        language: str = "en",
        **kwargs,
    ) -> SynthesizedAudio:
        self.calls.append(
            {"text": text, "voice": voice, "emotion": emotion,
             "speed": speed, "language": language, "kwargs": kwargs}
        )
        return SynthesizedAudio(audio=_wav_bytes(), sample_rate=22050)

    def available_voices(self) -> list[str]:
        return list(self.voices)


# ─────────────────────────────────────────────────────────────────────────────
# _lang_from_voice_id
# ─────────────────────────────────────────────────────────────────────────────


class TestLangFromVoiceId:
    def test_standard_us_voice(self):
        assert _lang_from_voice_id("en_US-lessac-medium") == "en"

    def test_standard_gb_voice(self):
        assert _lang_from_voice_id("en_GB-cori-high") == "en"

    def test_spanish_voice(self):
        assert _lang_from_voice_id("es_ES-davefx-medium") == "es"

    def test_french_voice(self):
        assert _lang_from_voice_id("fr_FR-siwis-low") == "fr"

    def test_german_voice(self):
        assert _lang_from_voice_id("de_DE-thorsten-high") == "de"

    def test_no_underscore_returns_none(self):
        assert _lang_from_voice_id("default") is None
        assert _lang_from_voice_id("af_heart") == "af"  # 2 letters → parses

    def test_empty_returns_none(self):
        assert _lang_from_voice_id("") is None

    def test_numeric_prefix_rejected(self):
        assert _lang_from_voice_id("12_US-foo-medium") is None

    def test_too_long_prefix_rejected(self):
        assert _lang_from_voice_id("english_US-foo-medium") is None


# ─────────────────────────────────────────────────────────────────────────────
# KokoroEngine — language validation
# ─────────────────────────────────────────────────────────────────────────────


class TestKokoroLanguage:
    def test_kokoro_languages_constant_is_english_only(self):
        assert KOKORO_LANGUAGES == frozenset({"en"})

    def test_synthesize_default_language_succeeds(self):
        engine = KokoroEngine()
        # Patch the lazy generate_audio function and pretend it wrote a wav.
        with patch(
            "vox_romana.tts.engines.kokoro_engine._lazy_generate_audio"
        ) as mock_gen, patch(
            "vox_romana.tts.engines.kokoro_engine.Path"
        ) as mock_path_cls:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.read_bytes.return_value = _wav_bytes()
            mock_path_cls.return_value.__truediv__.return_value = mock_path
            result = engine.synthesize("hello", voice="af_heart")
        assert isinstance(result, SynthesizedAudio)
        mock_gen.assert_called_once()

    def test_synthesize_spanish_raises(self):
        engine = KokoroEngine()
        with pytest.raises(ValueError, match="English"):
            engine.synthesize("hola", voice="af_heart", language="es")

    def test_synthesize_french_raises(self):
        engine = KokoroEngine()
        with pytest.raises(ValueError, match="French|fr|English"):
            engine.synthesize("bonjour", voice="af_heart", language="fr")

    def test_error_message_mentions_piper(self):
        engine = KokoroEngine()
        with pytest.raises(ValueError, match="Piper"):
            engine.synthesize("ciao", language="it")


# ─────────────────────────────────────────────────────────────────────────────
# PiperEngine — language helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_piper_dir(tmp_path: Path, voice_ids: list[str]) -> Path:
    """Create empty .onnx + .onnx.json files so PiperEngine sees them."""
    for vid in voice_ids:
        (tmp_path / f"{vid}.onnx").write_bytes(b"")
        (tmp_path / f"{vid}.onnx.json").write_text("{}")
    return tmp_path


class TestPiperLanguageHelpers:
    def test_available_languages_empty_dir(self, tmp_path):
        engine = PiperEngine(model_dir=tmp_path)
        assert engine.available_languages() == []

    def test_available_languages_english_only(self, tmp_path):
        _make_piper_dir(tmp_path, ["en_US-lessac-medium", "en_GB-cori-high"])
        engine = PiperEngine(model_dir=tmp_path)
        assert engine.available_languages() == ["en"]

    def test_available_languages_multiple(self, tmp_path):
        _make_piper_dir(tmp_path, [
            "en_US-lessac-medium",
            "es_ES-davefx-medium",
            "fr_FR-siwis-low",
        ])
        engine = PiperEngine(model_dir=tmp_path)
        assert engine.available_languages() == ["en", "es", "fr"]

    def test_available_voices_for_language(self, tmp_path):
        _make_piper_dir(tmp_path, [
            "en_US-lessac-medium",
            "en_GB-cori-high",
            "es_ES-davefx-medium",
        ])
        engine = PiperEngine(model_dir=tmp_path)
        assert engine.available_voices_for_language("en") == [
            "en_GB-cori-high",
            "en_US-lessac-medium",
        ]
        assert engine.available_voices_for_language("es") == [
            "es_ES-davefx-medium",
        ]
        assert engine.available_voices_for_language("de") == []

    def test_voices_without_lang_prefix_ignored(self, tmp_path):
        _make_piper_dir(tmp_path, ["en_US-lessac-medium", "custom-voice"])
        engine = PiperEngine(model_dir=tmp_path)
        # "custom-voice" has no underscore → not counted as any language.
        assert engine.available_languages() == ["en"]


# ─────────────────────────────────────────────────────────────────────────────
# PiperEngine — synthesize() language validation
# ─────────────────────────────────────────────────────────────────────────────


class TestPiperSynthesizeLanguage:
    def test_voice_language_mismatch_raises(self, tmp_path):
        _make_piper_dir(tmp_path, ["en_US-lessac-medium", "es_ES-davefx-medium"])
        engine = PiperEngine(model_dir=tmp_path)
        with pytest.raises(ValueError, match="speaks 'en'"):
            engine.synthesize(
                "hola", voice="en_US-lessac-medium", language="es",
            )

    def test_default_voice_picks_language_match(self, tmp_path):
        _make_piper_dir(tmp_path, [
            "en_US-lessac-medium",
            "es_ES-davefx-medium",
        ])
        engine = PiperEngine(model_dir=tmp_path)
        # Patch _load_voice so we don't actually try to load a fake .onnx.
        with patch.object(engine, "_load_voice") as mock_load:
            mock_voice = MagicMock()
            mock_voice.config.sample_rate = 22050

            def _fake_synth(text, wav_file, syn_config=None):
                # Write a minimal valid WAV header so wave.open succeeds.
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                wav_file.writeframes(b"\x00\x00" * 100)

            mock_voice.synthesize_wav.side_effect = _fake_synth
            mock_load.return_value = mock_voice

            engine.synthesize("hola", voice="default", language="es")

            # Verify the resolved voice was the Spanish one, not English.
            mock_load.assert_called_once_with("es_ES-davefx-medium")

    def test_voice_without_lang_prefix_passes_through(self, tmp_path):
        # Custom voice names that don't follow convention shouldn't trigger
        # validation — we can't tell what language they speak.
        _make_piper_dir(tmp_path, ["custom-voice"])
        engine = PiperEngine(model_dir=tmp_path)
        with patch.object(engine, "_load_voice") as mock_load:
            mock_voice = MagicMock()
            mock_voice.config.sample_rate = 22050

            def _fake_synth(text, wav_file, syn_config=None):
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                wav_file.writeframes(b"\x00\x00" * 100)

            mock_voice.synthesize_wav.side_effect = _fake_synth
            mock_load.return_value = mock_voice
            # Should not raise even though language=es and voice has no prefix.
            engine.synthesize("anything", voice="custom-voice", language="es")


# ─────────────────────────────────────────────────────────────────────────────
# TTSRenderer — language threading
# ─────────────────────────────────────────────────────────────────────────────


class TestRendererLanguage:
    def _show(self, language: str = "en") -> dict:
        return {
            "title": "Test Show",
            "language": language,
            "events": [
                {
                    "global_order": 1,
                    "speaker_name": "Alice",
                    "text": "Hello there.",
                    "event_kind": "utterance",
                },
                {
                    "global_order": 2,
                    "speaker_name": "Bob",
                    "text": "Hi back.",
                    "event_kind": "utterance",
                },
            ],
        }

    def test_default_language_is_en(self, tmp_path):
        engine = _RecordingEngine()
        renderer = TTSRenderer(engine)
        data = self._show()
        del data["language"]  # missing field → default "en"
        renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)
        assert all(c["language"] == "en" for c in engine.calls)

    def test_language_passed_through_to_synthesize(self, tmp_path):
        engine = _RecordingEngine()
        renderer = TTSRenderer(engine)
        renderer.render(
            self._show(language="es"),
            tmp_path / "out.wav",
            format="wav",
            quiet=True,
        )
        assert engine.calls, "expected at least one synthesize call"
        for call in engine.calls:
            assert call["language"] == "es"

    def test_piper_voice_pool_filtered_by_language(self, tmp_path):
        # Build a Piper engine with mixed-language voices.
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _make_piper_dir(models_dir, [
            "en_US-lessac-medium",
            "en_US-ryan-high",
            "es_ES-davefx-medium",
            "es_ES-sharvard-medium",
        ])
        engine = PiperEngine(model_dir=models_dir)
        renderer = TTSRenderer(engine)

        voice_map = renderer._build_voice_map(
            ["Alice", "Bob"], language="es",
        )
        # Both speakers should be assigned Spanish voices only.
        assigned = list(voice_map.values())
        assert all(v.startswith("es_") for v in assigned), assigned

    def test_piper_voice_pool_english_uses_full_catalog(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _make_piper_dir(models_dir, [
            "en_US-lessac-medium",
            "en_US-ryan-high",
            "es_ES-davefx-medium",
        ])
        engine = PiperEngine(model_dir=models_dir)
        renderer = TTSRenderer(engine)

        voice_map = renderer._build_voice_map(
            ["Alice", "Bob"], language="en",
        )
        # English speakers should be assigned English voices.
        assert all(v.startswith("en_") for v in voice_map.values())


# ─────────────────────────────────────────────────────────────────────────────
# supported_languages() — directory scan
# ─────────────────────────────────────────────────────────────────────────────


class TestSupportedLanguagesScan:
    def test_returns_en_when_piper_dir_missing(self, tmp_path, monkeypatch):
        from vox_romana.core import languages as lang_mod
        from vox_romana.core import paths as paths_mod

        monkeypatch.setattr(paths_mod, "PIPER_VOICES_DIR", tmp_path / "nope")
        # Re-import side effects: function reads PIPER_VOICES_DIR lazily,
        # so the monkeypatch is sufficient.
        result = lang_mod.supported_languages()
        assert result == ["en"]

    def test_includes_languages_from_installed_models(self, tmp_path, monkeypatch):
        from vox_romana.core import languages as lang_mod
        from vox_romana.core import paths as paths_mod

        _make_piper_dir(tmp_path, [
            "en_US-lessac-medium",
            "es_ES-davefx-medium",
            "fr_FR-siwis-low",
            "de_DE-thorsten-high",
        ])
        monkeypatch.setattr(paths_mod, "PIPER_VOICES_DIR", tmp_path)
        result = lang_mod.supported_languages()
        assert result == ["de", "en", "es", "fr"]

    def test_unknown_language_codes_filtered_out(self, tmp_path, monkeypatch):
        from vox_romana.core import languages as lang_mod
        from vox_romana.core import paths as paths_mod

        # "xx" is not in LANGUAGE_NAMES, so it should be skipped.
        _make_piper_dir(tmp_path, [
            "en_US-lessac-medium",
            "xx_XX-bogus-medium",
        ])
        monkeypatch.setattr(paths_mod, "PIPER_VOICES_DIR", tmp_path)
        result = lang_mod.supported_languages()
        assert result == ["en"]

    def test_always_includes_en_even_without_piper_models(self, tmp_path, monkeypatch):
        from vox_romana.core import languages as lang_mod
        from vox_romana.core import paths as paths_mod

        # Empty Piper directory — no models at all.
        monkeypatch.setattr(paths_mod, "PIPER_VOICES_DIR", tmp_path)
        result = lang_mod.supported_languages()
        assert "en" in result

    def test_nested_piper_voices_layout(self, tmp_path, monkeypatch):
        # Real piper-voices git clone uses nested dirs:
        #   en/en_US/lessac/medium/en_US-lessac-medium.onnx
        from vox_romana.core import languages as lang_mod
        from vox_romana.core import paths as paths_mod

        nested = tmp_path / "es" / "es_ES" / "davefx" / "medium"
        nested.mkdir(parents=True)
        (nested / "es_ES-davefx-medium.onnx").write_bytes(b"")
        (nested / "es_ES-davefx-medium.onnx.json").write_text("{}")
        monkeypatch.setattr(paths_mod, "PIPER_VOICES_DIR", tmp_path)
        result = lang_mod.supported_languages()
        assert "es" in result
