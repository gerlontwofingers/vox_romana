"""Tests for vox_romana.tts.engines.voice_clone_engine — VoiceCloneEngine."""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# pydub/f5_tts_mlx leave unclosed file handles during GC on CPython 3.12
pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _write_wav(path: Path, sample_rate: int = 24_000, n_samples: int = 2400) -> Path:
    """Write a minimal silent mono WAV so tests have a real file to reference."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return path


def _make_profile(profiles_dir: Path, voice_name: str) -> Path:
    """Create a minimal vc_ profile directory with reference.wav."""
    voice_id = f"vc_{voice_name}"
    profile_dir = profiles_dir / voice_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    _write_wav(profile_dir / "reference.wav")
    return profile_dir


# ─────────────────────────────────────────────────────────────────────────────
# Construction + properties
# ─────────────────────────────────────────────────────────────────────────────


class TestVoiceCloneEngineProperties:
    def test_name_property(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        engine = VoiceCloneEngine(profiles_dir=tmp_path)
        assert engine.name == "voice_clone"

    def test_profiles_dir_stored_as_resolved_path(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        engine = VoiceCloneEngine(profiles_dir=str(tmp_path))
        assert engine._profiles_dir == tmp_path.resolve()


# ─────────────────────────────────────────────────────────────────────────────
# available_voices()
# ─────────────────────────────────────────────────────────────────────────────


class TestAvailableVoices:
    def test_empty_when_dir_missing(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        engine = VoiceCloneEngine(profiles_dir=tmp_path / "nonexistent")
        assert engine.available_voices() == []

    def test_empty_when_no_profiles(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        engine = VoiceCloneEngine(profiles_dir=tmp_path)
        assert engine.available_voices() == []

    def test_returns_voice_ids_for_valid_profiles(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        _make_profile(tmp_path, "alice")
        _make_profile(tmp_path, "bob")
        engine = VoiceCloneEngine(profiles_dir=tmp_path)

        voices = engine.available_voices()
        assert "vc_alice" in voices
        assert "vc_bob" in voices

    def test_skips_dirs_without_vc_prefix(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        # A stray directory without the vc_ prefix should be ignored
        (tmp_path / "af_heart").mkdir()  # Kokoro-style name — must not appear
        _make_profile(tmp_path, "marcus")
        engine = VoiceCloneEngine(profiles_dir=tmp_path)

        voices = engine.available_voices()
        assert "af_heart" not in voices
        assert "vc_marcus" in voices

    def test_skips_profile_missing_reference_wav(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        # Directory with vc_ prefix but no reference.wav
        (tmp_path / "vc_ghost").mkdir()
        _make_profile(tmp_path, "real")
        engine = VoiceCloneEngine(profiles_dir=tmp_path)

        voices = engine.available_voices()
        assert "vc_ghost" not in voices
        assert "vc_real" in voices

    def test_populates_ref_cache(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        _make_profile(tmp_path, "alice")
        engine = VoiceCloneEngine(profiles_dir=tmp_path)
        engine.available_voices()

        assert "vc_alice" in engine._ref_cache
        assert engine._ref_cache["vc_alice"].exists()

    def test_returns_sorted_list(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        for name in ("zebra", "apple", "mango"):
            _make_profile(tmp_path, name)
        engine = VoiceCloneEngine(profiles_dir=tmp_path)

        voices = engine.available_voices()
        assert voices == sorted(voices)


# ─────────────────────────────────────────────────────────────────────────────
# synthesize() — guard-rail validation
# ─────────────────────────────────────────────────────────────────────────────


class TestSynthesizeValidation:
    def test_raises_on_empty_voice(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        engine = VoiceCloneEngine(profiles_dir=tmp_path)
        with pytest.raises(ValueError, match="voice must be a registered"):
            engine.synthesize("hello", voice="")

    def test_raises_on_non_vc_prefix(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        engine = VoiceCloneEngine(profiles_dir=tmp_path)
        with pytest.raises(ValueError, match="does not start with"):
            engine.synthesize("hello", voice="af_heart")

    def test_raises_on_unknown_voice_id(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        engine = VoiceCloneEngine(profiles_dir=tmp_path)
        with pytest.raises(ValueError, match="no profile found"):
            engine.synthesize("hello", voice="vc_unknown_voice")


# ─────────────────────────────────────────────────────────────────────────────
# synthesize() — f5-tts backend mocked
# ─────────────────────────────────────────────────────────────────────────────


class TestSynthesizeWithMockedF5:
    """Mock f5_tts_mlx.generate so tests run without the real GPU library."""

    def _mock_f5_generate(self, out_path_arg_index: int = 2):
        """Return a side-effect fn that writes a silent WAV to the output path."""

        def _side_effect(**kwargs):
            out = Path(kwargs["output_path"])
            _write_wav(out)

        return _side_effect

    def test_returns_synthesized_audio(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import (
            VOICE_CLONE_SAMPLE_RATE,
            VoiceCloneEngine,
        )

        _make_profile(tmp_path, "marcus")
        engine = VoiceCloneEngine(profiles_dir=tmp_path)

        fake_f5 = MagicMock(side_effect=self._mock_f5_generate())
        fake_module = MagicMock()
        fake_module.generate = fake_f5

        with patch.dict("sys.modules", {"f5_tts_mlx": fake_module, "f5_tts_mlx.generate": fake_module}):
            result = engine.synthesize("Hello world", voice="vc_marcus")

        from vox_romana.tts.engine import SynthesizedAudio

        assert isinstance(result, SynthesizedAudio)
        assert result.sample_rate == VOICE_CLONE_SAMPLE_RATE
        assert len(result.audio) > 0

    def test_passes_ref_wav_to_f5(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        _make_profile(tmp_path, "alice")
        engine = VoiceCloneEngine(profiles_dir=tmp_path)
        expected_ref = str(tmp_path / "vc_alice" / "reference.wav")

        calls_captured: list[dict] = []

        def _capturing_f5(**kwargs):
            calls_captured.append(kwargs)
            _write_wav(Path(kwargs["output_path"]))

        fake_module = MagicMock()
        fake_module.generate = _capturing_f5

        with patch.dict("sys.modules", {"f5_tts_mlx": fake_module, "f5_tts_mlx.generate": fake_module}):
            engine.synthesize("Test text", voice="vc_alice")

        assert len(calls_captured) == 1
        assert calls_captured[0]["ref_audio_path"] == expected_ref
        assert calls_captured[0]["generation_text"] == "Test text"

    def test_f5_import_error_raises_runtime_error(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        _make_profile(tmp_path, "bob")
        engine = VoiceCloneEngine(profiles_dir=tmp_path)

        import builtins
        real_import = builtins.__import__

        def _no_f5(name, *args, **kwargs):
            if name.startswith("f5_tts_mlx"):
                raise ImportError("No module named 'f5_tts_mlx'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_no_f5):
            with pytest.raises(RuntimeError, match="f5-tts-mlx"):
                engine.synthesize("hello", voice="vc_bob")

    def test_default_nfe_and_cfg_forwarded(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        _make_profile(tmp_path, "carol")
        engine = VoiceCloneEngine(profiles_dir=tmp_path)

        calls_captured: list[dict] = []

        def _capturing_f5(**kwargs):
            calls_captured.append(kwargs)
            _write_wav(Path(kwargs["output_path"]))

        fake_module = MagicMock()
        fake_module.generate = _capturing_f5

        with patch.dict("sys.modules", {"f5_tts_mlx": fake_module, "f5_tts_mlx.generate": fake_module}):
            engine.synthesize("Text", voice="vc_carol")

        assert calls_captured[0]["nfe_step"] == 32
        assert calls_captured[0]["cfg_strength"] == pytest.approx(2.0)

    def test_emotion_accepted_and_ignored(self, tmp_path):
        """emotion kwarg must not cause an error even though f5 ignores it."""
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        _make_profile(tmp_path, "dave")
        engine = VoiceCloneEngine(profiles_dir=tmp_path)

        fake_module = MagicMock()
        fake_module.generate = lambda **kwargs: _write_wav(Path(kwargs["output_path"]))

        with patch.dict("sys.modules", {"f5_tts_mlx": fake_module, "f5_tts_mlx.generate": fake_module}):
            result = engine.synthesize("Hello", voice="vc_dave", emotion="speak warmly")

        from vox_romana.tts.engine import SynthesizedAudio
        assert isinstance(result, SynthesizedAudio)


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_reference — cache refresh
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveReference:
    def test_returns_correct_path_after_registration(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        _make_profile(tmp_path, "eve")
        engine = VoiceCloneEngine(profiles_dir=tmp_path)

        ref = engine._resolve_reference("vc_eve")
        assert ref == tmp_path / "vc_eve" / "reference.wav"
        assert ref.exists()

    def test_raises_for_unknown_voice_id(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        engine = VoiceCloneEngine(profiles_dir=tmp_path)
        with pytest.raises(ValueError, match="no profile found"):
            engine._resolve_reference("vc_ghost")

    def test_refreshes_cache_when_profile_added_after_init(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        engine = VoiceCloneEngine(profiles_dir=tmp_path)
        # At this point vc_frank doesn't exist
        assert "vc_frank" not in engine._ref_cache

        # Now register the profile
        _make_profile(tmp_path, "frank")
        # _resolve_reference should pick it up via available_voices() refresh
        ref = engine._resolve_reference("vc_frank")
        assert ref.exists()

    def test_stale_cache_entry_triggers_rescan(self, tmp_path):
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        _make_profile(tmp_path, "grace")
        engine = VoiceCloneEngine(profiles_dir=tmp_path)
        engine.available_voices()  # warm cache

        # Simulate stale cache: point to a non-existent path
        engine._ref_cache["vc_grace"] = tmp_path / "vc_grace" / "gone.wav"

        # Should detect stale entry, rescan, find the real reference.wav
        ref = engine._resolve_reference("vc_grace")
        assert ref.name == "reference.wav"
        assert ref.exists()
