"""Tests for vox_romana.voice_prep — cleaner.py and profiler.py."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# pydub's AudioSegment.from_file() leaves an internal fd unclosed during GC
# on CPython 3.12 (upstream pydub bug — not in our code).  Suppress just for
# this file so conftest's broad error-on-warning rule doesn't kill the suite.
pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_wav(path: Path, duration_ms: int = 2000, sr: int = 24_000) -> Path:
    """Write a minimal valid WAV file using pydub so tests have real audio."""
    from pydub import AudioSegment
    from pydub.generators import Sine

    tone = Sine(440).to_audio_segment(duration=duration_ms)
    mono = tone.set_channels(1).set_frame_rate(sr).set_sample_width(2)
    fh = mono.export(str(path), format="wav")
    fh.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# cleaner.clean_audio — basic pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TestCleanAudioBasic:
    def test_returns_path_in_output_dir(self, tmp_path):
        from vox_romana.voice_prep.cleaner import clean_audio

        src = tmp_path / "voice.wav"
        _make_wav(src)
        out_dir = tmp_path / "out"

        result = clean_audio(src, skip_demucs=True, output_dir=out_dir)

        assert result.exists()
        assert result.parent == out_dir
        assert result.name == "voice_cleaned.wav"

    def test_default_output_dir_is_vp_cleaned_subdir(self, tmp_path):
        from vox_romana.voice_prep.cleaner import clean_audio

        src = tmp_path / "myvoice.wav"
        _make_wav(src)

        result = clean_audio(src, skip_demucs=True)

        assert result.parent == tmp_path / "_vp_cleaned"
        assert result.exists()

    def test_output_is_mono_24k_wav(self, tmp_path):
        from pydub import AudioSegment

        from vox_romana.voice_prep.cleaner import clean_audio

        src = tmp_path / "stereo.wav"
        # Create a stereo 44.1 kHz source to verify resampling
        from pydub.generators import Sine

        tone = Sine(440).to_audio_segment(duration=1000)
        stereo = tone.set_channels(2).set_frame_rate(44_100)
        fh = stereo.export(str(src), format="wav")
        fh.close()

        result = clean_audio(src, skip_demucs=True, output_dir=tmp_path / "out")

        seg = AudioSegment.from_file(str(result))
        assert seg.channels == 1
        assert seg.frame_rate == 24_000

    def test_raises_file_not_found_for_missing_source(self, tmp_path):
        from vox_romana.voice_prep.cleaner import clean_audio

        with pytest.raises(FileNotFoundError, match="source not found"):
            clean_audio(tmp_path / "nonexistent.wav", skip_demucs=True)


# ─────────────────────────────────────────────────────────────────────────────
# cleaner — demucs integration paths
# ─────────────────────────────────────────────────────────────────────────────


class TestCleanAudioDemucs:
    def test_skip_demucs_bypasses_separation(self, tmp_path):
        """With skip_demucs=True, _stage_demucs should never be called."""
        from vox_romana.voice_prep import cleaner

        src = tmp_path / "voice.wav"
        _make_wav(src)

        with patch.object(cleaner, "_stage_demucs", wraps=cleaner._stage_demucs) as mock_demucs:
            cleaner.clean_audio(src, skip_demucs=True, output_dir=tmp_path / "out")
            # _stage_demucs is still called but receives skip=True
            mock_demucs.assert_called_once()
            _, kwargs = mock_demucs.call_args
            assert kwargs["skip"] is True

    def test_demucs_import_error_falls_back_gracefully(self, tmp_path):
        """If demucs is not installed, pipeline continues with raw audio."""
        from vox_romana.voice_prep.cleaner import clean_audio

        src = tmp_path / "voice.wav"
        _make_wav(src)

        # Simulate demucs not installed by patching the import inside _stage_demucs
        import builtins
        real_import = builtins.__import__

        def _no_demucs(name, *args, **kwargs):
            if name.startswith("demucs"):
                raise ImportError("No module named 'demucs'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_no_demucs):
            result = clean_audio(src, skip_demucs=False, output_dir=tmp_path / "out")

        assert result.exists()

    def test_demucs_runtime_failure_falls_back_gracefully(self, tmp_path):
        """If demucs.separate.main() raises, pipeline continues with raw audio."""
        from vox_romana.voice_prep import cleaner

        src = tmp_path / "voice.wav"
        _make_wav(src)

        # Patch demucs.separate so the import succeeds but main() raises
        fake_demucs_sep = MagicMock()
        fake_demucs_sep.main.side_effect = RuntimeError("demucs exploded")
        fake_torch = MagicMock()

        with patch.dict(
            "sys.modules",
            {"demucs": MagicMock(), "demucs.separate": fake_demucs_sep, "torch": fake_torch},
        ):
            result = cleaner.clean_audio(
                src, skip_demucs=False, output_dir=tmp_path / "out"
            )

        assert result.exists()


# ─────────────────────────────────────────────────────────────────────────────
# cleaner — _stage_normalise
# ─────────────────────────────────────────────────────────────────────────────


class TestStageNormalise:
    def test_normalise_adjusts_dbfs(self, tmp_path):
        """After normalisation, the output dBFS should be near _TARGET_DBFS."""
        from pydub import AudioSegment

        from vox_romana.voice_prep.cleaner import _TARGET_DBFS, _stage_normalise

        src = tmp_path / "input.wav"
        _make_wav(src)

        out = _stage_normalise(src, tmp_dir=tmp_path)
        seg = AudioSegment.from_file(str(out))

        # Allow ±2 dB tolerance (silence trim can shift level slightly)
        assert abs(seg.dBFS - _TARGET_DBFS) < 2.0

    def test_normalise_output_is_wav(self, tmp_path):
        from vox_romana.voice_prep.cleaner import _stage_normalise

        src = tmp_path / "input.wav"
        _make_wav(src)
        out = _stage_normalise(src, tmp_dir=tmp_path)
        assert out.suffix == ".wav"


# ─────────────────────────────────────────────────────────────────────────────
# profiler.register_voice
# ─────────────────────────────────────────────────────────────────────────────


class TestRegisterVoice:
    def _call(self, cleaned_wav: Path, profiles_dir: Path, **kwargs) -> Path:
        from vox_romana.voice_prep.profiler import register_voice

        defaults = dict(
            gender="male",
            tone="deep, measured",
            best_for="narration",
            tags=["deep", "male", "measured"],
        )
        defaults.update(kwargs)
        return register_voice(
            cleaned_wav,
            "marcus_deep",
            profiles_dir=profiles_dir,
            **defaults,
        )

    def test_creates_profile_directory(self, tmp_path):
        wav = tmp_path / "ref.wav"
        _make_wav(wav)
        profile = self._call(wav, tmp_path / "profiles")
        assert profile.is_dir()
        assert profile.name == "vc_marcus_deep"

    def test_reference_wav_copied(self, tmp_path):
        wav = tmp_path / "ref.wav"
        _make_wav(wav)
        profile = self._call(wav, tmp_path / "profiles")
        assert (profile / "reference.wav").exists()

    def test_metadata_json_contents(self, tmp_path):
        wav = tmp_path / "ref.wav"
        _make_wav(wav)
        profile = self._call(
            wav,
            tmp_path / "profiles",
            gender="female",
            tone="warm, bright",
            best_for="narration, audiobook",
            tags=["warm", "female", "bright"],
        )
        meta = json.loads((profile / "metadata.json").read_text())

        assert meta["voice_id"] == "vc_marcus_deep"
        assert meta["voice_name"] == "marcus_deep"
        assert meta["gender"] == "female"
        assert meta["tone"] == "warm, bright"
        assert meta["best_for"] == "narration, audiobook"
        assert meta["tags"] == ["warm", "female", "bright"]
        assert meta["source_file"] == "ref.wav"
        assert "created_at" in meta

    def test_metadata_round_trips_via_load_profile_metadata(self, tmp_path):
        from vox_romana.voice_prep.profiler import load_profile_metadata

        wav = tmp_path / "ref.wav"
        _make_wav(wav)
        profile = self._call(wav, tmp_path / "profiles")
        meta = load_profile_metadata(profile)
        assert meta["voice_id"] == "vc_marcus_deep"

    def test_collision_raises_without_force(self, tmp_path):
        from vox_romana.voice_prep.profiler import register_voice

        wav = tmp_path / "ref.wav"
        _make_wav(wav)
        profiles = tmp_path / "profiles"

        self._call(wav, profiles)

        with pytest.raises(ValueError, match="already exists"):
            self._call(wav, profiles)

    def test_collision_overwrite_with_force(self, tmp_path):
        from vox_romana.voice_prep.profiler import register_voice

        wav = tmp_path / "ref.wav"
        _make_wav(wav)
        profiles = tmp_path / "profiles"

        self._call(wav, profiles)
        # Modify metadata to confirm force overwrites
        profile_dir = profiles / "vc_marcus_deep"
        meta_path = profile_dir / "metadata.json"
        old = json.loads(meta_path.read_text())
        old["tone"] = "SENTINEL"
        meta_path.write_text(json.dumps(old))

        # Now force-overwrite
        profile2 = self._call(wav, profiles, force=True)
        new_meta = json.loads((profile2 / "metadata.json").read_text())
        # tone should be restored to the default ("deep, measured")
        assert new_meta["tone"] == "deep, measured"

    def test_invalid_name_raises(self, tmp_path):
        from vox_romana.voice_prep.profiler import register_voice

        wav = tmp_path / "ref.wav"
        _make_wav(wav)

        with pytest.raises(ValueError, match="invalid"):
            register_voice(
                wav,
                "Marcus Deep",  # spaces not allowed
                profiles_dir=tmp_path / "profiles",
                gender="male",
                tone="deep",
                best_for="narration",
                tags=[],
            )

    def test_empty_name_raises(self, tmp_path):
        from vox_romana.voice_prep.profiler import register_voice

        wav = tmp_path / "ref.wav"
        _make_wav(wav)

        with pytest.raises(ValueError, match="must not be empty"):
            register_voice(
                wav,
                "",
                profiles_dir=tmp_path / "profiles",
                gender="male",
                tone="deep",
                best_for="narration",
                tags=[],
            )

    def test_missing_wav_raises(self, tmp_path):
        from vox_romana.voice_prep.profiler import register_voice

        with pytest.raises(FileNotFoundError, match="cleaned_wav not found"):
            register_voice(
                tmp_path / "ghost.wav",
                "ghost_voice",
                profiles_dir=tmp_path / "profiles",
                gender="male",
                tone="eerie",
                best_for="horror",
                tags=[],
            )

    def test_vc_prefix_in_voice_id(self, tmp_path):
        from vox_romana.voice_prep.profiler import VOICE_CLONE_PREFIX

        assert VOICE_CLONE_PREFIX == "vc_"


# ─────────────────────────────────────────────────────────────────────────────
# profiler.list_profiles
# ─────────────────────────────────────────────────────────────────────────────


class TestListProfiles:
    def test_empty_dir_returns_empty_list(self, tmp_path):
        from vox_romana.voice_prep.profiler import list_profiles

        assert list_profiles(tmp_path / "nonexistent") == []

    def test_lists_registered_profiles(self, tmp_path):
        from vox_romana.voice_prep.profiler import list_profiles, register_voice

        profiles_dir = tmp_path / "profiles"

        for name, gender in [("alice_warm", "female"), ("bob_deep", "male")]:
            wav = tmp_path / f"{name}.wav"
            _make_wav(wav)
            register_voice(
                wav,
                name,
                profiles_dir=profiles_dir,
                gender=gender,
                tone="neutral",
                best_for="narration",
                tags=[gender],
            )

        result = list_profiles(profiles_dir)
        assert len(result) == 2
        ids = {m["voice_id"] for m in result}
        assert ids == {"vc_alice_warm", "vc_bob_deep"}

    def test_malformed_profile_skipped(self, tmp_path):
        from vox_romana.voice_prep.profiler import list_profiles, register_voice

        profiles_dir = tmp_path / "profiles"
        wav = tmp_path / "voice.wav"
        _make_wav(wav)
        register_voice(
            wav,
            "good_voice",
            profiles_dir=profiles_dir,
            gender="male",
            tone="neutral",
            best_for="narration",
            tags=[],
        )

        # Inject a broken profile (no metadata.json)
        broken = profiles_dir / "vc_broken"
        broken.mkdir()

        result = list_profiles(profiles_dir)
        assert len(result) == 1
        assert result[0]["voice_id"] == "vc_good_voice"
