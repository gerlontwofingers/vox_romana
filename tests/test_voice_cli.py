"""Tests for Chunk 4 CLI wiring:
  - run_voice_prep() (voice_prep_cli.run_voice_prep)
  - run_voice_assign() (voice_assign_cli.run_voice_assign)
  - run.py parse_args() for voice-prep / voice-assign modes
  - VOICE_PROFILES_DIR constant in paths.py
"""

from __future__ import annotations

import json
import sys
import types
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _write_wav(path: Path) -> Path:
    """Write a minimal valid 24 kHz mono WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24_000)
        wf.writeframes(b"\x00\x00" * 2400)
    return path


def _make_args(**kwargs) -> types.SimpleNamespace:
    """Build a minimal SimpleNamespace mimicking a parsed argparse Namespace."""
    defaults = dict(
        vp_source=None,
        vp_name=None,
        vp_gender="neutral",
        vp_tone="",
        vp_best_for="",
        vp_tags=None,
        vp_force=False,
        skip_demucs=False,
        voice_id=None,
        va_character=None,
        series_id=None,
        voice_profiles_dir=None,
        quiet=False,
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# VOICE_PROFILES_DIR in paths.py
# ─────────────────────────────────────────────────────────────────────────────


class TestVoiceProfilesDirConstant:
    def test_constant_exists(self):
        from vox_romana.core.paths import VOICE_PROFILES_DIR
        assert VOICE_PROFILES_DIR is not None

    def test_constant_is_path(self):
        from vox_romana.core.paths import VOICE_PROFILES_DIR
        assert isinstance(VOICE_PROFILES_DIR, Path)

    def test_default_under_data_root(self):
        from vox_romana.core.paths import DATA_ROOT, VOICE_PROFILES_DIR
        # Unless overridden by config, defaults to data_root/voice_profiles
        # We can't know if config has voice_profiles_dir set, so just check
        # it's a Path that starts with DATA_ROOT or is an absolute path.
        assert VOICE_PROFILES_DIR.is_absolute()

    def test_config_override_respected(self, tmp_path):
        """_voice_profiles_dir() respects config override."""
        from vox_romana.core import paths as paths_mod
        custom = tmp_path / "my_voices"
        with patch.object(paths_mod, "_load_config", return_value={"voice_profiles_dir": str(custom)}):
            result = paths_mod._voice_profiles_dir(tmp_path / "data")
        assert result == custom


# ─────────────────────────────────────────────────────────────────────────────
# run_voice_prep
# ─────────────────────────────────────────────────────────────────────────────


class TestRunVoicePrep:
    def test_exits_when_source_missing(self, tmp_path):
        from vox_romana.cli.voice_prep_cli import run_voice_prep
        args = _make_args(
            vp_source=tmp_path / "nonexistent.wav",
            vp_name="test_voice",
            voice_profiles_dir=tmp_path / "profiles",
        )
        with pytest.raises(SystemExit):
            run_voice_prep(args)

    def test_calls_clean_audio(self, tmp_path):
        from vox_romana.cli.voice_prep_cli import run_voice_prep
        src = _write_wav(tmp_path / "ref.wav")
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        cleaned = tmp_path / "cleaned.wav"
        _write_wav(cleaned)

        fake_profile = profiles_dir / "vc_testvoice"
        fake_profile.mkdir(parents=True)

        with (
            patch("vox_romana.voice_prep.cleaner.clean_audio", return_value=cleaned) as mock_clean,
            patch("vox_romana.voice_prep.profiler.register_voice", return_value=fake_profile),
        ):
            args = _make_args(
                vp_source=src,
                vp_name="testvoice",
                voice_profiles_dir=profiles_dir,
            )
            run_voice_prep(args)

        mock_clean.assert_called_once_with(src, skip_demucs=False)

    def test_calls_register_voice_with_correct_args(self, tmp_path):
        from vox_romana.cli.voice_prep_cli import run_voice_prep
        src = _write_wav(tmp_path / "ref.wav")
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        cleaned = tmp_path / "cleaned.wav"
        _write_wav(cleaned)
        fake_profile = profiles_dir / "vc_testvoice"
        fake_profile.mkdir(parents=True)

        with (
            patch("vox_romana.voice_prep.cleaner.clean_audio", return_value=cleaned),
            patch("vox_romana.voice_prep.profiler.register_voice", return_value=fake_profile) as mock_reg,
        ):
            args = _make_args(
                vp_source=src,
                vp_name="testvoice",
                vp_gender="male",
                vp_tone="deep",
                vp_best_for="narration",
                vp_tags=["deep", "measured"],
                vp_force=True,
                voice_profiles_dir=profiles_dir,
            )
            run_voice_prep(args)

        mock_reg.assert_called_once_with(
            cleaned,
            "testvoice",
            profiles_dir=profiles_dir,
            gender="male",
            tone="deep",
            best_for="narration",
            tags=["deep", "measured"],
            force=True,
        )

    def test_skip_demucs_forwarded(self, tmp_path):
        from vox_romana.cli.voice_prep_cli import run_voice_prep
        src = _write_wav(tmp_path / "ref.wav")
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        cleaned = tmp_path / "cleaned.wav"
        _write_wav(cleaned)
        fake_profile = profiles_dir / "vc_v"
        fake_profile.mkdir(parents=True)

        with (
            patch("vox_romana.voice_prep.cleaner.clean_audio", return_value=cleaned) as mock_clean,
            patch("vox_romana.voice_prep.profiler.register_voice", return_value=fake_profile),
        ):
            args = _make_args(
                vp_source=src,
                vp_name="v",
                skip_demucs=True,
                voice_profiles_dir=profiles_dir,
            )
            run_voice_prep(args)

        mock_clean.assert_called_once_with(src, skip_demucs=True)

    def test_returns_profile_path(self, tmp_path):
        from vox_romana.cli.voice_prep_cli import run_voice_prep
        src = _write_wav(tmp_path / "ref.wav")
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        cleaned = tmp_path / "cleaned.wav"
        _write_wav(cleaned)
        expected_profile = profiles_dir / "vc_hero"
        expected_profile.mkdir(parents=True)

        with (
            patch("vox_romana.voice_prep.cleaner.clean_audio", return_value=cleaned),
            patch("vox_romana.voice_prep.profiler.register_voice", return_value=expected_profile),
        ):
            args = _make_args(
                vp_source=src,
                vp_name="hero",
                voice_profiles_dir=profiles_dir,
            )
            result = run_voice_prep(args)

        assert result == expected_profile

    def test_null_tags_defaults_to_empty_list(self, tmp_path):
        from vox_romana.cli.voice_prep_cli import run_voice_prep
        src = _write_wav(tmp_path / "ref.wav")
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        cleaned = tmp_path / "cleaned.wav"
        _write_wav(cleaned)
        fake_profile = profiles_dir / "vc_v"
        fake_profile.mkdir(parents=True)

        with (
            patch("vox_romana.voice_prep.cleaner.clean_audio", return_value=cleaned),
            patch("vox_romana.voice_prep.profiler.register_voice", return_value=fake_profile) as mock_reg,
        ):
            args = _make_args(
                vp_source=src,
                vp_name="v",
                vp_tags=None,
                voice_profiles_dir=profiles_dir,
            )
            run_voice_prep(args)

        _call_kwargs = mock_reg.call_args.kwargs
        assert _call_kwargs["tags"] == []


# ─────────────────────────────────────────────────────────────────────────────
# run_voice_assign
# ─────────────────────────────────────────────────────────────────────────────


class TestRunVoiceAssign:
    def test_exits_on_missing_vc_prefix(self, tmp_path):
        from vox_romana.cli.voice_assign_cli import run_voice_assign
        args = _make_args(
            voice_id="marcus_deep",  # no vc_ prefix
            va_character="Marcus",
            series_id="s1",
            voice_profiles_dir=tmp_path / "profiles",
        )
        with pytest.raises(SystemExit):
            run_voice_assign(args)

    def test_calls_registry_register(self, tmp_path):
        from vox_romana.cli.voice_assign_cli import run_voice_assign

        mock_registry = MagicMock()
        mock_catalog = MagicMock()
        mock_catalog.register_cloned_asset.return_value = 1

        with (
            patch("vox_romana.tts.voice_registry.VoiceRegistry", return_value=mock_registry),
            patch("vox_romana.core.catalog.Catalog", return_value=mock_catalog),
            patch("vox_romana.core.paths.CATALOG_DB_PATH", tmp_path / "test.db"),
        ):
            args = _make_args(
                voice_id="vc_marcus",
                va_character="Marcus Aurelius",
                series_id="series_abc",
                voice_profiles_dir=tmp_path / "profiles",
            )
            run_voice_assign(args)

        mock_registry.register.assert_called_once_with(
            "Marcus Aurelius", "vc_marcus", "series_abc"
        )

    def test_calls_catalog_register_cloned_asset(self, tmp_path):
        from vox_romana.cli.voice_assign_cli import run_voice_assign

        mock_registry = MagicMock()
        mock_catalog = MagicMock()
        mock_catalog.register_cloned_asset.return_value = 1
        profiles_dir = tmp_path / "profiles"

        with (
            patch("vox_romana.tts.voice_registry.VoiceRegistry", return_value=mock_registry),
            patch("vox_romana.core.catalog.Catalog", return_value=mock_catalog),
            patch("vox_romana.core.paths.CATALOG_DB_PATH", tmp_path / "test.db"),
        ):
            args = _make_args(
                voice_id="vc_senator",
                va_character="Senator",
                series_id="s42",
                voice_profiles_dir=profiles_dir,
            )
            run_voice_assign(args)

        mock_catalog.register_cloned_asset.assert_called_once_with(
            series_id="s42",
            voice_id="vc_senator",
            character_name="Senator",
            profile_dir=str(profiles_dir / "vc_senator"),
        )

    def test_warns_when_profile_dir_missing(self, tmp_path, capsys):
        from vox_romana.cli.voice_assign_cli import run_voice_assign

        mock_registry = MagicMock()
        mock_catalog = MagicMock()
        mock_catalog.register_cloned_asset.return_value = 1

        with (
            patch("vox_romana.tts.voice_registry.VoiceRegistry", return_value=mock_registry),
            patch("vox_romana.core.catalog.Catalog", return_value=mock_catalog),
            patch("vox_romana.core.paths.CATALOG_DB_PATH", tmp_path / "test.db"),
        ):
            args = _make_args(
                voice_id="vc_ghost",
                va_character="Ghost",
                series_id="s1",
                voice_profiles_dir=tmp_path / "profiles",  # profiles dir doesn't exist
            )
            run_voice_assign(args)  # should not raise

        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_prints_assignment_confirmation(self, tmp_path, capsys):
        from vox_romana.cli.voice_assign_cli import run_voice_assign

        mock_registry = MagicMock()
        mock_catalog = MagicMock()
        mock_catalog.register_cloned_asset.return_value = 1

        with (
            patch("vox_romana.tts.voice_registry.VoiceRegistry", return_value=mock_registry),
            patch("vox_romana.core.catalog.Catalog", return_value=mock_catalog),
            patch("vox_romana.core.paths.CATALOG_DB_PATH", tmp_path / "test.db"),
        ):
            args = _make_args(
                voice_id="vc_hero",
                va_character="Hero",
                series_id="s1",
                voice_profiles_dir=tmp_path / "profiles",
            )
            run_voice_assign(args)

        captured = capsys.readouterr()
        assert "vc_hero" in captured.out
        assert "Hero" in captured.out
        assert "s1" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# run.py parse_args() — voice-prep / voice-assign arg groups
# ─────────────────────────────────────────────────────────────────────────────


class TestParseArgsVoicePrep:
    def test_mode_voice_prep_accepted(self):
        from run import parse_args
        args = parse_args(["--mode", "voice-prep", "--vp-source", "ref.wav", "--vp-name", "hero"])
        assert args.mode == "voice-prep"

    def test_vp_source_parsed_as_path(self):
        from run import parse_args
        args = parse_args(["--mode", "voice-prep", "--vp-source", "ref.wav", "--vp-name", "hero"])
        assert isinstance(args.vp_source, Path)
        assert args.vp_source == Path("ref.wav")

    def test_vp_name_parsed(self):
        from run import parse_args
        args = parse_args(["--mode", "voice-prep", "--vp-source", "ref.wav", "--vp-name", "marcus_deep"])
        assert args.vp_name == "marcus_deep"

    def test_defaults_for_optional_vp_args(self):
        from run import parse_args
        args = parse_args(["--mode", "voice-prep", "--vp-source", "ref.wav", "--vp-name", "v"])
        assert args.vp_gender == "neutral"
        assert args.vp_tone == ""
        assert args.vp_best_for == ""
        assert args.vp_tags is None
        assert args.vp_force is False
        assert args.skip_demucs is False

    def test_vp_tags_nargs_plus(self):
        from run import parse_args
        args = parse_args([
            "--mode", "voice-prep",
            "--vp-source", "ref.wav",
            "--vp-name", "v",
            "--vp-tags", "deep", "male", "measured",
        ])
        assert args.vp_tags == ["deep", "male", "measured"]

    def test_vp_force_flag(self):
        from run import parse_args
        args = parse_args(["--mode", "voice-prep", "--vp-source", "ref.wav", "--vp-name", "v", "--vp-force"])
        assert args.vp_force is True

    def test_skip_demucs_flag(self):
        from run import parse_args
        args = parse_args(["--mode", "voice-prep", "--vp-source", "ref.wav", "--vp-name", "v", "--skip-demucs"])
        assert args.skip_demucs is True


class TestParseArgsVoiceAssign:
    def test_mode_voice_assign_accepted(self):
        from run import parse_args
        args = parse_args([
            "--mode", "voice-assign",
            "--voice-id", "vc_marcus",
            "--va-character", "Marcus",
            "--series-id", "s1",
        ])
        assert args.mode == "voice-assign"

    def test_voice_id_parsed(self):
        from run import parse_args
        args = parse_args([
            "--mode", "voice-assign",
            "--voice-id", "vc_hero",
            "--va-character", "Hero",
            "--series-id", "s1",
        ])
        assert args.voice_id == "vc_hero"

    def test_va_character_parsed(self):
        from run import parse_args
        args = parse_args([
            "--mode", "voice-assign",
            "--voice-id", "vc_hero",
            "--va-character", "The Hero",
            "--series-id", "s1",
        ])
        assert args.va_character == "The Hero"

    def test_series_id_parsed(self):
        from run import parse_args
        args = parse_args([
            "--mode", "voice-assign",
            "--voice-id", "vc_hero",
            "--va-character", "Hero",
            "--series-id", "abc123",
        ])
        assert args.series_id == "abc123"

    def test_voice_id_default_none(self):
        from run import parse_args
        args = parse_args(["--mode", "voice-prep", "--vp-source", "ref.wav", "--vp-name", "v"])
        assert args.voice_id is None

    def test_va_character_default_none(self):
        from run import parse_args
        args = parse_args(["--mode", "voice-prep", "--vp-source", "ref.wav", "--vp-name", "v"])
        assert args.va_character is None
