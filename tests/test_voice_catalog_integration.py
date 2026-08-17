"""Tests for Chunk 3 catalog integration:
  - voice_catalog.load_cloned_voice_catalog()
  - multi_engine.MultiEngine.from_engine_names() with voice_clone
  - catalog.Catalog.register_cloned_asset() / list_voice_profiles()
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _write_wav(path: Path) -> Path:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24_000)
        wf.writeframes(b"\x00\x00" * 2400)
    return path


def _make_profile(
    profiles_dir: Path,
    voice_name: str,
    *,
    gender: str = "male",
    tone: str = "deep",
    best_for: str = "narration",
    tags: list[str] | None = None,
    accent: str = "american",
    age: str = "middle",
) -> Path:
    voice_id = f"vc_{voice_name}"
    d = profiles_dir / voice_id
    d.mkdir(parents=True, exist_ok=True)
    _write_wav(d / "reference.wav")
    meta = {
        "voice_id": voice_id,
        "voice_name": voice_name,
        "gender": gender,
        "accent": accent,
        "age": age,
        "tone": tone,
        "best_for": best_for,
        "tags": tags or [gender, tone],
        "created_at": "2026-05-09T00:00:00+00:00",
        "source_file": "ref.wav",
    }
    (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# load_cloned_voice_catalog
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadClonedVoiceCatalog:
    def test_empty_when_dir_missing(self, tmp_path):
        from vox_romana.tts.voice_catalog import load_cloned_voice_catalog

        result = load_cloned_voice_catalog(tmp_path / "nonexistent")
        assert result == {}

    def test_empty_when_no_profiles(self, tmp_path):
        from vox_romana.tts.voice_catalog import load_cloned_voice_catalog

        result = load_cloned_voice_catalog(tmp_path)
        assert result == {}

    def test_returns_descriptor_for_valid_profile(self, tmp_path):
        from vox_romana.tts.voice_catalog import load_cloned_voice_catalog

        _make_profile(tmp_path, "marcus", gender="male", tone="deep", tags=["deep", "male"])
        result = load_cloned_voice_catalog(tmp_path)

        assert "vc_marcus" in result
        desc = result["vc_marcus"]
        assert desc.voice_id == "vc_marcus"
        assert desc.gender == "male"
        assert desc.tone == "deep"
        assert desc.engine == "voice_clone"
        assert "deep" in desc.tags

    def test_tags_stored_as_tuple(self, tmp_path):
        from vox_romana.tts.voice_catalog import load_cloned_voice_catalog

        _make_profile(tmp_path, "alice", tags=["warm", "female"])
        result = load_cloned_voice_catalog(tmp_path)
        assert isinstance(result["vc_alice"].tags, tuple)

    def test_skips_dirs_without_vc_prefix(self, tmp_path):
        from vox_romana.tts.voice_catalog import load_cloned_voice_catalog

        (tmp_path / "af_heart").mkdir()
        _make_profile(tmp_path, "bob")
        result = load_cloned_voice_catalog(tmp_path)
        assert "af_heart" not in result
        assert "vc_bob" in result

    def test_skips_profile_with_missing_metadata(self, tmp_path):
        from vox_romana.tts.voice_catalog import load_cloned_voice_catalog

        # vc_ dir but no metadata.json
        (tmp_path / "vc_ghost").mkdir()
        _write_wav(tmp_path / "vc_ghost" / "reference.wav")
        _make_profile(tmp_path, "real")
        result = load_cloned_voice_catalog(tmp_path)
        assert "vc_ghost" not in result
        assert "vc_real" in result

    def test_skips_malformed_json(self, tmp_path):
        from vox_romana.tts.voice_catalog import load_cloned_voice_catalog

        bad = tmp_path / "vc_bad"
        bad.mkdir()
        (bad / "metadata.json").write_text("{not json", encoding="utf-8")
        _make_profile(tmp_path, "good")
        result = load_cloned_voice_catalog(tmp_path)
        assert "vc_bad" not in result
        assert "vc_good" in result

    def test_engine_field_is_voice_clone(self, tmp_path):
        from vox_romana.tts.voice_catalog import load_cloned_voice_catalog

        _make_profile(tmp_path, "clara")
        desc = load_cloned_voice_catalog(tmp_path)["vc_clara"]
        assert desc.engine == "voice_clone"

    def test_multiple_profiles_all_returned(self, tmp_path):
        from vox_romana.tts.voice_catalog import load_cloned_voice_catalog

        for name in ("alice", "bob", "carol"):
            _make_profile(tmp_path, name)
        result = load_cloned_voice_catalog(tmp_path)
        assert set(result) == {"vc_alice", "vc_bob", "vc_carol"}

    def test_merge_into_unified_catalog(self, tmp_path):
        """Cloned voices can be merged into UNIFIED_VOICE_CATALOG."""
        from vox_romana.tts.voice_catalog import (
            UNIFIED_VOICE_CATALOG,
            load_cloned_voice_catalog,
            match_voice,
        )

        _make_profile(
            tmp_path, "deep_voice",
            gender="male", tone="deep, authoritative",
            tags=["deep", "authoritative", "male"],
        )
        cloned = load_cloned_voice_catalog(tmp_path)
        merged = {**UNIFIED_VOICE_CATALOG, **cloned}

        # match_voice should be able to find the cloned voice
        result = match_voice(
            "deep authoritative",
            gender="male",
            available_voices=list(cloned.keys()),
            catalog=merged,
        )
        assert result == "vc_deep_voice"


# ─────────────────────────────────────────────────────────────────────────────
# MultiEngine.from_engine_names — voice_clone support
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiEngineVoiceClone:
    def test_from_engine_names_raises_without_profiles_dir(self, tmp_path):
        from vox_romana.tts.engines.multi_engine import MultiEngine

        with pytest.raises(ValueError, match="profiles_dir"):
            MultiEngine.from_engine_names(["voice_clone"])

    def test_from_engine_names_instantiates_voice_clone_engine(self, tmp_path):
        from vox_romana.tts.engines.multi_engine import MultiEngine
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        _make_profile(tmp_path, "test_voice")
        engine = MultiEngine.from_engine_names(
            ["voice_clone"], profiles_dir=tmp_path
        )
        assert len(engine._engines) == 1
        assert isinstance(engine._engines[0], VoiceCloneEngine)

    def test_from_engine_names_unknown_engine_still_raises(self, tmp_path):
        from vox_romana.tts.engines.multi_engine import MultiEngine

        with pytest.raises(ValueError, match="Unknown engine name"):
            MultiEngine.from_engine_names(["banana"])

    def test_voice_clone_name_is_lowercase_with_underscore(self, tmp_path):
        """VoiceCloneEngine.name must be 'voice_clone' to match _by_name key."""
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        engine = VoiceCloneEngine(profiles_dir=tmp_path)
        assert engine.name == "voice_clone"

    def test_voice_clone_in_by_name_dict(self, tmp_path):
        """MultiEngine._by_name should contain 'voice_clone' key."""
        from vox_romana.tts.engines.multi_engine import MultiEngine

        _make_profile(tmp_path, "alpha")
        multi = MultiEngine.from_engine_names(
            ["voice_clone"], profiles_dir=tmp_path
        )
        assert "voice_clone" in multi._by_name

    def test_vc_voice_dispatched_to_voice_clone_engine(self, tmp_path):
        """_engine_for_voice('vc_*') must return the VoiceCloneEngine."""
        from vox_romana.tts.engines.multi_engine import MultiEngine
        from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

        _make_profile(tmp_path, "dispatch_test")
        multi = MultiEngine.from_engine_names(
            ["voice_clone"], profiles_dir=tmp_path
        )
        dispatched = multi._engine_for_voice("vc_dispatch_test")
        assert isinstance(dispatched, VoiceCloneEngine)

    def test_unknown_voice_raises_in_multi_engine(self, tmp_path):
        from vox_romana.tts.engines.multi_engine import MultiEngine

        _make_profile(tmp_path, "known")
        multi = MultiEngine.from_engine_names(
            ["voice_clone"], profiles_dir=tmp_path
        )
        with pytest.raises(ValueError, match="No wrapped engine knows voice"):
            multi._engine_for_voice("vc_unknown_voice_xyz")

    def test_duplicate_engine_name_deduplicated(self, tmp_path):
        from vox_romana.tts.engines.multi_engine import MultiEngine

        _make_profile(tmp_path, "dedup")
        multi = MultiEngine.from_engine_names(
            ["voice_clone", "voice_clone"], profiles_dir=tmp_path
        )
        assert len(multi._engines) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Catalog.register_cloned_asset / list_voice_profiles
# ─────────────────────────────────────────────────────────────────────────────


class TestCatalogVoiceClone:
    @pytest.fixture
    def catalog(self):
        from vox_romana.core.catalog import Catalog

        cat = Catalog(db_path=":memory:")
        yield cat
        cat.close()

    @pytest.fixture
    def series_id(self, catalog):
        return catalog.create_series(title="Test Show", show_type="debate")

    def test_register_cloned_asset_returns_int(self, catalog, series_id):
        pk = catalog.register_cloned_asset(
            series_id=series_id,
            voice_id="vc_marcus_deep",
            profile_dir="/data/voice_profiles/vc_marcus_deep",
        )
        assert isinstance(pk, int)
        assert pk > 0

    def test_list_voice_profiles_returns_registered(self, catalog, series_id):
        catalog.register_cloned_asset(
            series_id=series_id,
            voice_id="vc_alice",
            character_name="Dr. Alice",
            profile_dir="/data/voice_profiles/vc_alice",
            notes="cloned from podcast",
        )
        results = catalog.list_voice_profiles(series_id)

        assert len(results) == 1
        rec = results[0]
        assert rec.asset_id == "vc_alice"
        assert rec.asset_type == "voice_clone"
        assert rec.character_name == "Dr. Alice"
        assert rec.source_path == "/data/voice_profiles/vc_alice"
        assert rec.notes == "cloned from podcast"

    def test_list_voice_profiles_empty_when_none_registered(self, catalog, series_id):
        assert catalog.list_voice_profiles(series_id) == []

    def test_list_voice_profiles_only_returns_voice_clone_type(
        self, catalog, series_id
    ):
        catalog.register_cloned_asset(
            series_id=series_id, voice_id="vc_bob"
        )
        catalog.approve_asset(
            series_id=series_id,
            asset_type="music",
            asset_id="jazz_bumper",
        )
        results = catalog.list_voice_profiles(series_id)
        assert len(results) == 1
        assert results[0].asset_id == "vc_bob"

    def test_multiple_cloned_voices_all_listed(self, catalog, series_id):
        for vid in ("vc_alice", "vc_bob", "vc_carol"):
            catalog.register_cloned_asset(series_id=series_id, voice_id=vid)
        results = catalog.list_voice_profiles(series_id)
        assert {r.asset_id for r in results} == {"vc_alice", "vc_bob", "vc_carol"}

    def test_register_cloned_asset_series_isolation(self, catalog):
        s1 = catalog.create_series(title="Series 1")
        s2 = catalog.create_series(title="Series 2")
        catalog.register_cloned_asset(series_id=s1, voice_id="vc_shared")

        assert len(catalog.list_voice_profiles(s1)) == 1
        assert len(catalog.list_voice_profiles(s2)) == 0
