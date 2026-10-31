"""Tests for vox_romana.studio.bundle (shot 5.2-c).

Covers materialize_bundle (series-ID-agnostic, with/without episode brief)
and zip_bundle.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from vox_romana.core.audience import AudienceProfile
from vox_romana.core.catalog import SeriesRecord
from vox_romana.planning.episode_planner import EpisodeBrief
from vox_romana.studio import materialize_bundle, zip_bundle


def _series() -> SeriesRecord:
    return SeriesRecord(
        series_id="uuid1234567890",
        title="Signal & Static",
        show_type="podcast",
        tone="warm and skeptical",
        bible="Tone: curious. Motifs: systems, trade-offs.",
        created_at="2026-10-31T00:00:00+00:00",
        audience_profile=AudienceProfile(
            age_range="25-40",
            education_level="undergraduate",
            vocabulary_level="educated",
            preferred_tone="conversational",
            prior_knowledge="Consumer tech familiar.",
        ),
    )


def _episode() -> EpisodeBrief:
    return EpisodeBrief(
        title="Episode 1",
        angle="Why we trust machines.",
        prompt_text="A podcast episode about trust in technology.",
        episode_number=1,
    )


class TestMaterializeBundle:
    def test_full_bundle(self, tmp_path):
        bundle_dir = materialize_bundle(
            _series(), tmp_path, slug="signal-and-static",
            episode_brief=_episode(),
        )
        assert bundle_dir == tmp_path / "studio" / "signal-and-static"

        # studio.yaml
        import yaml
        yaml_data = yaml.safe_load((bundle_dir / "studio.yaml").read_text())
        assert yaml_data["series_id"] == "uuid1234567890"
        assert yaml_data["slug"] == "signal-and-static"
        assert yaml_data["show_type"] == "podcast"
        assert yaml_data["audience"]["vocabulary_level"] == "educated"

        # bible.md
        assert "Tone: curious" in (bundle_dir / "bible.md").read_text()

        # audience.json
        audience = json.loads((bundle_dir / "audience.json").read_text())
        assert audience["vocabulary_level"] == "educated"

        # prompts/
        assert (bundle_dir / "prompts" / "template.txt").exists()
        assert "uuid1234567890" in (bundle_dir / "prompts" / "template.txt").read_text()
        ep1 = (bundle_dir / "prompts" / "episode_1_brief.txt").read_text()
        assert "trust in technology" in ep1

        # auditions manifest
        queue = json.loads((bundle_dir / "auditions" / "queue.json").read_text())
        assert queue["series_id"] == "uuid1234567890"
        assert queue["queue"] == []

    def test_default_slug_is_series_id(self, tmp_path):
        bundle_dir = materialize_bundle(_series(), tmp_path)
        assert bundle_dir.name == "uuid1234567890"

    def test_no_episode_brief_omits_file(self, tmp_path):
        bundle_dir = materialize_bundle(_series(), tmp_path, slug="s")
        assert not (bundle_dir / "prompts" / "episode_1_brief.txt").exists()

    def test_empty_bible_written_as_placeholder(self, tmp_path):
        series = _series()
        series.bible = ""
        bundle_dir = materialize_bundle(series, tmp_path, slug="s")
        assert "(empty bible)" in (bundle_dir / "bible.md").read_text()

    def test_idempotent_rewrite(self, tmp_path):
        materialize_bundle(_series(), tmp_path, slug="s")
        # Write a stale file, then re-materialize — should be wiped.
        stale = tmp_path / "studio" / "s" / "stale.txt"
        stale.write_text("old")
        bundle_dir = materialize_bundle(_series(), tmp_path, slug="s")
        assert not (bundle_dir / "stale.txt").exists()


class TestZipBundle:
    def test_zips_bundle(self, tmp_path):
        bundle_dir = materialize_bundle(
            _series(), tmp_path, slug="signal-and-static",
            episode_brief=_episode(),
        )
        archive = zip_bundle(bundle_dir)
        assert archive == tmp_path / "studio" / "signal-and-static.zip"
        assert archive.exists()

        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
        assert "studio.yaml" in names
        assert "bible.md" in names
        assert "audience.json" in names
        assert "prompts/episode_1_brief.txt" in names
        assert "auditions/queue.json" in names

    def test_zip_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            zip_bundle(tmp_path / "does_not_exist")
