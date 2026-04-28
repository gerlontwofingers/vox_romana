"""Tests for the Series feature — writers'-bible records.

Covers:
    1. Catalog.create_series / get_series / list_series / update_series_bible
    2. Catalog.get_recent_episode_summaries
    3. _build_series_context() — formats SERIES CONTEXT block
    4. run_generate() injects series_context into bootstrap kwargs
    5. run_generate() exits with error for unknown --series
    6. vox_romana.cli.series subcommands (create, list, inspect)
    7. run.py dispatches --mode series to the series CLI
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.core.audience import AudienceProfile
from vox_romana.core.catalog import Catalog, SeriesRecord


@pytest.fixture
def catalog():
    cat = Catalog(db_path=":memory:")
    yield cat
    cat.close()


# ═══════════════════════════════════════════════════════════════════════
# Catalog CRUD
# ═══════════════════════════════════════════════════════════════════════


class TestSeriesCRUD:
    def test_create_series_generates_uuid(self, catalog):
        sid = catalog.create_series(title="Test Show")
        assert isinstance(sid, str)
        assert len(sid) == 12

    def test_create_series_persists_fields(self, catalog):
        sid = catalog.create_series(
            title="Trash Pandas",
            show_type="debate",
            tone="satirical",
            bible="Always close on a punchline.",
        )
        s = catalog.get_series(sid)
        assert s is not None
        assert s.title == "Trash Pandas"
        assert s.show_type == "debate"
        assert s.tone == "satirical"
        assert s.bible == "Always close on a punchline."
        assert s.audience_profile is None

    def test_get_series_not_found_returns_none(self, catalog):
        assert catalog.get_series("nope") is None

    def test_list_series_newest_first(self, catalog):
        sid1 = catalog.create_series(title="First")
        sid2 = catalog.create_series(title="Second")
        rows = catalog.list_series()
        ids = [r.series_id for r in rows]
        assert sid1 in ids and sid2 in ids
        # Newest first; created_at is ISO so lexical ordering matches
        assert rows[0].created_at >= rows[-1].created_at

    def test_update_series_bible(self, catalog):
        sid = catalog.create_series(title="X", bible="v1")
        catalog.update_series_bible(sid, "v2 — overhauled")
        assert catalog.get_series(sid).bible == "v2 — overhauled"

    def test_get_recent_episode_summaries_empty(self, catalog):
        sid = catalog.create_series(title="Empty")
        assert catalog.get_recent_episode_summaries(sid) == []

    def test_get_recent_episode_summaries_returns_eps(self, catalog):
        sid = catalog.create_series(title="With Eps")
        catalog.log_run(
            content_type="debate",
            title="Episode A",
            brief="brief A",
            series_id=sid,
            episode_number=1,
        )
        catalog.log_run(
            content_type="debate",
            title="Episode B",
            brief="brief B",
            series_id=sid,
            episode_number=2,
        )
        eps = catalog.get_recent_episode_summaries(sid, n=5)
        assert len(eps) == 2
        # Newest first
        titles = [e["title"] for e in eps]
        assert "Episode A" in titles and "Episode B" in titles


# ═══════════════════════════════════════════════════════════════════════
# _build_series_context
# ═══════════════════════════════════════════════════════════════════════


class TestBuildSeriesContext:
    def test_unknown_series_raises(self, catalog):
        from vox_romana.cli.runners import _build_series_context
        with pytest.raises(ValueError, match="Unknown series"):
            _build_series_context(catalog, "does-not-exist")

    def test_includes_title_and_bible(self, catalog):
        from vox_romana.cli.runners import _build_series_context
        sid = catalog.create_series(
            title="Trash Pandas",
            tone="satirical",
            bible="Recurring rule: always end on a punchline.",
        )
        block = _build_series_context(catalog, sid)
        assert "SERIES CONTEXT" in block
        assert "Trash Pandas" in block
        assert "satirical" in block
        assert "always end on a punchline" in block

    def test_includes_recent_episodes(self, catalog):
        from vox_romana.cli.runners import _build_series_context
        sid = catalog.create_series(title="With Eps")
        catalog.log_run(
            content_type="debate",
            title="Episode One",
            brief="b",
            series_id=sid,
            episode_number=1,
        )
        block = _build_series_context(catalog, sid)
        assert "Episode One" in block


# ═══════════════════════════════════════════════════════════════════════
# run_generate integration
# ═══════════════════════════════════════════════════════════════════════


class TestRunGenerateSeriesWiring:
    def _make_args(self, tmp_path, **extra):
        prompt = tmp_path / "p.txt"
        prompt.write_text("debate UBI")
        out = tmp_path / "out"
        ns = argparse.Namespace(
            prompt=prompt,
            output=out,
            show_type="debate",
            research=False,
            quiet=True,
            tag=None,
            fact_check="off",
            content_mode="auto",
            duration=None,
            tolerance=None,
            checkpoint=None,
            engine="kokoro",
            turns=None,
            tone=None,
            debate_format=None,
            segments=None,
            max_retries=None,
            reference_material=None,
            allow_bootstrap_fabrications=False,
            series=None,
            name=None,
        )
        for k, v in extra.items():
            setattr(ns, k, v)
        return ns

    def test_unknown_series_exits(self, tmp_path):
        from vox_romana.cli import runners
        args = self._make_args(tmp_path, series="ghost-id")
        with patch.object(runners, "DeepSeekClient"):
            with pytest.raises(SystemExit):
                runners.run_generate(args)

    def test_known_series_threads_context_to_runner(self, tmp_path, monkeypatch):
        """run_generate looks up the series, builds context, and stores it on args."""
        from vox_romana.cli import runners
        from vox_romana.core.catalog import Catalog

        # Create the series in the live DB the runner will open
        with Catalog() as cat:
            sid = cat.create_series(title="Test Series", bible="Be funny.")

        try:
            captured = {}

            def fake_runner(brief, client, args):
                captured["series_context"] = getattr(args, "series_context", None)
                captured["series_id"] = getattr(args, "series_id", None)
                # Return a placeholder path
                out = tmp_path / "out" / "fake.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("{}")
                return out

            monkeypatch.setitem(runners._RUNNERS, "debate", fake_runner)
            monkeypatch.setattr(runners, "DeepSeekClient", MagicMock())

            args = self._make_args(tmp_path, series=sid)
            runners.run_generate(args)

            assert captured["series_context"] is not None
            assert "Test Series" in captured["series_context"]
            assert "Be funny." in captured["series_context"]
            assert captured["series_id"] == sid
        finally:
            with Catalog() as cat:
                cat._conn.execute("DELETE FROM series WHERE series_id = ?", (sid,))
                cat._conn.commit()


# ═══════════════════════════════════════════════════════════════════════
# Series CLI subcommands
# ═══════════════════════════════════════════════════════════════════════


class TestSeriesCLI:
    def test_create_prints_uuid(self, monkeypatch, capsys):
        from vox_romana.cli import series as series_cli
        from vox_romana.core.catalog import Catalog

        fake = Catalog(db_path=":memory:")
        monkeypatch.setattr(series_cli, "Catalog", lambda: _ContextCatalog(fake))

        series_cli.main(["create", "My Show", "--type", "debate"])
        out = capsys.readouterr().out.strip()
        assert len(out) == 12
        # And it really was created
        assert fake.get_series(out) is not None

    def test_list_empty(self, monkeypatch, capsys):
        from vox_romana.cli import series as series_cli
        from vox_romana.core.catalog import Catalog

        fake = Catalog(db_path=":memory:")
        monkeypatch.setattr(series_cli, "Catalog", lambda: _ContextCatalog(fake))

        series_cli.main(["list"])
        assert "No series yet" in capsys.readouterr().out

    def test_list_shows_entries(self, monkeypatch, capsys):
        from vox_romana.cli import series as series_cli
        from vox_romana.core.catalog import Catalog

        fake = Catalog(db_path=":memory:")
        sid = fake.create_series(title="Listable", show_type="news")
        monkeypatch.setattr(series_cli, "Catalog", lambda: _ContextCatalog(fake))

        series_cli.main(["list"])
        out = capsys.readouterr().out
        assert sid in out
        assert "Listable" in out

    def test_inspect_unknown_returns_nonzero(self, monkeypatch):
        from vox_romana.cli import series as series_cli
        from vox_romana.core.catalog import Catalog

        fake = Catalog(db_path=":memory:")
        monkeypatch.setattr(series_cli, "Catalog", lambda: _ContextCatalog(fake))

        with pytest.raises(SystemExit):
            series_cli.main(["inspect", "missing"])

    def test_inspect_prints_record(self, monkeypatch, capsys):
        from vox_romana.cli import series as series_cli
        from vox_romana.core.catalog import Catalog

        fake = Catalog(db_path=":memory:")
        sid = fake.create_series(title="Inspectable", bible="rules!")
        monkeypatch.setattr(series_cli, "Catalog", lambda: _ContextCatalog(fake))

        series_cli.main(["inspect", sid])
        out = capsys.readouterr().out
        assert "Inspectable" in out
        assert "rules!" in out


class _ContextCatalog:
    """Wrap an existing Catalog so `with Catalog() as cat:` reuses it."""

    def __init__(self, inner):
        self._inner = inner

    def __enter__(self):
        return self._inner

    def __exit__(self, *a):
        return None


# ═══════════════════════════════════════════════════════════════════════
# run.py dispatch
# ═══════════════════════════════════════════════════════════════════════


class TestRunPyDispatch:
    def test_extract_series_argv_basic(self):
        import run as run_module
        assert run_module._extract_series_argv(
            ["--mode", "series", "create", "Foo"]
        ) == ["create", "Foo"]

    def test_extract_series_argv_short(self):
        import run as run_module
        assert run_module._extract_series_argv(
            ["-m", "series", "list"]
        ) == ["list"]

    def test_extract_series_argv_none_when_other_mode(self):
        import run as run_module
        assert run_module._extract_series_argv(
            ["--mode", "generate", "p.txt"]
        ) is None

    def test_main_dispatches_to_series_cli(self, monkeypatch):
        import run as run_module
        called = {}

        def fake_main(argv):
            called["argv"] = argv

        monkeypatch.setattr(
            "vox_romana.cli.series.main", fake_main
        )
        run_module.main(["--mode", "series", "list"])
        assert called["argv"] == ["list"]


# ═══════════════════════════════════════════════════════════════════════
# A3 — AudienceProfile
# ═══════════════════════════════════════════════════════════════════════


class TestAudienceProfile:
    def test_audience_profile_roundtrip_json(self):
        """Serialise → deserialise → fields match original."""
        original = AudienceProfile(
            age_range="25-40",
            education_level="graduate",
            vocabulary_level="educated",
            preferred_tone="rigorous but warm",
            prior_knowledge="knows Python",
        )
        restored = AudienceProfile.from_json(original.to_json())
        assert restored.age_range == "25-40"
        assert restored.education_level == "graduate"
        assert restored.vocabulary_level == "educated"
        assert restored.preferred_tone == "rigorous but warm"
        assert restored.prior_knowledge == "knows Python"

    def test_audience_profile_roundtrip_partial(self):
        """Only vocabulary_level is required; all other fields may be None."""
        partial = AudienceProfile(vocabulary_level="plain")
        restored = AudienceProfile.from_json(partial.to_json())
        assert restored.vocabulary_level == "plain"
        assert restored.age_range is None
        assert restored.prior_knowledge is None

    def test_audience_profile_to_prompt_text(self):
        """to_prompt_text() includes all set fields; omits None fields."""
        ap = AudienceProfile(
            age_range="18-25",
            education_level="undergraduate",
            vocabulary_level="conversational",
            preferred_tone="energetic",
            prior_knowledge="new to economics",
        )
        text = ap.to_prompt_text()
        assert "18-25" in text
        assert "undergraduate" in text
        assert "conversational" in text
        assert "energetic" in text
        assert "new to economics" in text

    def test_audience_profile_to_prompt_text_sparse(self):
        """to_prompt_text() only emits vocabulary when all other fields are None."""
        ap = AudienceProfile()  # all defaults
        text = ap.to_prompt_text()
        assert "conversational" in text
        assert "Age range" not in text
        assert "Education" not in text


class TestAudienceProfileCatalogIntegration:
    def test_create_series_with_audience_profile(self, catalog):
        """create_series() accepts AudienceProfile and stores it."""
        ap = AudienceProfile(
            age_range="30-50",
            vocabulary_level="educated",
        )
        sid = catalog.create_series(title="Profiled Show", audience_profile=ap)
        s = catalog.get_series(sid)
        assert s is not None
        assert s.audience_profile is not None
        assert s.audience_profile.age_range == "30-50"
        assert s.audience_profile.vocabulary_level == "educated"

    def test_set_audience_profile_persists(self, catalog):
        """set_audience_profile() replaces the profile on an existing series."""
        sid = catalog.create_series(title="Late Profile")
        assert catalog.get_series(sid).audience_profile is None

        ap = AudienceProfile(preferred_tone="sceptical", prior_knowledge="ML basics")
        catalog.set_audience_profile(sid, ap)

        s = catalog.get_series(sid)
        assert s.audience_profile is not None
        assert s.audience_profile.preferred_tone == "sceptical"
        assert s.audience_profile.prior_knowledge == "ML basics"

    def test_audience_profile_injected_into_series_context(self, catalog):
        """_build_series_context() emits AUDIENCE block when profile is set."""
        from vox_romana.cli.runners import _build_series_context

        ap = AudienceProfile(
            age_range="35-55",
            education_level="expert",
            vocabulary_level="expert",
            preferred_tone="rigorous",
            prior_knowledge="PhD-level physics",
        )
        sid = catalog.create_series(
            title="Expert Show",
            bible="Keep it precise.",
            audience_profile=ap,
        )
        block = _build_series_context(catalog, sid)
        assert "AUDIENCE" in block
        assert "35-55" in block
        assert "expert" in block
        assert "rigorous" in block
        assert "PhD-level physics" in block

    def test_series_context_no_audience_section_when_no_profile(self, catalog):
        """_build_series_context() omits AUDIENCE when no profile is set."""
        from vox_romana.cli.runners import _build_series_context

        sid = catalog.create_series(title="No Profile Show", bible="keep it light")
        block = _build_series_context(catalog, sid)
        assert "AUDIENCE" not in block


class TestSeriesCLIAudienceFlags:
    def test_create_with_audience_flags(self, monkeypatch, capsys):
        """--audience-* flags on `create` persist an AudienceProfile."""
        from vox_romana.cli import series as series_cli
        from vox_romana.core.catalog import Catalog

        fake = Catalog(db_path=":memory:")
        monkeypatch.setattr(series_cli, "Catalog", lambda: _ContextCatalog(fake))

        series_cli.main([
            "create", "Audience Show",
            "--audience-age", "20-30",
            "--audience-education", "undergraduate",
            "--audience-vocab", "conversational",
            "--audience-tone", "playful",
            "--audience-prior", "some Python",
        ])
        sid = capsys.readouterr().out.strip()
        s = fake.get_series(sid)
        assert s.audience_profile is not None
        assert s.audience_profile.age_range == "20-30"
        assert s.audience_profile.preferred_tone == "playful"

    def test_inspect_shows_audience_profile(self, monkeypatch, capsys):
        """inspect subcommand prints audience profile when present."""
        from vox_romana.cli import series as series_cli
        from vox_romana.core.catalog import Catalog

        fake = Catalog(db_path=":memory:")
        ap = AudienceProfile(age_range="40+", preferred_tone="serious")
        sid = fake.create_series(title="Inspect Audience", audience_profile=ap)
        monkeypatch.setattr(series_cli, "Catalog", lambda: _ContextCatalog(fake))

        series_cli.main(["inspect", sid])
        out = capsys.readouterr().out
        assert "40+" in out
        assert "serious" in out
