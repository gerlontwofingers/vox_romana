"""Tests for the high-quality voice pool (--hqvoices) and on-demand download."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from vox_romana.tts import hq_voices
from vox_romana.tts.engine import BaseTTSEngine, SynthesizedAudio, VoiceConfig
from vox_romana.tts.engines.kokoro_engine import KOKORO_VOICES
from vox_romana.tts.hq_voices import (
    HQ_DOWNLOAD_SUGGESTIONS,
    KOKORO_VOICE_GRADES,
    download_piper_voice,
    ensure_hq_voices,
    is_hq_voice,
    piper_quality,
    piper_voice_local_dir,
    piper_voice_urls,
    plan_hq_downloads,
    restrict_to_hq,
)
from vox_romana.tts.renderer import TTSRenderer
from vox_romana.tts.voice_catalog import KOKORO_VOICE_CATALOG, PIPER_VOICE_CATALOG


# ─────────────────────────────────────────────────────────────────────────────
# Quality rules
# ─────────────────────────────────────────────────────────────────────────────


class TestIsHqVoice:
    @pytest.mark.parametrize("voice", ["af_heart", "af_nicole", "bf_emma", "am_michael", "bm_george"])
    def test_kokoro_grade_c_or_better_is_hq(self, voice):
        assert is_hq_voice(voice)

    @pytest.mark.parametrize("voice", ["am_adam", "bm_lewis", "af_sky", "af_river", "bf_lily", "am_onyx"])
    def test_kokoro_below_grade_c_is_not_hq(self, voice):
        assert not is_hq_voice(voice)

    def test_grade_c_is_the_floor(self):
        assert KOKORO_VOICE_GRADES["af_nova"] == "C"
        assert is_hq_voice("af_nova")
        assert KOKORO_VOICE_GRADES["af_sky"] == "C-"
        assert not is_hq_voice("af_sky")

    @pytest.mark.parametrize("voice", ["en_US-lessac-high", "en_US-amy-medium", "en_GB-cori-high"])
    def test_cataloged_english_piper_medium_and_high_are_hq(self, voice):
        assert is_hq_voice(voice)

    @pytest.mark.parametrize("voice", [
        "en_US-danny-low", "en_US-kathleen-low", "en_GB-southern_english_female-low",
    ])
    def test_piper_low_tier_is_not_hq(self, voice):
        assert not is_hq_voice(voice)

    @pytest.mark.parametrize("voice", ["en_US-libritts-high", "en_US-arctic-medium", "en_GB-vctk-medium"])
    def test_uncataloged_english_piper_is_not_hq(self, voice):
        """Multi-speaker corpora and other uncurated English voices are excluded."""
        assert not is_hq_voice(voice)

    def test_non_english_piper_uses_tier_only(self):
        assert is_hq_voice("it_IT-paola-medium")
        assert is_hq_voice("de_DE-thorsten-high")
        assert not is_hq_voice("it_IT-riccardo-x_low")
        assert not is_hq_voice("fr_FR-siwis-low")

    def test_cloned_voices_always_allowed(self):
        assert is_hq_voice("vc_marcus")

    def test_unknown_ids_are_not_hq(self):
        assert not is_hq_voice("default")
        assert not is_hq_voice("mystery_voice")

    def test_extra_exclude_overrides_the_rules(self, monkeypatch):
        monkeypatch.setattr(hq_voices, "HQ_EXTRA_EXCLUDE", frozenset({"af_heart"}))
        assert not is_hq_voice("af_heart")

    def test_every_catalog_and_engine_kokoro_voice_has_a_grade(self):
        """A voice without a grade would silently count as non-HQ."""
        assert set(KOKORO_VOICE_CATALOG) <= set(KOKORO_VOICE_GRADES)
        assert set(KOKORO_VOICES) <= set(KOKORO_VOICE_GRADES)

    def test_piper_quality_parses_tier(self):
        assert piper_quality("en_US-lessac-high") == "high"
        assert piper_quality("it_IT-riccardo-x_low") == "x_low"
        assert piper_quality("af_heart") is None

    def test_pool_is_large_enough_for_variety(self):
        hq = [v for v in {**KOKORO_VOICE_CATALOG, **PIPER_VOICE_CATALOG} if is_hq_voice(v)]
        genders = [({**KOKORO_VOICE_CATALOG, **PIPER_VOICE_CATALOG})[v].gender for v in hq]
        assert genders.count("male") >= 8
        assert genders.count("female") >= 8


class TestRestrictToHq:
    def test_drops_low_quality_english_voices(self):
        voices = ["am_adam", "af_heart", "en_US-danny-low", "en_US-ryan-high"]
        assert restrict_to_hq(voices) == ["af_heart", "en_US-ryan-high"]

    def test_preserves_order(self):
        voices = ["en_US-ryan-high", "af_heart", "bm_george"]
        assert restrict_to_hq(voices) == voices

    def test_never_empties_a_language(self):
        """Italian only has an x_low voice here; keep it rather than lose Italian."""
        voices = ["af_heart", "it_IT-riccardo-x_low"]
        assert restrict_to_hq(voices) == voices

    def test_languages_are_judged_independently(self):
        voices = ["it_IT-riccardo-x_low", "es_ES-davefx-medium", "es_ES-mls_10246-low"]
        assert restrict_to_hq(voices) == ["it_IT-riccardo-x_low", "es_ES-davefx-medium"]

    def test_english_with_no_hq_voice_keeps_normal_voices(self):
        voices = ["am_adam", "en_US-danny-low"]
        assert restrict_to_hq(voices) == voices

    def test_cloned_voices_do_not_cause_normal_voices_to_be_dropped(self):
        voices = ["vc_marcus", "am_adam", "en_US-danny-low"]
        assert restrict_to_hq(voices) == voices

    def test_empty_input(self):
        assert restrict_to_hq([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# URLs and paths
# ─────────────────────────────────────────────────────────────────────────────


class TestPiperUrls:
    def test_url_layout_matches_the_voices_repository(self):
        model, config = piper_voice_urls("en_US-lessac-high")
        base = f"{hq_voices.PIPER_VOICES_BASE_URL}/en/en_US/lessac/high/en_US-lessac-high"
        assert model == f"{base}.onnx"
        assert config == f"{base}.onnx.json"

    def test_non_ascii_names_are_percent_encoded(self):
        model, _ = piper_voice_urls("pt_PT-tugão-medium")
        assert "tug%C3%A3o" in model
        assert "ã" not in model

    def test_underscored_speaker_names(self):
        model, _ = piper_voice_urls("en_GB-southern_english_female-low")
        assert "/en_GB/southern_english_female/low/" in model

    def test_rejects_non_piper_ids(self):
        with pytest.raises(ValueError):
            piper_voice_urls("af_heart")

    def test_local_dir_mirrors_repo_layout(self, tmp_path):
        assert piper_voice_local_dir("en_US-lessac-high", tmp_path) == (
            tmp_path / "en" / "en_US" / "lessac" / "high"
        )

    def test_every_suggestion_is_a_medium_or_better_piper_id(self):
        for lang, voice in HQ_DOWNLOAD_SUGGESTIONS.items():
            assert piper_quality(voice) in hq_voices.HQ_PIPER_TIERS, voice
            assert voice.startswith(lang + "_"), voice
            piper_voice_urls(voice)  # parses


# ─────────────────────────────────────────────────────────────────────────────
# Downloading
# ─────────────────────────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, body: bytes, *, content_length: int | None = None):
        self._buf = io.BytesIO(body)
        length = len(body) if content_length is None else content_length
        self.headers = {"Content-Length": str(length)}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener(*, model_body: bytes = b"x" * 2_000_000, config_body: bytes = b"{}",
            model_length: int | None = None):
    calls: list[str] = []

    def opener(url, timeout=None):
        calls.append(url)
        if url.endswith(".onnx.json"):
            return _FakeResponse(config_body)
        return _FakeResponse(model_body, content_length=model_length)

    opener.calls = calls  # type: ignore[attr-defined]
    return opener


class TestDownloadPiperVoice:
    def test_success_writes_both_files_and_no_partials(self, tmp_path):
        ok = download_piper_voice(
            "en_US-lessac-high", tmp_path, show_progress=False, opener=_opener(),
        )
        target = tmp_path / "en" / "en_US" / "lessac" / "high"
        assert ok
        assert (target / "en_US-lessac-high.onnx").stat().st_size == 2_000_000
        assert json.loads((target / "en_US-lessac-high.onnx.json").read_text()) == {}
        assert not list(tmp_path.rglob("*.part"))

    def test_downloaded_voice_is_discoverable_by_the_engine(self, tmp_path):
        from vox_romana.tts.engines.piper_engine import PiperEngine

        download_piper_voice("en_US-lessac-high", tmp_path, show_progress=False, opener=_opener())
        assert "en_US-lessac-high" in PiperEngine(model_dir=tmp_path).available_voices()

    def test_truncated_download_is_rejected_and_cleaned_up(self, tmp_path):
        opener = _opener(model_body=b"x" * 2_000_000, model_length=3_000_000)
        assert not download_piper_voice("en_US-lessac-high", tmp_path, show_progress=False, opener=opener)
        assert not list(tmp_path.rglob("*.onnx*"))

    def test_tiny_file_is_rejected(self, tmp_path):
        opener = _opener(model_body=b"<html>not a model</html>")
        assert not download_piper_voice("en_US-lessac-high", tmp_path, show_progress=False, opener=opener)
        assert not list(tmp_path.rglob("*.onnx*"))

    def test_invalid_config_json_is_rejected(self, tmp_path):
        opener = _opener(config_body=b"not json")
        assert not download_piper_voice("en_US-lessac-high", tmp_path, show_progress=False, opener=opener)
        assert not list(tmp_path.rglob("*.onnx*"))

    def test_network_error_returns_false(self, tmp_path):
        def opener(url, timeout=None):
            raise urllib.error.URLError("offline")

        assert not download_piper_voice("en_US-lessac-high", tmp_path, show_progress=False, opener=opener)
        assert not list(tmp_path.rglob("*.part"))

    def test_http_error_returns_false(self, tmp_path):
        def opener(url, timeout=None):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        assert not download_piper_voice("en_US-lessac-high", tmp_path, show_progress=False, opener=opener)

    def test_progress_is_reported_when_enabled(self, tmp_path):
        stream = io.StringIO()
        download_piper_voice(
            "en_US-lessac-high", tmp_path, stream=stream, opener=_opener(),
        )
        out = stream.getvalue()
        assert "en_US-lessac-high" in out
        assert "100%" in out
        assert "MB" in out

    def test_no_progress_output_when_disabled(self, tmp_path):
        stream = io.StringIO()
        download_piper_voice(
            "en_US-lessac-high", tmp_path, show_progress=False, stream=stream, opener=_opener(),
        )
        assert stream.getvalue() == ""


# ─────────────────────────────────────────────────────────────────────────────
# Planning
# ─────────────────────────────────────────────────────────────────────────────


def _show(casting=None, *, language="en", character_langs=None):
    return {
        "language": language,
        "character_langs": character_langs or {},
        "voice_casting": casting or {},
    }


class TestPlanHqDownloads:
    def test_kokoro_only_engine_never_downloads(self):
        casting = {f"M{i}": "male" for i in range(10)}
        assert plan_hq_downloads(_show(casting), set(), engine="kokoro") == []

    def test_both_engine_is_covered_by_kokoro_for_small_casts(self):
        casting = {"A": "male", "B": "female", "C": "male"}
        assert plan_hq_downloads(_show(casting), set(), engine="both") == []

    def test_both_engine_downloads_only_the_shortfall(self):
        """Kokoro has 4 HQ male voices, so 6 male speakers need 2 more."""
        casting = {f"M{i}": "male" for i in range(6)}
        plan = plan_hq_downloads(_show(casting), set(), engine="both")
        assert len(plan) == 2
        assert all(PIPER_VOICE_CATALOG[v].gender == "male" for v, _ in plan)

    def test_piper_engine_downloads_what_the_cast_needs(self):
        casting = {"A": "male", "B": "male", "C": "female"}
        installed = {"en_US-ryan-high"}
        plan = plan_hq_downloads(_show(casting), installed, engine="piper")
        voices = [v for v, _ in plan]
        assert len(voices) == 2  # one more male, one female
        assert "en_US-ryan-high" not in voices
        assert {PIPER_VOICE_CATALOG[v].gender for v in voices} == {"male", "female"}

    def test_high_tier_voices_are_preferred(self):
        plan = plan_hq_downloads(_show({"A": "female"}), set(), engine="piper")
        assert piper_quality(plan[0][0]) == "high"

    def test_nothing_planned_when_pool_already_covers_the_cast(self):
        installed = {"en_US-ryan-high", "en_US-joe-medium", "en_US-lessac-high"}
        casting = {"A": "male", "B": "male", "C": "female"}
        assert plan_hq_downloads(_show(casting), installed, engine="piper") == []

    def test_low_tier_installed_voices_do_not_count(self):
        installed = {"en_US-danny-low"}
        plan = plan_hq_downloads(_show({"A": "male"}), installed, engine="piper")
        assert len(plan) == 1

    def test_non_english_language_without_hq_voice_is_planned(self):
        plan = plan_hq_downloads(_show(language="it"), {"it_IT-riccardo-x_low"}, engine="piper")
        assert [v for v, _ in plan] == ["it_IT-paola-medium"]
        assert "'it'" in plan[0][1]

    def test_non_english_language_with_hq_voice_is_left_alone(self):
        plan = plan_hq_downloads(_show(language="it"), {"it_IT-paola-medium"}, engine="piper")
        assert plan == []

    def test_character_languages_are_included(self):
        show = _show(character_langs={"Maria": "es"})
        plan = plan_hq_downloads(show, set(), engine="both")
        assert [v for v, _ in plan] == ["es_ES-davefx-medium"]

    def test_non_english_speakers_do_not_count_toward_english_need(self):
        show = _show({"Maria": "female"}, character_langs={"Maria": "es"})
        plan = plan_hq_downloads(show, {"es_ES-davefx-medium"}, engine="piper")
        assert plan == []

    def test_no_duplicates(self):
        show = _show({"A": "male"}, language="es", character_langs={"B": "es"})
        plan = plan_hq_downloads(show, set(), engine="piper")
        voices = [v for v, _ in plan]
        assert len(voices) == len(set(voices))

    def test_unknown_language_without_suggestion_is_ignored(self):
        assert plan_hq_downloads(_show(language="xx"), set(), engine="piper") == []


class TestEnsureHqVoices:
    def _install(self, model_dir: Path, voice_id: str) -> None:
        d = piper_voice_local_dir(voice_id, model_dir)
        d.mkdir(parents=True)
        (d / f"{voice_id}.onnx").write_bytes(b"x")

    def test_nothing_missing_is_silent_and_does_not_download(self, tmp_path):
        stream = io.StringIO()
        calls = []
        done = ensure_hq_voices(
            _show({"A": "male"}), engine="kokoro", model_dir=tmp_path,
            downloader=lambda *a, **k: calls.append(a) or True, stream=stream,
        )
        assert done == [] and calls == [] and stream.getvalue() == ""

    def test_tells_the_user_before_downloading(self, tmp_path):
        stream = io.StringIO()
        events: list[str] = []

        def downloader(voice_id, model_dir, *, show_progress, stream):
            events.append(f"download:{voice_id}")
            return True

        done = ensure_hq_voices(
            _show({"A": "female"}), engine="piper", model_dir=tmp_path,
            downloader=downloader, stream=stream,
        )
        out = stream.getvalue()
        assert len(done) == 1
        assert "will be downloaded" in out
        assert "--no-hqvoices" in out
        assert done[0] in out
        assert "Downloaded 1 of 1" in out

    def test_quiet_still_announces_but_hides_progress_bars(self, tmp_path):
        stream = io.StringIO()
        seen = {}

        def downloader(voice_id, model_dir, *, show_progress, stream):
            seen["show_progress"] = show_progress
            return True

        ensure_hq_voices(
            _show({"A": "female"}), engine="piper", model_dir=tmp_path,
            quiet=True, downloader=downloader, stream=stream,
        )
        assert "will be downloaded" in stream.getvalue()
        assert seen["show_progress"] is False

    def test_failures_are_reported_and_not_fatal(self, tmp_path):
        stream = io.StringIO()
        done = ensure_hq_voices(
            _show({"A": "female"}), engine="piper", model_dir=tmp_path,
            downloader=lambda *a, **k: False, stream=stream,
        )
        assert done == []
        assert "Downloaded 0 of 1" in stream.getvalue()
        assert "continuing" in stream.getvalue()

    def test_accepts_a_path_to_the_show_json(self, tmp_path):
        show_path = tmp_path / "show.json"
        show_path.write_text(json.dumps(_show({"A": "female"})))
        done = ensure_hq_voices(
            show_path, engine="piper", model_dir=tmp_path / "voices",
            downloader=lambda *a, **k: True, stream=io.StringIO(),
        )
        assert len(done) == 1

    def test_unreadable_show_json_is_not_fatal(self, tmp_path):
        assert ensure_hq_voices(
            tmp_path / "missing.json", engine="piper", model_dir=tmp_path,
            downloader=lambda *a, **k: True, stream=io.StringIO(),
        ) == []

    def test_installed_voices_are_not_downloaded_again(self, tmp_path):
        self._install(tmp_path, "it_IT-paola-medium")
        calls = []
        ensure_hq_voices(
            _show(language="it"), engine="piper", model_dir=tmp_path,
            downloader=lambda *a, **k: calls.append(a) or True, stream=io.StringIO(),
        )
        assert calls == []


# ─────────────────────────────────────────────────────────────────────────────
# Renderer integration
# ─────────────────────────────────────────────────────────────────────────────


class _PoolEngine(BaseTTSEngine):
    def __init__(self, voices):
        self._voices = voices

    def synthesize(self, text, *, voice="default", emotion="", speed=1.0, **kwargs):
        return SynthesizedAudio(audio=b"", sample_rate=22050)

    def available_voices(self):
        return list(self._voices)


POOL = ["am_adam", "am_michael", "af_river", "af_heart", "am_onyx"]
CASTING = {"Alice": "female", "Bob": "male"}


class TestRendererHqPool:
    def test_default_keeps_legacy_pool(self):
        renderer = TTSRenderer(_PoolEngine(POOL))
        voices = renderer._build_voice_map(["Alice", "Bob"], CASTING)
        assert voices["Bob"] == "am_adam"  # F+ grade, first male in the pool
        assert voices["Alice"] == "af_river"

    def test_hq_pool_skips_low_quality_voices(self):
        renderer = TTSRenderer(_PoolEngine(POOL), hq_voices=True)
        voices = renderer._build_voice_map(["Alice", "Bob"], CASTING)
        assert voices == {"Alice": "af_heart", "Bob": "am_michael"}

    def test_explicit_voice_map_is_never_changed(self):
        renderer = TTSRenderer(
            _PoolEngine(POOL), hq_voices=True,
            voice_map={"Bob": VoiceConfig(voice_id="am_adam")},
        )
        voices = renderer._build_voice_map(["Alice", "Bob"], CASTING)
        assert voices["Bob"] == "am_adam"

    def test_pool_with_no_hq_voice_still_assigns(self):
        renderer = TTSRenderer(_PoolEngine(["am_adam", "af_river"]), hq_voices=True)
        voices = renderer._build_voice_map(["Alice", "Bob"], CASTING)
        assert voices == {"Alice": "af_river", "Bob": "am_adam"}

    def test_default_voice_is_hq(self):
        renderer = TTSRenderer(_PoolEngine(["am_adam", "af_heart"]), hq_voices=True)
        assert renderer._default_voice() == "af_heart"

    def test_hint_matching_only_uses_hq_installed_voices(self):
        hints = {"Bob": "deep authoritative baritone", "Alice": "warm clear narrator"}
        renderer = TTSRenderer(_PoolEngine(POOL), hq_voices=True)
        voices = renderer._build_voice_map(["Alice", "Bob"], CASTING, hints)
        assert set(voices.values()) <= {"af_heart", "am_michael"}

    def test_locked_series_voice_that_is_not_hq_is_reassigned(self):
        class _Registry:
            def resolve(self, name, series_key):
                return "am_adam"

            def usage_counts(self):
                return {}

            def register(self, *a, **k):
                pass

        renderer = TTSRenderer(
            _PoolEngine(POOL), hq_voices=True, registry=_Registry(), series_key="s",
        )
        voices = renderer._build_voice_map(["Bob"], CASTING, registry=_Registry(), series_key="s")
        assert voices["Bob"] == "am_michael"


# ─────────────────────────────────────────────────────────────────────────────
# CLI flag
# ─────────────────────────────────────────────────────────────────────────────


class TestHqVoicesFlag:
    def test_run_py_defaults_on(self):
        from run import parse_args

        assert parse_args(["a prompt"]).hqvoices is True

    def test_run_py_can_be_turned_off_and_on(self):
        from run import parse_args

        assert parse_args(["a prompt", "--no-hqvoices"]).hqvoices is False
        assert parse_args(["a prompt", "--no-hqvoices", "--hqvoices"]).hqvoices is True

    def test_render_argv_forwards_only_when_disabled(self, tmp_path):
        from run import _build_render_argv, parse_args

        on = _build_render_argv(tmp_path / "s.json", parse_args(["a prompt"]))
        off = _build_render_argv(tmp_path / "s.json", parse_args(["a prompt", "--no-hqvoices"]))
        assert "--no-hqvoices" not in on
        assert "--no-hqvoices" in off

    def test_render_parser_defaults_on_and_honours_the_flag(self, tmp_path):
        from vox_romana.cli.runners import _render_parse_args

        assert _render_parse_args([str(tmp_path / "s.json")]).hqvoices is True
        assert _render_parse_args([str(tmp_path / "s.json"), "--no-hqvoices"]).hqvoices is False
