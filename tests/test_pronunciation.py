"""Tests for the pronunciation guidance pipeline.

Covers:
    1. Brief parser — _parse_pronunciations() in cli/runners.py
    2. Renderer helpers — apply_pronunciation_rewrites() and
       build_pronunciation_hint() in tts/renderer.py
    3. Model round-trip — pronunciation_overrides field on all 5 merged
       show models persists through to_dict()
    4. Renderer wiring — pronunciation_overrides from show JSON reaches
       the synthesize() call (text rewrite for Piper, emotion hint for
       Kokoro)

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_pronunciation.py -v
"""

from __future__ import annotations

import dataclasses
import io
import wave
from unittest.mock import patch

import pytest

from vox_romana.cli.runners import _parse_pronunciations
from vox_romana.tts.renderer import (
    TTSRenderer,
    apply_pronunciation_rewrites,
    build_pronunciation_hint,
)
from vox_romana.tts.engine import BaseTTSEngine, SynthesizedAudio, VoiceConfig


# ───────────────────────────────────────────────────────────────────────────
# Shared helpers
# ───────────────────────────────────────────────────────────────────────────

def _make_wav(duration_ms: int = 200, sample_rate: int = 22050) -> bytes:
    num_samples = int(sample_rate * duration_ms / 1000)
    raw = b"\x00" * num_samples * 2
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)
    return buf.getvalue()


class FakeEngine(BaseTTSEngine):
    """Records every synthesize() call and returns silent WAV bytes."""

    def __init__(self, voice_list: list[str] | None = None):
        self.voices = voice_list or ["af_heart", "am_adam"]
        self.calls: list[dict] = []

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "default",
        emotion: str = "",
        speed: float = 1.0,
        **kwargs,
    ) -> SynthesizedAudio:
        self.calls.append({"text": text, "voice": voice, "emotion": emotion})
        return SynthesizedAudio(audio=_make_wav(), sample_rate=22050)

    def available_voices(self) -> list[str]:
        return list(self.voices)


def _debate_data(**extra) -> dict:
    """Minimal show JSON that TTSRenderer.render() accepts."""
    data: dict = {
        "title": "Test Show",
        "participants": [{"name": "Alice"}, {"name": "Bob"}],
        "events": [
            {
                "event_kind": "utterance",
                "speaker_name": "Alice",
                "text": "Okonkwo walked through New York.",
                "tone": [],
                "pace": "medium",
                "volume": "normal",
                "nonverbal": [],
                "interruption": False,
                "overlap_with_previous": False,
                "pause_before": None,
                "pause_after": None,
            }
        ],
    }
    data.update(extra)
    return data


# ═══════════════════════════════════════════════════════════════════════════
# 1. BRIEF PARSER
# ═══════════════════════════════════════════════════════════════════════════


class TestParsePronunciations:
    """Unit tests for _parse_pronunciations()."""

    def test_no_block_returns_brief_unchanged(self):
        brief = "This is a plain brief with no pronunciation block."
        cleaned, overrides = _parse_pronunciations(brief)
        assert cleaned == brief
        assert overrides == {}

    def test_basic_entry_parsed(self):
        brief = "My brief.\n\nPronunciations:\nOkonkwo = oh-KOHN-kwoh\n"
        cleaned, overrides = _parse_pronunciations(brief)
        assert "Okonkwo" in overrides
        assert overrides["Okonkwo"]["spoken"] == "oh-KOHN-kwoh"
        assert overrides["Okonkwo"]["ipa"] is None

    def test_ipa_parsed_when_present(self):
        brief = "Brief.\n\nPronunciations:\nNguyen = nwin | /ŋwɪn/\n"
        _, overrides = _parse_pronunciations(brief)
        assert overrides["Nguyen"]["spoken"] == "nwin"
        assert overrides["Nguyen"]["ipa"] == "/ŋwɪn/"

    def test_multiple_entries_all_parsed(self):
        brief = (
            "Brief.\n\nPronunciations:\n"
            "Okonkwo = oh-KOHN-kwoh\n"
            "Nguyen = nwin | /ŋwɪn/\n"
            "AI = ay-eye\n"
        )
        _, overrides = _parse_pronunciations(brief)
        assert set(overrides.keys()) == {"Okonkwo", "Nguyen", "AI"}

    def test_block_stripped_from_cleaned_brief(self):
        brief = "Topic about Okonkwo.\n\nPronunciations:\nOkonkwo = oh-KOHN-kwoh\n\nKeep this."
        cleaned, _ = _parse_pronunciations(brief)
        assert "Pronunciations:" not in cleaned
        assert "oh-KOHN-kwoh" not in cleaned
        assert "Topic about Okonkwo." in cleaned
        assert "Keep this." in cleaned

    def test_case_insensitive_header(self):
        brief = "Brief.\n\npronunciations:\nWord = wurd\n"
        _, overrides = _parse_pronunciations(brief)
        assert "Word" in overrides

    def test_lines_without_equals_skipped(self):
        brief = "Brief.\n\nPronunciations:\nskip this line\nReal = reel\n"
        _, overrides = _parse_pronunciations(brief)
        assert "skip this line" not in overrides
        assert "Real" in overrides

    def test_empty_brief_unchanged(self):
        brief = ""
        cleaned, overrides = _parse_pronunciations(brief)
        assert cleaned == ""
        assert overrides == {}

    def test_extra_whitespace_around_equals_trimmed(self):
        brief = "Brief.\n\nPronunciations:\n  Phrase  =  spoken form  \n"
        _, overrides = _parse_pronunciations(brief)
        assert "Phrase" in overrides
        assert overrides["Phrase"]["spoken"] == "spoken form"

    def test_multi_word_phrase_key(self):
        brief = "Brief.\n\nPronunciations:\nNew York = noo york\n"
        _, overrides = _parse_pronunciations(brief)
        assert "New York" in overrides
        assert overrides["New York"]["spoken"] == "noo york"


# ═══════════════════════════════════════════════════════════════════════════
# 2. APPLY PRONUNCIATION REWRITES (Piper text-rewrite path)
# ═══════════════════════════════════════════════════════════════════════════


class TestApplyPronunciationRewrites:
    """Unit tests for apply_pronunciation_rewrites()."""

    def test_empty_overrides_returns_text_unchanged(self):
        text = "Hello world."
        assert apply_pronunciation_rewrites(text, {}) == text

    def test_single_phrase_replaced(self):
        overrides = {"Okonkwo": {"spoken": "oh-KOHN-kwoh", "ipa": None}}
        result = apply_pronunciation_rewrites("Meet Okonkwo today.", overrides)
        assert result == "Meet oh-KOHN-kwoh today."

    def test_replacement_is_case_insensitive(self):
        overrides = {"okonkwo": {"spoken": "oh-KOHN-kwoh", "ipa": None}}
        result = apply_pronunciation_rewrites("Meet OKONKWO today.", overrides)
        assert "oh-KOHN-kwoh" in result

    def test_multi_word_phrase_replaced(self):
        overrides = {"New York": {"spoken": "noo york", "ipa": None}}
        result = apply_pronunciation_rewrites("Flying to New York tomorrow.", overrides)
        assert "noo york" in result

    def test_longer_phrase_replaced_before_shorter(self):
        """'New York' is replaced before 'York' and the result is not mangled.

        Single-pass replacement means the spoken form 'noo york' is never
        re-examined, so the shorter 'York' → 'yawk' rule cannot fire on it.
        """
        overrides = {
            "New York": {"spoken": "noo york", "ipa": None},
            "York": {"spoken": "yawk", "ipa": None},
        }
        result = apply_pronunciation_rewrites("I love New York.", overrides)
        assert result == "I love noo york."

    def test_whole_word_boundary_respected(self):
        """'AI' should not match inside 'SAID' or 'RAIN'."""
        overrides = {"AI": {"spoken": "ay-eye", "ipa": None}}
        result = apply_pronunciation_rewrites("SAID it was raining.", overrides)
        assert result == "SAID it was raining."

    def test_phrase_in_middle_of_sentence(self):
        overrides = {"Nguyen": {"spoken": "nwin", "ipa": None}}
        result = apply_pronunciation_rewrites("My friend Nguyen said hello.", overrides)
        assert result == "My friend nwin said hello."

    def test_empty_spoken_form_skipped(self):
        """An entry with an empty spoken form must not rewrite."""
        overrides = {"Word": {"spoken": "", "ipa": None}}
        text = "This Word appears."
        assert apply_pronunciation_rewrites(text, overrides) == text

    def test_plain_string_value_used_as_spoken(self):
        """Dict values that are plain strings (not nested dicts) still work."""
        overrides = {"Okonkwo": "oh-KOHN-kwoh"}  # legacy flat format
        result = apply_pronunciation_rewrites("Meet Okonkwo.", overrides)
        # The function falls back to str() for non-dict values.
        assert "oh-KOHN-kwoh" in result


# ═══════════════════════════════════════════════════════════════════════════
# 3. BUILD PRONUNCIATION HINT (Kokoro emotion-hint path)
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildPronunciationHint:
    """Unit tests for build_pronunciation_hint()."""

    def test_empty_overrides_returns_empty_string(self):
        assert build_pronunciation_hint({}) == ""

    def test_single_entry(self):
        overrides = {"Okonkwo": {"spoken": "oh-KOHN-kwoh", "ipa": None}}
        result = build_pronunciation_hint(overrides)
        assert result == "pronounce Okonkwo as oh-KOHN-kwoh"

    def test_multiple_entries_joined_by_semicolon(self):
        overrides = {
            "Okonkwo": {"spoken": "oh-KOHN-kwoh", "ipa": None},
            "AI": {"spoken": "ay-eye", "ipa": None},
        }
        result = build_pronunciation_hint(overrides)
        assert "pronounce Okonkwo as oh-KOHN-kwoh" in result
        assert "pronounce AI as ay-eye" in result
        assert ";" in result

    def test_entries_with_empty_spoken_skipped(self):
        overrides = {
            "Word": {"spoken": "", "ipa": None},
            "Real": {"spoken": "reel", "ipa": None},
        }
        result = build_pronunciation_hint(overrides)
        assert "Word" not in result
        assert "pronounce Real as reel" in result

    def test_ipa_not_included_in_hint(self):
        """IPA is stored as metadata only — must not appear in the hint string."""
        overrides = {"Okonkwo": {"spoken": "oh-KOHN-kwoh", "ipa": "/oʊˈkɒŋkwoʊ/"}}
        result = build_pronunciation_hint(overrides)
        assert "/oʊˈkɒŋkwoʊ/" not in result


# ═══════════════════════════════════════════════════════════════════════════
# 4. MODEL ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════════════


class TestPronunciationOverridesInModels:
    """pronunciation_overrides persists through dataclasses.replace() and to_dict()."""

    OVERRIDES = {"Okonkwo": {"spoken": "oh-KOHN-kwoh", "ipa": None}}

    def test_podcast_model(self):
        from vox_romana.shows.podcast.models import MergedPodcast
        m = MergedPodcast(podcast_id="p1", title="Test")
        m2 = dataclasses.replace(m, pronunciation_overrides=self.OVERRIDES)
        assert m2.pronunciation_overrides == self.OVERRIDES
        assert m2.to_dict()["pronunciation_overrides"] == self.OVERRIDES

    def test_podcast_default_is_empty(self):
        from vox_romana.shows.podcast.models import MergedPodcast
        m = MergedPodcast(podcast_id="p1", title="Test")
        assert m.to_dict()["pronunciation_overrides"] == {}

    def test_debate_model(self):
        from vox_romana.shows.debates.models import MergedDebate
        m = MergedDebate(template_id="t", debate_id="d", title="T", chunks=(), events=())
        m2 = dataclasses.replace(m, pronunciation_overrides=self.OVERRIDES)
        assert m2.pronunciation_overrides == self.OVERRIDES
        assert m2.to_dict()["pronunciation_overrides"] == self.OVERRIDES

    def test_news_model(self):
        from vox_romana.shows.news.models import MergedNewsShow
        m = MergedNewsShow(show_id="n1", title="Test")
        m2 = dataclasses.replace(m, pronunciation_overrides=self.OVERRIDES)
        assert m2.to_dict()["pronunciation_overrides"] == self.OVERRIDES

    def test_audiobook_model(self):
        from vox_romana.shows.audiobooks.models import MergedAudiobook
        m = MergedAudiobook(audiobook_id="a1", title="Test")
        m2 = dataclasses.replace(m, pronunciation_overrides=self.OVERRIDES)
        assert m2.to_dict()["pronunciation_overrides"] == self.OVERRIDES

    def test_lecture_model(self):
        from vox_romana.shows.lecture.models import MergedLecture
        m = MergedLecture(lecture_id="l1", title="Test")
        m2 = dataclasses.replace(m, pronunciation_overrides=self.OVERRIDES)
        assert m2.to_dict()["pronunciation_overrides"] == self.OVERRIDES

    def test_to_dict_is_a_copy(self):
        """Mutating the dict from to_dict() must not affect the model."""
        from vox_romana.shows.podcast.models import MergedPodcast
        m = MergedPodcast(podcast_id="p1", title="Test",
                          pronunciation_overrides={"A": {"spoken": "ay", "ipa": None}})
        d = m.to_dict()
        d["pronunciation_overrides"]["B"] = {"spoken": "bee", "ipa": None}
        assert "B" not in m.pronunciation_overrides


# ═══════════════════════════════════════════════════════════════════════════
# 5. RENDERER WIRING
# ═══════════════════════════════════════════════════════════════════════════


class TestRendererPronunciationWiring:
    """pronunciation_overrides in show JSON reaches the synthesis call correctly."""

    def _render(self, tmp_path, overrides: dict, voice_id: str = "am_adam",
                engine_name: str = "kokoro") -> FakeEngine:
        """Helper: render a single-event show with given pronunciation_overrides."""
        engine = FakeEngine(voice_list=[voice_id])
        renderer = TTSRenderer(engine, voice_map={"Alice": VoiceConfig(voice_id=voice_id)})

        data = _debate_data(pronunciation_overrides=overrides)

        # Patch the catalog so the test doesn't depend on real catalog data.
        from vox_romana.tts.voice_catalog import VoiceDescriptor
        fake_desc = VoiceDescriptor(
            voice_id=voice_id,
            gender="male",
            accent="american",
            age="middle",
            tone="neutral",
            best_for="test",
            engine=engine_name,
        )
        with patch("vox_romana.tts.renderer.UNIFIED_VOICE_CATALOG",
                   {voice_id: fake_desc}):
            renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        return engine

    def test_kokoro_voice_gets_hint_in_emotion(self, tmp_path):
        """For a Kokoro voice, pronunciation hint appears in the emotion arg."""
        overrides = {"Okonkwo": {"spoken": "oh-KOHN-kwoh", "ipa": None}}
        engine = self._render(tmp_path, overrides, voice_id="am_adam", engine_name="kokoro")
        assert len(engine.calls) == 1
        emotion = engine.calls[0]["emotion"]
        assert "pronounce Okonkwo as oh-KOHN-kwoh" in emotion

    def test_piper_voice_rewrites_text(self, tmp_path):
        """For a Piper voice, the source phrase is rewritten in the text arg."""
        overrides = {"Okonkwo": {"spoken": "oh-KOHN-kwoh", "ipa": None}}
        engine = self._render(tmp_path, overrides, voice_id="en_us_test", engine_name="piper")
        assert len(engine.calls) == 1
        text = engine.calls[0]["text"]
        assert "oh-KOHN-kwoh" in text
        assert "Okonkwo" not in text

    def test_piper_voice_no_hint_in_emotion(self, tmp_path):
        """For a Piper voice, pronunciation hint must NOT appear in emotion."""
        overrides = {"Okonkwo": {"spoken": "oh-KOHN-kwoh", "ipa": None}}
        engine = self._render(tmp_path, overrides, voice_id="en_us_test", engine_name="piper")
        emotion = engine.calls[0]["emotion"]
        assert "pronounce" not in emotion

    def test_kokoro_voice_text_not_rewritten(self, tmp_path):
        """For a Kokoro voice, source phrase must remain in the text arg unchanged."""
        overrides = {"Okonkwo": {"spoken": "oh-KOHN-kwoh", "ipa": None}}
        engine = self._render(tmp_path, overrides, voice_id="am_adam", engine_name="kokoro")
        text = engine.calls[0]["text"]
        # Original phrase must still be present (not rewritten)
        assert "Okonkwo" in text

    def test_empty_overrides_no_hint_added(self, tmp_path):
        """When pronunciation_overrides is empty, emotion is built normally."""
        engine = self._render(tmp_path, {}, voice_id="am_adam", engine_name="kokoro")
        emotion = engine.calls[0]["emotion"]
        assert "pronounce" not in emotion

    def test_no_pronunciation_overrides_key_in_json(self, tmp_path):
        """If the show JSON has no pronunciation_overrides key at all, render succeeds."""
        engine = FakeEngine(voice_list=["am_adam"])
        renderer = TTSRenderer(engine, voice_map={"Alice": VoiceConfig(voice_id="am_adam")})
        data = _debate_data()  # no pronunciation_overrides key
        renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)
        assert len(engine.calls) == 1
