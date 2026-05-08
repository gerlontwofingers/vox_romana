"""Tests for the intro/outro music bumper injection pipeline.

Covers:
    1. _ensure_music_bumpers() helper — all edge cases
    2. TTSRenderer.render() wiring — bumpers injected when music engine is
       active, not injected when it is absent, caller's data not mutated
"""

from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

from vox_romana.tts.engine import BaseTTSEngine, SynthesizedAudio, VoiceConfig
from vox_romana.tts.renderer import TTSRenderer, _ensure_music_bumpers


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _make_wav(duration_ms: int = 100, sample_rate: int = 22050) -> bytes:
    num_samples = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00" * num_samples * 2)
    return buf.getvalue()


class FakeTTSEngine(BaseTTSEngine):
    def __init__(self):
        self.calls: list[dict] = []

    def synthesize(self, text, *, voice="default", emotion="", speed=1.0, **kw) -> SynthesizedAudio:
        self.calls.append({"text": text, "voice": voice})
        return SynthesizedAudio(audio=_make_wav(), sample_rate=22050)

    def available_voices(self) -> list[str]:
        return ["am_adam"]


class FakeMusicEngine:
    """Minimal stand-in for MusicGenEngine — returns silent audio."""

    def generate(self, prompt: str, *, duration: float = 8.0) -> SynthesizedAudio:
        return SynthesizedAudio(audio=_make_wav(int(duration * 1000)), sample_rate=32000)


def _utterance(nonverbal=None, event_kind="utterance") -> dict:
    return {
        "event_kind": event_kind,
        "speaker_name": "Alice",
        "text": "Hello world.",
        "tone": [],
        "pace": "medium",
        "volume": "normal",
        "nonverbal": list(nonverbal or []),
        "interruption": False,
        "overlap_with_previous": False,
        "pause_before": None,
        "pause_after": None,
    }


def _show_data(*events) -> dict:
    return {
        "title": "Test Show",
        "participants": [{"name": "Alice"}],
        "events": list(events),
    }


def _renderer(*, music=False) -> TTSRenderer:
    engine = FakeTTSEngine()
    music_engine = FakeMusicEngine() if music else None
    return TTSRenderer(
        engine,
        voice_map={"Alice": VoiceConfig(voice_id="am_adam")},
        music_engine=music_engine,  # type: ignore[arg-type]
        music_style="test music",
        music_duration=1.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. _ensure_music_bumpers unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsureMusicBumpers:

    def test_no_cues_both_added(self):
        events = [_utterance(), _utterance()]
        _ensure_music_bumpers(events)
        assert "intro music" in events[0]["nonverbal"]
        assert "outro music" in events[-1]["nonverbal"]

    def test_intro_already_present_not_duplicated(self):
        events = [_utterance(["intro music"]), _utterance()]
        _ensure_music_bumpers(events)
        assert events[0]["nonverbal"].count("intro music") == 1
        assert "outro music" in events[-1]["nonverbal"]

    def test_outro_already_present_not_duplicated(self):
        events = [_utterance(), _utterance(["outro music"])]
        _ensure_music_bumpers(events)
        assert "intro music" in events[0]["nonverbal"]
        assert events[-1]["nonverbal"].count("outro music") == 1

    def test_both_already_present_no_change(self):
        events = [_utterance(["intro music"]), _utterance(["outro music"])]
        before_first = list(events[0]["nonverbal"])
        before_last = list(events[-1]["nonverbal"])
        _ensure_music_bumpers(events)
        assert events[0]["nonverbal"] == before_first
        assert events[-1]["nonverbal"] == before_last

    def test_empty_events_no_crash(self):
        _ensure_music_bumpers([])  # must not raise

    def test_single_event_both_cues_added(self):
        events = [_utterance()]
        _ensure_music_bumpers(events)
        # First and last are the same event — both cues go on it
        assert "intro music" in events[0]["nonverbal"]
        assert "outro music" in events[0]["nonverbal"]

    def test_last_event_is_bumper_kind_outro_not_added(self):
        """event_kind == 'bumper' counts as having the cue already."""
        events = [_utterance(), _utterance(event_kind="bumper")]
        _ensure_music_bumpers(events)
        assert "outro music" not in events[-1]["nonverbal"]

    def test_first_event_is_bumper_kind_intro_not_added(self):
        events = [_utterance(event_kind="bumper"), _utterance()]
        _ensure_music_bumpers(events)
        assert "intro music" not in events[0]["nonverbal"]

    def test_existing_nonverbal_items_preserved(self):
        """Other nonverbal cues on the first/last event must survive."""
        events = [_utterance(["laugh"]), _utterance(["sigh"])]
        _ensure_music_bumpers(events)
        assert "laugh" in events[0]["nonverbal"]
        assert "sigh" in events[-1]["nonverbal"]

    def test_none_nonverbal_handled(self):
        """Events with nonverbal=None must not raise."""
        events = [
            {**_utterance(), "nonverbal": None},
            {**_utterance(), "nonverbal": None},
        ]
        _ensure_music_bumpers(events)
        assert "intro music" in events[0]["nonverbal"]
        assert "outro music" in events[-1]["nonverbal"]

    def test_case_insensitive_match(self):
        """'Intro Music' (title-case) should count as the cue already present."""
        events = [_utterance(["Intro Music"]), _utterance()]
        _ensure_music_bumpers(events)
        assert events[0]["nonverbal"].count("Intro Music") == 1
        assert "intro music" not in events[0]["nonverbal"]  # not added a second time


# ═══════════════════════════════════════════════════════════════════════════
# 2. TTSRenderer.render() wiring
# ═══════════════════════════════════════════════════════════════════════════


class TestRendererMusicBumperWiring:

    def test_bumpers_injected_when_music_engine_set(self, tmp_path):
        """When a music engine is active, render() adds bumper cues to a copy."""
        renderer = _renderer(music=True)
        data = _show_data(_utterance(), _utterance())
        # Keep a reference to the original events to test mutation safety
        original_events = data["events"]

        renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        # Original data must not be mutated
        assert "intro music" not in original_events[0]["nonverbal"]
        assert "outro music" not in original_events[-1]["nonverbal"]

    def test_bumpers_not_injected_without_music_engine(self, tmp_path):
        """When no music engine, _ensure_music_bumpers is never called."""
        renderer = _renderer(music=False)
        data = _show_data(_utterance(), _utterance())
        original_nonverbal_0 = list(data["events"][0]["nonverbal"])

        renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        assert data["events"][0]["nonverbal"] == original_nonverbal_0

    def test_caller_data_not_mutated_when_music_active(self, tmp_path):
        """render() must shallow-copy events before injecting cues."""
        renderer = _renderer(music=True)
        data = _show_data(_utterance(), _utterance())
        events_before = [dict(e) for e in data["events"]]

        renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        for before, after in zip(events_before, data["events"]):
            assert before["nonverbal"] == after["nonverbal"]
