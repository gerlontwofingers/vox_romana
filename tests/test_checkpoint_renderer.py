"""Tests for vox_romana.tts.checkpoint — CheckpointRenderer."""

from __future__ import annotations

import io
import math
import wave

import pytest

from vox_romana.tts.checkpoint import CheckpointRenderer, MeasurementResult
from vox_romana.tts.engine import BaseTTSEngine, SynthesizedAudio


def _make_wav_bytes(seconds: float, sample_rate: int = 22050) -> bytes:
    """Build a valid silent-WAV byte blob of the given duration."""
    n_frames = int(seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


class _FakeEngine(BaseTTSEngine):
    """Synthesizes silent WAV bytes whose duration encodes a fixed WPM."""

    def __init__(self, wpm: float = 150.0, sample_rate: int = 22050) -> None:
        self.wpm = wpm
        self.sample_rate = sample_rate
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
        self.calls.append({
            "text": text, "voice": voice, "emotion": emotion, "speed": speed,
        })
        word_count = len(text.split())
        seconds = word_count / self.wpm * 60.0
        return SynthesizedAudio(
            audio=_make_wav_bytes(seconds, self.sample_rate),
            sample_rate=self.sample_rate,
            format="wav",
        )

    def available_voices(self) -> list[str]:
        return ["default", "test_voice"]


# ── Basic measurement ─────────────────────────────────────────────


class TestMeasure:
    def test_returns_seconds(self):
        engine = _FakeEngine(wpm=150.0)
        renderer = CheckpointRenderer(engine)
        # 150 words at 150 wpm = 60 seconds
        text = " ".join(["word"] * 150)
        seconds = renderer.measure(text, voice="default")
        assert seconds == pytest.approx(60.0, rel=0.02)

    def test_empty_text_returns_zero_no_engine_call(self):
        engine = _FakeEngine()
        renderer = CheckpointRenderer(engine)
        assert renderer.measure("") == 0.0
        assert renderer.measure("   \n  ") == 0.0
        assert engine.calls == []

    def test_passes_voice_and_emotion_to_engine(self):
        engine = _FakeEngine()
        renderer = CheckpointRenderer(engine)
        renderer.measure("hello world", voice="my_voice", emotion="excited")
        assert len(engine.calls) == 1
        assert engine.calls[0]["voice"] == "my_voice"
        assert engine.calls[0]["emotion"] == "excited"

    def test_uses_default_voice_when_none_supplied(self):
        engine = _FakeEngine()
        renderer = CheckpointRenderer(engine, default_voice="fallback")
        renderer.measure("hello")
        assert engine.calls[0]["voice"] == "fallback"


# ── Detailed measurement ──────────────────────────────────────────


class TestMeasureDetailed:
    def test_returns_full_result(self):
        engine = _FakeEngine(wpm=150.0)
        renderer = CheckpointRenderer(engine)
        text = " ".join(["word"] * 150)
        result = renderer.measure_detailed(text, voice="default")
        assert isinstance(result, MeasurementResult)
        assert result.word_count == 150
        assert result.seconds == pytest.approx(60.0, rel=0.02)
        assert result.wpm == pytest.approx(150.0, rel=0.02)

    def test_wpm_calculation_reflects_voice_pace(self):
        # Slow engine: 100 wpm
        slow = CheckpointRenderer(_FakeEngine(wpm=100.0))
        text = " ".join(["w"] * 50)
        result = slow.measure_detailed(text)
        assert result.wpm == pytest.approx(100.0, rel=0.05)

    def test_empty_returns_zero_wpm(self):
        renderer = CheckpointRenderer(_FakeEngine())
        result = renderer.measure_detailed("")
        assert result.wpm == 0.0
        assert result.word_count == 0


# ── Duration parsing strategies ───────────────────────────────────


class TestDurationOf:
    def test_wav_header_parsed_in_memory(self):
        # Build a WAV directly; verify parse without temp files / ffmpeg.
        wav = _make_wav_bytes(seconds=3.5, sample_rate=22050)
        synth = SynthesizedAudio(audio=wav, sample_rate=22050, format="wav")
        seconds = CheckpointRenderer._duration_of(synth)
        assert seconds == pytest.approx(3.5, abs=0.01)

    def test_corrupt_wav_falls_back_to_pydub_or_raises(self):
        synth = SynthesizedAudio(
            audio=b"not a wav at all", sample_rate=22050, format="wav",
        )
        # Either pydub raises (no codec) or returns junk; we just make sure
        # this doesn't crash silently with a meaningless number.
        with pytest.raises(Exception):
            CheckpointRenderer._duration_of(synth)
