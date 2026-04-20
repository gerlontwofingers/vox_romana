"""CheckpointRenderer — measure the actual TTS duration of one segment.

This is the digital-timing step from TV post-production. After a segment
is generated (text), we run TTS on JUST that segment, measure its
duration, throw the audio away, and report back. The conductor uses the
measurement to decide whether to retry the segment, rebalance the
remaining budget, or just keep going.

WHY MEASURE INSTEAD OF ESTIMATE?
    Word counts × WPM is a rough estimate. A specific voice on a specific
    engine can run 10-20% faster or slower than the genre default. By the
    time you notice the show is short, it's too late to fix it.

    Measuring lets us close the loop. We learn the true voice-specific
    WPM as we go, and we react to drift in real time.

USAGE:

    from vox_romana.tts.checkpoint import CheckpointRenderer
    from vox_romana.tts.engines.kokoro_engine import KokoroEngine

    engine = KokoroEngine()
    checkpoint = CheckpointRenderer(engine)

    text = "She walked into the room and froze."
    duration_seconds = checkpoint.measure(text, voice="af_heart")
    # → ~3.2

    # For richer telemetry:
    result = checkpoint.measure_detailed(text, voice="af_heart")
    # → MeasurementResult(seconds=3.2, word_count=8, wpm=150.0)

DESIGN NOTES:
    * Audio is rendered to a temp file and immediately deleted. We never
      keep it around — the final render still happens through the normal
      pipeline.
    * Duration is read with pydub (which uses ffmpeg under the hood).
    * If the engine raises, we surface the exception unchanged so the
      conductor can decide whether to skip checkpointing for that segment.
"""

from __future__ import annotations

import io
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from .engine import BaseTTSEngine, SynthesizedAudio


@dataclass(frozen=True)
class MeasurementResult:
    """One checkpoint measurement.

    Attributes:
        seconds:    Measured audio duration.
        word_count: Number of words in the source text.
        wpm:        Effective words per minute = word_count / seconds * 60.
                    Useful for feeding back into the WPM model.
    """

    seconds: float
    word_count: int
    wpm: float


class CheckpointRenderer:
    """Render → measure → discard. One segment at a time.

    Wraps any BaseTTSEngine and exposes a `measure()` method that returns
    the actual audio duration in seconds.

    Args:
        engine:        The TTS engine to use for measurement renders.
        default_voice: Default voice to use when none is supplied.
    """

    def __init__(
        self,
        engine: BaseTTSEngine,
        *,
        default_voice: str | None = None,
    ) -> None:
        self._engine = engine
        self._default_voice = default_voice

    # ── Public API ──────────────────────────────────────────────────

    def measure(
        self,
        text: str,
        *,
        voice: str | None = None,
        emotion: str = "",
        speed: float = 1.0,
        **kwargs,
    ) -> float:
        """Render the text and return its duration in seconds.

        See ``measure_detailed`` for richer telemetry.
        """
        return self.measure_detailed(
            text, voice=voice, emotion=emotion, speed=speed, **kwargs,
        ).seconds

    def measure_detailed(
        self,
        text: str,
        *,
        voice: str | None = None,
        emotion: str = "",
        speed: float = 1.0,
        **kwargs,
    ) -> MeasurementResult:
        """Render the text and return seconds + word_count + effective WPM.

        Empty / whitespace-only text returns a zero-duration result without
        invoking the engine.
        """
        clean_text = text.strip()
        word_count = len(clean_text.split())
        if not clean_text:
            return MeasurementResult(seconds=0.0, word_count=0, wpm=0.0)

        chosen_voice = voice or self._default_voice or "default"
        synth = self._engine.synthesize(
            clean_text,
            voice=chosen_voice,
            emotion=emotion,
            speed=speed,
            **kwargs,
        )
        seconds = self._duration_of(synth)
        wpm = (word_count / seconds * 60.0) if seconds > 0 else 0.0
        return MeasurementResult(
            seconds=seconds,
            word_count=word_count,
            wpm=wpm,
        )

    # ── Internal: duration extraction ──────────────────────────────

    @staticmethod
    def _duration_of(synth: SynthesizedAudio) -> float:
        """Compute audio duration in seconds.

        Strategy 1 (preferred): parse the WAV header in-memory using the
        stdlib ``wave`` module. No ffmpeg dependency, no temp file.

        Strategy 2 (fallback): write to a temp file and use pydub. Used
        when the bytes aren't a recognizable WAV (e.g. mp3 from OpenAI).
        """
        # Strategy 1: in-memory WAV header read.
        if synth.format.lower() == "wav":
            try:
                with wave.open(io.BytesIO(synth.audio), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        return frames / float(rate)
            except (wave.Error, EOFError):
                pass  # Fall through to pydub.

        # Strategy 2: temp file + pydub.
        return CheckpointRenderer._duration_via_pydub(
            synth.audio, synth.format,
        )

    @staticmethod
    def _duration_via_pydub(audio_bytes: bytes, fmt: str) -> float:
        """Fallback path: write to a temp file, ask pydub for duration."""
        from pydub import AudioSegment

        suffix = f".{fmt}" if not fmt.startswith(".") else fmt
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False,
            ) as f:
                f.write(audio_bytes)
                tmp_path = Path(f.name)
            seg = AudioSegment.from_file(str(tmp_path), format=fmt)
            return seg.duration_seconds
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
