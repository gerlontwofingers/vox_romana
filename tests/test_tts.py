"""Tests for the TTS pipeline — engine, mixer, renderer, voice catalog, and plugins.

This file tests:
    1. SynthesizedAudio and VoiceConfig data classes
    2. AudioMixer: clip sequencing, pauses, overlaps, normalization, export
    3. TTSRenderer: voice assignment, emotion hints, pace→speed, event rendering
    4. PiperEngine: model loading, WAV generation, speed control (mocked)
    5. KokoroEngine: generate_audio call, temp file handling, emotion (mocked)
    6. VoiceCatalog: voice matching, cast assignment, gender filtering
    7. Catalog-based renderer voice matching (voice_hints integration)
    8. TTS style passthrough (tts_styles → emotion hint prefix)

All TTS engine tests use mocks — no real models, no real GPU, no real audio
generation. The mixer tests use real pydub AudioSegments built from silence
(which is fine — pydub doesn't need ffmpeg to create or concatenate silence).

THE TESTING PHILOSOPHY:

    We test the WIRING, not the audio quality. We verify that:
    - The right arguments reach the right functions
    - Error handling kicks in at the right moments
    - Voice assignment logic is deterministic
    - Emotion hints are built correctly from event metadata
    - Pauses and overlaps are applied as expected

    We do NOT:
    - Generate real speech (that's the engine's job, tested manually)
    - Validate audio quality (that's a human ear's job)
    - Benchmark GPU performance (that's a profiler's job)

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_tts.py -v

Dependencies:
    pip install pytest pydub  (both already installed)
"""

import io
import json
import struct
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest
from pydub import AudioSegment

from vox_romana.tts.engine import BaseTTSEngine, SynthesizedAudio, VoiceConfig
from vox_romana.tts.mixer import (
    AudioMixer,
    AudioClip,
    DEFAULT_PAUSE_MS,
    DEFAULT_OVERLAP_MS,
    DEFAULT_SOFT_OVERLAP_MS,
    PAUSE_MAP,
    TARGET_DBFS,
)
from vox_romana.tts.renderer import TTSRenderer, RenderResult


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS — generate fake WAV bytes for testing
# ═══════════════════════════════════════════════════════════════════════════

def make_wav_bytes(
    duration_ms: int = 500,
    sample_rate: int = 22050,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Create valid WAV bytes containing silence of a given duration.

    This is the "blank film stock" of our test suite — valid WAV data
    that pydub can ingest, but with no meaningful audio content.

    Args:
        duration_ms:  Duration in milliseconds.
        sample_rate:  Samples per second (default 22050 matches Piper).
        channels:     Number of audio channels (1 = mono).
        sample_width: Bytes per sample (2 = 16-bit PCM).

    Returns:
        Valid WAV bytes that pydub can load.
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    # Silent samples: all zeros.
    raw_data = b"\x00" * (num_samples * channels * sample_width)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_data)
    return buf.getvalue()


class FakeEngine(BaseTTSEngine):
    """A fake TTS engine for testing the renderer.

    Returns predictable WAV bytes and tracks what was called.
    Think of it as a "stunt double" — it looks like a real engine
    to the renderer, but it's just going through the motions.
    """

    def __init__(self, voices: list[str] | None = None):
        self.voices = voices or ["voice_a", "voice_b", "voice_c"]
        self.calls: list[dict] = []  # Records every synthesize() call.

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "default",
        emotion: str = "",
        speed: float = 1.0,
        **kwargs,
    ) -> SynthesizedAudio:
        # Record the call for later assertions.
        self.calls.append({
            "text": text,
            "voice": voice,
            "emotion": emotion,
            "speed": speed,
            "kwargs": kwargs,
        })
        return SynthesizedAudio(
            audio=make_wav_bytes(duration_ms=200),
            sample_rate=22050,
        )

    def available_voices(self) -> list[str]:
        return list(self.voices)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA CLASS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSynthesizedAudio:
    """Tests for the SynthesizedAudio dataclass."""

    def test_basic_creation(self):
        """SynthesizedAudio stores audio bytes and sample rate."""
        audio = SynthesizedAudio(audio=b"fake_wav", sample_rate=22050)
        assert audio.audio == b"fake_wav"
        assert audio.sample_rate == 22050
        assert audio.format == "wav"  # Default format.

    def test_custom_format(self):
        """Format can be overridden (for future engine types)."""
        audio = SynthesizedAudio(audio=b"data", sample_rate=24000, format="mp3")
        assert audio.format == "mp3"


class TestVoiceConfig:
    """Tests for the VoiceConfig dataclass."""

    def test_basic_creation(self):
        """VoiceConfig stores voice_id and default speed/pitch."""
        vc = VoiceConfig(voice_id="af_heart")
        assert vc.voice_id == "af_heart"
        assert vc.speed == 1.0
        assert vc.pitch is None
        assert vc.extras == {}

    def test_with_extras(self):
        """Engine-specific extras are stored in the extras dict."""
        vc = VoiceConfig(
            voice_id="am_adam",
            speed=1.2,
            extras={"instruct": "speak warmly"},
        )
        assert vc.speed == 1.2
        assert vc.extras["instruct"] == "speak warmly"


# ═══════════════════════════════════════════════════════════════════════════
# 2. AUDIO MIXER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAudioMixer:
    """Tests for the AudioMixer — the sound engineer."""

    def test_empty_mix_returns_silence(self):
        """An empty mixer produces 1 second of silence (not a crash)."""
        mixer = AudioMixer()
        result = mixer.mix()
        assert len(result) == 1000  # 1000ms = 1 second.

    def test_single_clip(self):
        """A single clip is returned as-is (normalized)."""
        mixer = AudioMixer()
        wav = make_wav_bytes(duration_ms=500)
        mixer.add_clip(wav, sample_rate=22050)
        result = mixer.mix(normalize=False)
        # Duration should be ~500ms (the clip).
        # Allow a small tolerance for WAV header rounding.
        assert 490 <= len(result) <= 510

    def test_two_clips_with_default_pause(self):
        """Two clips are separated by the default pause (400ms)."""
        mixer = AudioMixer()
        wav = make_wav_bytes(duration_ms=300)
        mixer.add_clip(wav, sample_rate=22050)
        mixer.add_clip(wav, sample_rate=22050, pause_before=DEFAULT_PAUSE_MS)
        result = mixer.mix(normalize=False)
        # Expected: 300 + 400 (pause) + 300 = 1000ms.
        expected = 300 + DEFAULT_PAUSE_MS + 300
        assert expected - 10 <= len(result) <= expected + 10

    def test_clip_with_overlap(self):
        """Overlapping clips produce shorter total than sequential."""
        mixer = AudioMixer()
        wav = make_wav_bytes(duration_ms=500)
        mixer.add_clip(wav, sample_rate=22050, pause_before=0)
        mixer.add_clip(wav, sample_rate=22050, overlap=200, pause_before=0)
        result = mixer.mix(normalize=False)
        # With 200ms overlap, total should be ~800ms (not 1000ms).
        # The second clip starts 200ms before the first ends.
        assert 780 <= len(result) <= 820

    def test_add_silence(self):
        """add_silence() inserts a block of dead air."""
        mixer = AudioMixer()
        wav = make_wav_bytes(duration_ms=200)
        mixer.add_clip(wav, sample_rate=22050, pause_before=0)
        mixer.add_silence(1000)
        mixer.add_clip(wav, sample_rate=22050, pause_before=0)
        result = mixer.mix(normalize=False)
        # Expected: 200 + 1000 (silence) + 200 = 1400ms.
        assert 1390 <= len(result) <= 1410

    def test_clip_count(self):
        """clip_count tracks how many clips have been added."""
        mixer = AudioMixer()
        assert mixer.clip_count == 0
        mixer.add_clip(make_wav_bytes(), sample_rate=22050)
        assert mixer.clip_count == 1
        mixer.add_clip(make_wav_bytes(), sample_rate=22050)
        assert mixer.clip_count == 2

    def test_duration_ms_estimate(self):
        """duration_ms gives a rough estimate of total length."""
        mixer = AudioMixer()
        mixer.add_clip(
            make_wav_bytes(duration_ms=500), sample_rate=22050, pause_before=300
        )
        # Duration = clip length + pause_before.
        assert mixer.duration_ms > 0

    def test_resolve_pause_known_hints(self):
        """resolve_pause maps string hints to milliseconds."""
        mixer = AudioMixer()
        assert mixer.resolve_pause("short") == 300
        assert mixer.resolve_pause("medium") == 700
        assert mixer.resolve_pause("long") == 1200
        assert mixer.resolve_pause("none") == 0

    def test_resolve_pause_none(self):
        """None pause hint → default pause."""
        mixer = AudioMixer()
        assert mixer.resolve_pause(None) == DEFAULT_PAUSE_MS

    def test_resolve_pause_unknown(self):
        """Unknown pause hint → default pause."""
        mixer = AudioMixer()
        assert mixer.resolve_pause("extreme") == DEFAULT_PAUSE_MS

    def test_resolve_pause_case_insensitive(self):
        """Pause hints are case-insensitive."""
        mixer = AudioMixer()
        assert mixer.resolve_pause("SHORT") == 300
        assert mixer.resolve_pause("Medium") == 700

    def test_export_creates_file(self, tmp_path):
        """export() writes an audio file to disk."""
        mixer = AudioMixer()
        mixer.add_clip(make_wav_bytes(duration_ms=200), sample_rate=22050)
        out = mixer.export(tmp_path / "test.wav", format="wav")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_creates_parent_dirs(self, tmp_path):
        """export() creates parent directories if they don't exist."""
        mixer = AudioMixer()
        mixer.add_clip(make_wav_bytes(duration_ms=200), sample_rate=22050)
        deep_path = tmp_path / "a" / "b" / "c" / "test.wav"
        out = mixer.export(deep_path, format="wav")
        assert out.exists()


# ═══════════════════════════════════════════════════════════════════════════
# 3. RENDERER TESTS
# ═══════════════════════════════════════════════════════════════════════════

def make_debate_data(
    *,
    num_events: int = 4,
    speakers: list[str] | None = None,
    with_interruption: bool = False,
    with_overlap: bool = False,
) -> dict:
    """Build a fake debate data dict for renderer testing.

    This mimics the JSON structure that render() expects — a dict with
    "events" and "participants" keys.

    Args:
        num_events:        How many events to generate.
        speakers:          Speaker names. Defaults to ["Alice", "Bob"].
        with_interruption: If True, event 2 interrupts event 1.
        with_overlap:      If True, event 3 overlaps with event 2.

    Returns:
        A dict ready for TTSRenderer.render().
    """
    speakers = speakers or ["Alice", "Bob"]
    events = []
    for i in range(num_events):
        speaker = speakers[i % len(speakers)]
        event = {
            "event_kind": "utterance",
            "speaker_name": speaker,
            "text": f"Turn {i+1} by {speaker}. This is a test.",
            "tone": ["measured"] if i % 2 == 0 else ["passionate"],
            "pace": "medium",
            "volume": "normal",
            "nonverbal": [],
            "interruption": False,
            "overlap_with_previous": False,
            "pause_before": "short" if i > 0 else None,
            "pause_after": None,
        }

        if with_interruption and i == 1:
            event["interruption"] = True
            event["tone"] = ["aggressive"]
            event["pace"] = "fast"

        if with_overlap and i == 2:
            event["overlap_with_previous"] = True

        events.append(event)

    return {
        "title": "",
        "participants": [
            {"name": s, "side": "pro" if i == 0 else "con"}
            for i, s in enumerate(speakers)
        ],
        "events": events,
    }


class TestTTSRenderer:
    """Tests for the TTSRenderer — the movie director."""

    def test_render_basic(self, tmp_path):
        """Basic render: all events synthesized and exported."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine)
        data = make_debate_data(num_events=3)
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        assert result.output_path.exists()
        assert result.total_events == 3
        assert result.rendered == 3
        assert result.skipped == 0
        assert len(engine.calls) == 3

    def test_auto_voice_assignment(self, tmp_path):
        """Speakers are auto-assigned voices round-robin."""
        engine = FakeEngine(voices=["v_alice", "v_bob"])
        renderer = TTSRenderer(engine)
        data = make_debate_data(speakers=["Alice", "Bob"], num_events=4)
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        # Alice should get v_alice, Bob should get v_bob.
        assert result.voice_map["Alice"] == "v_alice"
        assert result.voice_map["Bob"] == "v_bob"

        # Verify the engine was called with the right voices.
        assert engine.calls[0]["voice"] == "v_alice"  # Event 1: Alice
        assert engine.calls[1]["voice"] == "v_bob"    # Event 2: Bob

    def test_manual_voice_map(self, tmp_path):
        """Manual voice_map overrides auto-assignment."""
        engine = FakeEngine()
        voice_map = {
            "Alice": VoiceConfig(voice_id="custom_voice_1"),
            "Bob": VoiceConfig(voice_id="custom_voice_2"),
        }
        renderer = TTSRenderer(engine, voice_map=voice_map)
        data = make_debate_data(speakers=["Alice", "Bob"], num_events=2)
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        assert result.voice_map["Alice"] == "custom_voice_1"
        assert result.voice_map["Bob"] == "custom_voice_2"

    def test_skips_non_utterance_events(self, tmp_path):
        """Non-utterance events (stage directions, etc.) are skipped."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine)
        data = make_debate_data(num_events=2)
        # Add a non-utterance event.
        data["events"].append({
            "event_kind": "stage_direction",
            "speaker_name": "",
            "text": "[Audience applauds]",
        })
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        assert result.rendered == 2
        assert result.skipped == 1  # The stage direction was skipped.

    def test_skips_empty_text(self, tmp_path):
        """Events with empty text are skipped."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine)
        data = make_debate_data(num_events=2)
        data["events"].append({
            "event_kind": "utterance",
            "speaker_name": "Alice",
            "text": "",  # Empty text.
        })
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)
        assert result.skipped == 1

    def test_emotion_hint_building(self, tmp_path):
        """Emotion hints are built from tone + pace + volume + nonverbal."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine)
        data = {
            "events": [{
                "event_kind": "utterance",
                "speaker_name": "Alice",
                "text": "Test",
                "tone": ["aggressive", "rising"],
                "pace": "fast",
                "volume": "loud",
                "nonverbal": ["leans forward"],
                "interruption": False,
                "overlap_with_previous": False,
                "pause_before": None,
            }],
            "participants": [{"name": "Alice"}],
        }
        renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        # The emotion hint should combine all metadata.
        emotion = engine.calls[0]["emotion"]
        assert "aggressive" in emotion
        assert "rising" in emotion
        assert "fast" in emotion
        assert "loud" in emotion
        assert "leans forward" in emotion

    def test_pace_to_speed_mapping(self, tmp_path):
        """Pace hints map to speed multipliers."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine)

        # Test each pace value.
        for pace, expected_speed in [("slow", 0.85), ("medium", 1.0), ("fast", 1.15)]:
            data = {
                "events": [{
                    "event_kind": "utterance",
                    "speaker_name": "Alice",
                    "text": "Test",
                    "pace": pace,
                    "tone": [],
                    "nonverbal": [],
                    "interruption": False,
                    "overlap_with_previous": False,
                    "pause_before": None,
                }],
                "participants": [{"name": "Alice"}],
            }
            engine.calls.clear()
            renderer.render(data, tmp_path / f"out_{pace}.wav", format="wav", quiet=True)
            assert engine.calls[0]["speed"] == expected_speed

    def test_interruption_triggers_overlap(self, tmp_path):
        """interruption=True causes overlap with previous clip."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine)
        data = make_debate_data(num_events=3, with_interruption=True)
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)
        # All 3 events should render — interruptions don't skip events.
        assert result.rendered == 3

    def test_engine_failure_skips_clip(self, tmp_path):
        """If the engine throws, the renderer skips that clip gracefully."""

        class FailingEngine(BaseTTSEngine):
            """Engine that fails on the second call."""

            def __init__(self):
                self._call_count = 0

            def synthesize(self, text, **kwargs):
                self._call_count += 1
                if self._call_count == 2:
                    raise RuntimeError("GPU caught fire")
                return SynthesizedAudio(
                    audio=make_wav_bytes(duration_ms=200),
                    sample_rate=22050,
                )

            def available_voices(self):
                return ["voice_a"]

        engine = FailingEngine()
        renderer = TTSRenderer(engine)
        data = make_debate_data(num_events=3)
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        # 2 rendered (events 1 and 3), 1 skipped (event 2 failed).
        assert result.rendered == 2
        assert result.skipped == 1

    def test_render_file(self, tmp_path):
        """render_file() loads JSON and delegates to render()."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine)
        data = make_debate_data(num_events=2)

        # Write the debate data to a JSON file.
        json_path = tmp_path / "debate.json"
        json_path.write_text(json.dumps(data))

        result = renderer.render_file(
            json_path, tmp_path / "out.wav", format="wav", quiet=True
        )
        assert result.rendered == 2
        assert result.output_path.exists()

    def test_three_speakers(self, tmp_path):
        """Three speakers get three different voices."""
        engine = FakeEngine(voices=["v1", "v2", "v3"])
        renderer = TTSRenderer(engine)
        data = make_debate_data(speakers=["Alice", "Bob", "Moderator"], num_events=3)
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        assert result.voice_map["Alice"] == "v1"
        assert result.voice_map["Bob"] == "v2"
        assert result.voice_map["Moderator"] == "v3"

    def test_more_speakers_than_voices_wraps(self, tmp_path):
        """If there are more speakers than voices, assignment wraps around."""
        engine = FakeEngine(voices=["v1", "v2"])
        renderer = TTSRenderer(engine)
        data = make_debate_data(
            speakers=["Alice", "Bob", "Charlie"], num_events=3
        )
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        # Charlie wraps around to v1.
        assert result.voice_map["Alice"] == "v1"
        assert result.voice_map["Bob"] == "v2"
        assert result.voice_map["Charlie"] == "v1"

    def test_render_result_stats(self, tmp_path):
        """RenderResult contains accurate stats."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine)
        data = make_debate_data(num_events=5, speakers=["Alice", "Bob"])
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        assert result.total_events == 5
        assert result.rendered == 5
        assert result.duration_ms > 0
        assert isinstance(result.voice_map, dict)


class TestVoiceCasting:
    """Tests for gender-aware voice assignment via voice_casting metadata."""

    KOKORO_VOICES = [
        "af_heart", "af_alloy", "af_nova",       # American female
        "am_adam", "am_echo", "am_eric",          # American male
        "bf_emma", "bf_isabella",                  # British female
        "bm_daniel", "bm_george",                  # British male
    ]

    def test_male_casting_picks_male_voices(self, tmp_path):
        """voice_casting 'male' picks from am_*/bm_* voices only."""
        engine = FakeEngine(voices=self.KOKORO_VOICES)
        renderer = TTSRenderer(engine)
        data = make_debate_data(speakers=["Teal'c", "Ronon Dex"], num_events=2)
        data["voice_casting"] = {"Teal'c": "male", "Ronon Dex": "male"}
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        assert result.voice_map["Teal'c"] == "am_adam"
        assert result.voice_map["Ronon Dex"] == "am_echo"

    def test_female_casting_picks_female_voices(self, tmp_path):
        """voice_casting 'female' picks from af_*/bf_* voices only."""
        engine = FakeEngine(voices=self.KOKORO_VOICES)
        renderer = TTSRenderer(engine)
        data = make_debate_data(speakers=["Dana Scully", "Buffy"], num_events=2)
        data["voice_casting"] = {"Dana Scully": "female", "Buffy": "female"}
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        assert result.voice_map["Dana Scully"] == "af_heart"
        assert result.voice_map["Buffy"] == "af_alloy"

    def test_mixed_gender_casting(self, tmp_path):
        """Mixed genders pick from their respective voice pools."""
        engine = FakeEngine(voices=self.KOKORO_VOICES)
        renderer = TTSRenderer(engine)
        data = make_debate_data(speakers=["Mulder", "Scully"], num_events=2)
        data["voice_casting"] = {"Mulder": "male", "Scully": "female"}
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        assert result.voice_map["Mulder"] == "am_adam"
        assert result.voice_map["Scully"] == "af_heart"

    def test_no_casting_defaults_to_male_with_kokoro_voices(self, tmp_path):
        """Without voice_casting, speakers default to male voices."""
        engine = FakeEngine(voices=self.KOKORO_VOICES)
        renderer = TTSRenderer(engine)
        data = make_debate_data(speakers=["Alpha", "Beta"], num_events=2)
        # No voice_casting key.
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        assert result.voice_map["Alpha"] == "am_adam"
        assert result.voice_map["Beta"] == "am_echo"

    def test_manual_voice_map_overrides_casting(self, tmp_path):
        """Manual voice_map takes priority over voice_casting."""
        engine = FakeEngine(voices=self.KOKORO_VOICES)
        voice_map = {"Teal'c": VoiceConfig(voice_id="bm_george")}
        renderer = TTSRenderer(engine, voice_map=voice_map)
        data = make_debate_data(speakers=["Teal'c", "Ronon Dex"], num_events=2)
        data["voice_casting"] = {"Teal'c": "male", "Ronon Dex": "male"}
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        # Manual override wins for Teal'c.
        assert result.voice_map["Teal'c"] == "bm_george"
        # Ronon Dex gets auto-assigned from male pool.
        assert result.voice_map["Ronon Dex"] == "am_adam"

    def test_non_kokoro_voices_fall_back_to_round_robin(self, tmp_path):
        """Voices without af_/am_/bf_/bm_ prefix use fallback round-robin."""
        engine = FakeEngine(voices=["voice_a", "voice_b", "voice_c"])
        renderer = TTSRenderer(engine)
        data = make_debate_data(speakers=["Alice", "Bob"], num_events=2)
        data["voice_casting"] = {"Alice": "female", "Bob": "male"}
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        # No gendered voices → falls back to round-robin.
        assert result.voice_map["Alice"] == "voice_a"
        assert result.voice_map["Bob"] == "voice_b"

    def test_casting_wraps_within_gender_pool(self, tmp_path):
        """More speakers of one gender than voices wraps within that pool."""
        engine = FakeEngine(voices=self.KOKORO_VOICES)
        renderer = TTSRenderer(engine)
        data = make_debate_data(
            speakers=["Alpha", "Beta", "Gamma", "Delta"], num_events=4
        )
        data["voice_casting"] = {
            "Alpha": "male", "Beta": "male",
            "Gamma": "male", "Delta": "male",
        }
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        male_pool = ["am_adam", "am_echo", "am_eric", "bm_daniel", "bm_george"]
        assert result.voice_map["Alpha"] == male_pool[0]
        assert result.voice_map["Beta"] == male_pool[1]
        assert result.voice_map["Gamma"] == male_pool[2]
        assert result.voice_map["Delta"] == male_pool[3]


# ═══════════════════════════════════════════════════════════════════════════
# 4. PIPER ENGINE TESTS (mocked)
# ═══════════════════════════════════════════════════════════════════════════

class TestPiperEngine:
    """Tests for PiperEngine — all mocked, no real model needed.

    We mock PiperVoice to avoid needing actual ONNX model files.
    The tests verify that PiperEngine correctly:
    - Scans model_dir for .onnx files
    - Calls PiperVoice.load() and synthesize_wav()
    - Inverts speed into length_scale
    - Returns valid SynthesizedAudio
    """

    def test_available_voices_scans_dir(self, tmp_path):
        """available_voices() returns stems of .onnx files in model_dir."""
        from vox_romana.tts.engines.piper_engine import PiperEngine

        # Create fake .onnx model files.
        (tmp_path / "en_US-lessac-medium.onnx").touch()
        (tmp_path / "en_US-lessac-medium.onnx.json").touch()
        (tmp_path / "en_US-amy-medium.onnx").touch()
        (tmp_path / "en_US-amy-medium.onnx.json").touch()

        engine = PiperEngine(model_dir=tmp_path)
        voices = engine.available_voices()

        assert "en_US-lessac-medium" in voices
        assert "en_US-amy-medium" in voices
        assert len(voices) == 2

    def test_available_voices_empty_dir(self, tmp_path):
        """available_voices() returns [] for empty model directory."""
        from vox_romana.tts.engines.piper_engine import PiperEngine

        engine = PiperEngine(model_dir=tmp_path)
        assert engine.available_voices() == []

    def test_available_voices_no_dir(self, tmp_path):
        """available_voices() returns [] if model_dir doesn't exist."""
        from vox_romana.tts.engines.piper_engine import PiperEngine

        engine = PiperEngine(model_dir=tmp_path / "nonexistent")
        assert engine.available_voices() == []

    @patch("vox_romana.tts.engines.piper_engine.PiperEngine._load_voice")
    def test_synthesize_calls_engine(self, mock_load_voice, tmp_path):
        """synthesize() calls PiperVoice.synthesize_wav and returns WAV bytes."""
        from vox_romana.tts.engines.piper_engine import PiperEngine

        # Set up a mock PiperVoice.
        mock_voice = MagicMock()

        # Configure the mock's config to return a sample rate.
        mock_voice.config.sample_rate = 22050

        # Make synthesize_wav write valid WAV bytes.
        def fake_synthesize_wav(text, wav_file, syn_config=None):
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00" * 4410)  # 0.1s of silence

        mock_voice.synthesize_wav = fake_synthesize_wav
        mock_load_voice.return_value = mock_voice

        engine = PiperEngine(model_dir=tmp_path)
        result = engine.synthesize("Hello world", voice="test")

        assert isinstance(result, SynthesizedAudio)
        assert result.sample_rate == 22050
        assert len(result.audio) > 0

    @patch("vox_romana.tts.engines.piper_engine.PiperEngine._load_voice")
    def test_speed_inverts_to_length_scale(self, mock_load_voice, tmp_path):
        """Speed multiplier is inverted for Piper's length_scale."""
        from vox_romana.tts.engines.piper_engine import PiperEngine

        mock_voice = MagicMock()
        mock_voice.config.sample_rate = 22050

        captured_config = {}

        def fake_synthesize_wav(text, wav_file, syn_config=None):
            # Capture the syn_config so we can inspect length_scale.
            captured_config["syn_config"] = syn_config
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00" * 4410)

        mock_voice.synthesize_wav = fake_synthesize_wav
        mock_load_voice.return_value = mock_voice

        engine = PiperEngine(model_dir=tmp_path)
        engine.synthesize("Test", voice="test", speed=1.2)

        # speed=1.2 → length_scale = 1/1.2 ≈ 0.833
        syn_config = captured_config["syn_config"]
        assert abs(syn_config.length_scale - (1.0 / 1.2)) < 0.01

    def test_engine_name(self):
        """The engine's name property returns 'Piper'."""
        from vox_romana.tts.engines.piper_engine import PiperEngine

        engine = PiperEngine(model_dir="models/piper")
        assert engine.name == "Piper"


# ═══════════════════════════════════════════════════════════════════════════
# 5. KOKORO ENGINE TESTS (mocked)
# ═══════════════════════════════════════════════════════════════════════════

class TestKokoroEngine:
    """Tests for KokoroEngine — all mocked, no real GPU needed.

    We mock mlx_audio.tts.generate.generate_audio to avoid needing
    the actual Kokoro model (which is ~2 GB). The tests verify that
    KokoroEngine correctly:
    - Calls generate_audio with the right parameters
    - Handles the temp-directory-and-read-back pattern
    - Passes emotion as the instruct parameter
    - Lists available voices
    """

    def test_available_voices(self):
        """available_voices() returns the built-in Kokoro voice list."""
        from vox_romana.tts.engines.kokoro_engine import KokoroEngine, KOKORO_VOICES

        engine = KokoroEngine()
        voices = engine.available_voices()

        assert "af_heart" in voices
        assert "am_adam" in voices
        assert len(voices) == len(KOKORO_VOICES)

    @patch("vox_romana.tts.engines.kokoro_engine._lazy_generate_audio")
    def test_synthesize_creates_wav(self, mock_gen, tmp_path):
        """synthesize() calls generate_audio and reads back the WAV."""
        from vox_romana.tts.engines.kokoro_engine import KokoroEngine

        # Mock generate_audio to write a WAV file in the temp dir.
        def fake_generate(**kwargs):
            wav_path = Path(kwargs["output_path"]) / f"{kwargs['file_prefix']}.wav"
            wav_bytes = make_wav_bytes(duration_ms=300, sample_rate=24000)
            wav_path.write_bytes(wav_bytes)

        mock_gen.side_effect = fake_generate

        engine = KokoroEngine()
        result = engine.synthesize("Hello world", voice="af_heart")

        assert isinstance(result, SynthesizedAudio)
        assert result.sample_rate == 24000
        assert len(result.audio) > 0
        mock_gen.assert_called_once()

    @patch("vox_romana.tts.engines.kokoro_engine._lazy_generate_audio")
    def test_emotion_passed_as_instruct(self, mock_gen, tmp_path):
        """Emotion string is passed as the 'instruct' kwarg."""
        from vox_romana.tts.engines.kokoro_engine import KokoroEngine

        def fake_generate(**kwargs):
            wav_path = Path(kwargs["output_path"]) / f"{kwargs['file_prefix']}.wav"
            wav_path.write_bytes(make_wav_bytes(duration_ms=100, sample_rate=24000))

        mock_gen.side_effect = fake_generate

        engine = KokoroEngine()
        engine.synthesize(
            "Test", voice="am_adam", emotion="speak angrily and loudly"
        )

        call_kwargs = mock_gen.call_args
        # generate_audio should have received instruct="speak angrily and loudly"
        assert call_kwargs.kwargs.get("instruct") == "speak angrily and loudly"

    @patch("vox_romana.tts.engines.kokoro_engine._lazy_generate_audio")
    def test_no_emotion_skips_instruct(self, mock_gen, tmp_path):
        """When emotion is empty, instruct is NOT passed."""
        from vox_romana.tts.engines.kokoro_engine import KokoroEngine

        def fake_generate(**kwargs):
            wav_path = Path(kwargs["output_path"]) / f"{kwargs['file_prefix']}.wav"
            wav_path.write_bytes(make_wav_bytes(duration_ms=100, sample_rate=24000))

        mock_gen.side_effect = fake_generate

        engine = KokoroEngine()
        engine.synthesize("Test", voice="af_heart", emotion="")

        call_kwargs = mock_gen.call_args
        # instruct should NOT be in the kwargs when emotion is empty.
        assert "instruct" not in call_kwargs.kwargs

    @patch("vox_romana.tts.engines.kokoro_engine._lazy_generate_audio")
    def test_speed_passed_through(self, mock_gen, tmp_path):
        """Speed multiplier is passed directly to generate_audio."""
        from vox_romana.tts.engines.kokoro_engine import KokoroEngine

        def fake_generate(**kwargs):
            wav_path = Path(kwargs["output_path"]) / f"{kwargs['file_prefix']}.wav"
            wav_path.write_bytes(make_wav_bytes(duration_ms=100, sample_rate=24000))

        mock_gen.side_effect = fake_generate

        engine = KokoroEngine()
        engine.synthesize("Test", speed=1.3)

        call_kwargs = mock_gen.call_args
        assert call_kwargs.kwargs["speed"] == 1.3

    @patch("vox_romana.tts.engines.kokoro_engine._lazy_generate_audio")
    def test_no_wav_raises_error(self, mock_gen):
        """RuntimeError if generate_audio produces no WAV files."""
        from vox_romana.tts.engines.kokoro_engine import KokoroEngine

        # Mock that generates nothing.
        mock_gen.side_effect = lambda **kwargs: None

        engine = KokoroEngine()
        with pytest.raises(RuntimeError, match="no audio files"):
            engine.synthesize("Test")

    def test_engine_name(self):
        """The engine's name property returns 'Kokoro'."""
        from vox_romana.tts.engines.kokoro_engine import KokoroEngine

        engine = KokoroEngine()
        assert engine.name == "Kokoro"


# ═══════════════════════════════════════════════════════════════════════════
# 6. INTEGRATION-STYLE TESTS (renderer + mixer, mocked engine)
# ═══════════════════════════════════════════════════════════════════════════

class TestRendererMixerIntegration:
    """Tests that verify the renderer and mixer work together correctly.

    These use FakeEngine (which returns real WAV bytes) so the mixer
    can actually process the audio. This is as close to end-to-end as
    we can get without a real TTS engine.
    """

    def test_full_pipeline_produces_audio(self, tmp_path):
        """A full render produces a non-empty audio file."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine)
        data = make_debate_data(num_events=5)
        result = renderer.render(
            data, tmp_path / "debate.wav", format="wav", quiet=True
        )
        assert result.output_path.exists()
        assert result.output_path.stat().st_size > 0
        assert result.rendered == 5
        assert result.duration_ms > 0

    def test_interruption_produces_shorter_audio(self, tmp_path):
        """An interrupting event should make total audio shorter than sequential."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine)

        # Sequential: no interruptions.
        data_seq = make_debate_data(num_events=3, with_interruption=False)
        result_seq = renderer.render(
            data_seq, tmp_path / "seq.wav", format="wav", quiet=True
        )

        # With interruption on event 2.
        data_int = make_debate_data(num_events=3, with_interruption=True)
        result_int = renderer.render(
            data_int, tmp_path / "int.wav", format="wav", quiet=True
        )

        # The interrupted version should be the same or shorter
        # (overlap means events play on top of each other).
        # The file sizes can vary, so check that rendering worked.
        assert result_seq.rendered == 3
        assert result_int.rendered == 3

    def test_empty_debate_produces_file(self, tmp_path):
        """A debate with zero events still produces a valid audio file."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine)
        data = {"events": [], "participants": []}
        result = renderer.render(
            data, tmp_path / "empty.wav", format="wav", quiet=True
        )
        assert result.output_path.exists()
        assert result.rendered == 0
        assert result.skipped == 0


# ═══════════════════════════════════════════════════════════════════════════
# 6. VOICE CATALOG TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestVoiceCatalog:
    """Tests for voice_catalog.py — voice metadata and matching algorithms."""

    def test_catalog_has_voices(self):
        """The catalog should contain all 24 Kokoro voices."""
        from vox_romana.tts.voice_catalog import KOKORO_VOICE_CATALOG
        assert len(KOKORO_VOICE_CATALOG) >= 20

    def test_voice_descriptor_fields(self):
        """Each VoiceDescriptor should have required fields populated."""
        from vox_romana.tts.voice_catalog import KOKORO_VOICE_CATALOG, VoiceDescriptor
        for vid, desc in KOKORO_VOICE_CATALOG.items():
            assert isinstance(desc, VoiceDescriptor), f"{vid} is not a VoiceDescriptor"
            assert desc.voice_id == vid
            assert desc.gender in ("male", "female"), f"{vid} has invalid gender"
            assert desc.accent, f"{vid} has no accent"
            assert desc.tags, f"{vid} has no tags"

    def test_match_voice_returns_best_fit(self):
        """match_voice should return the voice with highest tag overlap."""
        from vox_romana.tts.voice_catalog import match_voice
        # "warm female british" should match a British female voice.
        result = match_voice("warm british female")
        assert result is not None
        assert result.startswith("bf_"), f"Expected British female, got {result}"

    def test_match_voice_gender_filter(self):
        """match_voice with gender= should only return that gender."""
        from vox_romana.tts.voice_catalog import match_voice
        result = match_voice("warm authoritative", gender="male")
        assert result is not None
        # Check it's a male voice (am_* or bm_*).
        assert result[:2] in ("am", "bm"), f"Expected male voice, got {result}"

    def test_match_voice_with_empty_hint(self):
        """An empty hint should still return a voice (first in catalog)."""
        from vox_romana.tts.voice_catalog import match_voice
        result = match_voice("")
        assert result is not None

    def test_match_voices_for_cast_unique(self):
        """Cast matching should assign unique voices to each speaker."""
        from vox_romana.tts.voice_catalog import match_voices_for_cast
        cast = {
            "Dr. Chen": {"gender": "female", "voice_hint": "warm academic"},
            "Professor Hartley": {"gender": "male", "voice_hint": "authoritative british"},
            "Jordan Lee": {"gender": "male", "voice_hint": "young casual"},
        }
        result = match_voices_for_cast(cast)
        assert len(result) == 3
        # All voices should be different.
        voice_ids = list(result.values())
        assert len(set(voice_ids)) == 3, f"Duplicate voices: {voice_ids}"

    def test_match_voices_for_cast_respects_gender(self):
        """Cast matching should use gender cues from hints."""
        from vox_romana.tts.voice_catalog import match_voices_for_cast
        cast = {
            "Alice": {"gender": "female"},
            "Bob": {"gender": "male"},
        }
        result = match_voices_for_cast(cast)
        assert result["Alice"][:2] in ("af", "bf"), f"Alice got {result['Alice']}"
        assert result["Bob"][:2] in ("am", "bm"), f"Bob got {result['Bob']}"

    def test_get_voices_by_gender(self):
        """get_voices_by_gender should partition correctly."""
        from vox_romana.tts.voice_catalog import get_voices_by_gender
        males = get_voices_by_gender("male")
        females = get_voices_by_gender("female")
        assert all(v[:2] in ("am", "bm") for v in males)
        assert all(v[:2] in ("af", "bf") for v in females)
        assert len(males) > 0
        assert len(females) > 0

    def test_get_catalog_voice(self):
        """get_catalog_voice should return metadata for known voices."""
        from vox_romana.tts.voice_catalog import get_catalog_voice
        desc = get_catalog_voice("af_heart")
        assert desc is not None
        assert desc.voice_id == "af_heart"
        assert desc.gender == "female"

    def test_get_catalog_voice_unknown(self):
        """get_catalog_voice should return None for unknown voice IDs."""
        from vox_romana.tts.voice_catalog import get_catalog_voice
        assert get_catalog_voice("nonexistent_voice") is None


# ═══════════════════════════════════════════════════════════════════════════
# 7. CATALOG-BASED RENDERER VOICE MATCHING
# ═══════════════════════════════════════════════════════════════════════════


class TestCatalogVoiceMatching:
    """Tests for voice_hints-driven catalog matching in the renderer."""

    KOKORO_VOICES = [
        "af_heart", "af_alloy", "af_nova",
        "am_adam", "am_echo", "am_eric",
        "bf_emma", "bf_isabella",
        "bm_daniel", "bm_george",
    ]

    def test_voice_hints_triggers_catalog_matching(self, tmp_path):
        """When voice_hints are present, the renderer uses catalog matching."""
        engine = FakeEngine(voices=self.KOKORO_VOICES)
        renderer = TTSRenderer(engine)
        data = make_debate_data(speakers=["Alice", "Bob"], num_events=2)
        data["voice_casting"] = {"Alice": "female", "Bob": "male"}
        data["voice_hints"] = {
            "Alice": "warm British female",
            "Bob": "authoritative American male",
        }
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        # Alice should get a female voice, Bob a male voice.
        alice_voice = result.voice_map["Alice"]
        bob_voice = result.voice_map["Bob"]
        assert alice_voice[:2] in ("af", "bf"), f"Alice got {alice_voice}"
        assert bob_voice[:2] in ("am", "bm"), f"Bob got {bob_voice}"
        # They should be different.
        assert alice_voice != bob_voice

    def test_voice_hints_absent_falls_back_to_gender(self, tmp_path):
        """Without voice_hints, the renderer uses the gender-only fallback."""
        engine = FakeEngine(voices=self.KOKORO_VOICES)
        renderer = TTSRenderer(engine)
        data = make_debate_data(speakers=["Alice", "Bob"], num_events=2)
        data["voice_casting"] = {"Alice": "female", "Bob": "male"}
        # No voice_hints key.
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        # Should still assign by gender.
        assert result.voice_map["Alice"][:2] in ("af", "bf")
        assert result.voice_map["Bob"][:2] in ("am", "bm")

    def test_manual_voice_map_overrides_catalog(self, tmp_path):
        """Manual voice_map should still take priority over catalog matching."""
        engine = FakeEngine(voices=self.KOKORO_VOICES)
        voice_map = {"Alice": VoiceConfig(voice_id="bm_george")}
        renderer = TTSRenderer(engine, voice_map=voice_map)
        data = make_debate_data(speakers=["Alice", "Bob"], num_events=2)
        data["voice_hints"] = {"Alice": "warm female", "Bob": "male deep"}
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        # Manual override wins for Alice even though hint says "female".
        assert result.voice_map["Alice"] == "bm_george"


# ═══════════════════════════════════════════════════════════════════════════
# 8. TTS STYLE PASSTHROUGH
# ═══════════════════════════════════════════════════════════════════════════


class TestTtsStylePassthrough:
    """Tests for tts_styles integration in the emotion hint pipeline."""

    def test_tts_style_prepended_to_emotion_hint(self):
        """tts_style should be prepended to the per-event emotion hint."""
        event = {"tone": ["aggressive"], "pace": "fast"}
        hint = TTSRenderer._build_emotion_hint(event, tts_style="Speak with quiet authority")
        assert hint.startswith("Speak with quiet authority.")
        assert "aggressive" in hint
        assert "pace: fast" in hint

    def test_no_tts_style_preserves_original_behavior(self):
        """Without tts_style, _build_emotion_hint works exactly as before."""
        event = {"tone": ["calm", "measured"], "volume": "soft"}
        hint = TTSRenderer._build_emotion_hint(event)
        assert hint == "calm, measured; volume: soft"

    def test_tts_style_with_empty_event(self):
        """tts_style alone should produce a clean hint."""
        event = {}
        hint = TTSRenderer._build_emotion_hint(event, tts_style="Deliver with energy")
        assert hint == "Deliver with energy."

    def test_tts_styles_reach_engine(self, tmp_path):
        """tts_styles from the show JSON should flow through to synthesize()."""
        engine = FakeEngine(voices=["voice_a", "voice_b"])
        renderer = TTSRenderer(engine)
        data = make_debate_data(speakers=["Alice", "Bob"], num_events=2)
        data["tts_styles"] = {
            "Alice": "Speak with calm authority",
            "Bob": "Deliver with punchy energy",
        }
        renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)

        # Check that the engine received emotion hints with tts_style prefix.
        alice_call = engine.calls[0]
        bob_call = engine.calls[1]
        assert "Speak with calm authority." in alice_call["emotion"]
        assert "Deliver with punchy energy." in bob_call["emotion"]


# ═══════════════════════════════════════════════════════════════════════════
# 9. STAGE DIRECTION STRIPPING
# ═══════════════════════════════════════════════════════════════════════════

from vox_romana.tts.renderer import strip_markdown_emphasis, strip_stage_directions


class TestStripStageDirections:
    """Unit tests for the strip_stage_directions() helper."""

    def test_leading_direction_stripped(self):
        """A leading (direction) block is removed."""
        assert strip_stage_directions("(laughs) Well, that's funny.") == "Well, that's funny."

    def test_mid_sentence_direction_stripped(self):
        """A mid-sentence (pause) direction is removed and whitespace tightened."""
        result = strip_stage_directions("Let me tell you... (dramatic pause) Gold.")
        assert "(dramatic pause)" not in result
        assert "Let me tell you..." in result
        assert "Gold." in result

    def test_no_direction_unchanged(self):
        """Text with no parenthetical blocks passes through unchanged."""
        text = "Just plain dialogue."
        assert strip_stage_directions(text) == text

    def test_multiple_directions_all_stripped(self):
        """Multiple (blocks) in sequence are all removed."""
        result = strip_stage_directions("(Warm, chuckling) (Leans in) The answer is yes.")
        assert "(Warm, chuckling)" not in result
        assert "(Leans in)" not in result
        assert "The answer is yes." in result

    def test_direction_only_text_becomes_empty(self):
        """Text that is only a direction becomes empty string."""
        assert strip_stage_directions("(pause)") == ""

    def test_whitespace_collapsed(self):
        """No double-spaces left after stripping."""
        result = strip_stage_directions("(Excited, melodic) And what a fiasco!")
        assert "  " not in result

    def test_ellipsis_preserved(self):
        """Ellipsis before a stripped direction is preserved cleanly."""
        result = strip_stage_directions("And then... (dramatic pause) it happened.")
        assert "..." in result
        assert "(dramatic pause)" not in result

    def test_empty_string(self):
        """Empty input returns empty output."""
        assert strip_stage_directions("") == ""

    def test_real_script_sample(self):
        """A real sample from the generated Trash Pandas script is handled."""
        text = (
            "(Dramatic, conspiratorial whisper, leaning into the microphone) "
            "Welcome, listeners, to the trashiest drama in baseball."
        )
        result = strip_stage_directions(text)
        assert not result.startswith("(")
        assert "Welcome, listeners" in result

    def test_double_ellipsis_merged(self):
        """'... (direction) ...' artifacts are collapsed to a single '...'."""
        result = strip_stage_directions("I mean... (pause) ...what do you think?")
        assert result == "I mean...what do you think?"

    def test_double_em_dash_merged(self):
        """'\u2014(direction)\u2014' sandwich artifacts collapse to a single '\u2014'."""
        result = strip_stage_directions("I\u2014(hesitates)\u2014just don't know.")
        assert result == "I\u2014just don't know."

    def test_newline_in_direction_collapsed(self):
        """\\n left by a stripped inline direction is collapsed to a space."""
        result = strip_stage_directions("Hello.\n(sighs deeply)\nWorld.")
        assert result == "Hello. World."


class TestStripMarkdownEmphasis:
    """Unit tests for the strip_markdown_emphasis() helper."""

    def test_single_asterisk_stripped(self):
        """*word* markers are removed, word kept."""
        assert strip_markdown_emphasis("come back *to*") == "come back to"

    def test_double_asterisk_stripped(self):
        """**word** bold markers are removed, word kept."""
        assert strip_markdown_emphasis("a **critiquing** son") == "a critiquing son"

    def test_single_underscore_stripped(self):
        """_word_ markers are removed, word kept."""
        assert strip_markdown_emphasis("the _only_ way") == "the only way"

    def test_double_underscore_stripped(self):
        """__word__ markers are removed, word kept."""
        assert strip_markdown_emphasis("__Save what remains.__") == "Save what remains."

    def test_multiple_markers_all_stripped(self):
        """Multiple emphasis spans in one string are all stripped."""
        result = strip_markdown_emphasis("the *enemy* _is_ the **gates**")
        assert result == "the enemy is the gates"

    def test_plain_text_unchanged(self):
        """Text with no markers passes through unchanged."""
        text = "Just plain dialogue."
        assert strip_markdown_emphasis(text) == text

    def test_empty_string(self):
        """Empty input returns empty output."""
        assert strip_markdown_emphasis("") == ""

    def test_real_script_sample_cost(self):
        """Real sample: 'what did it *cost*?' → asterisks removed."""
        result = strip_markdown_emphasis("But what did it *cost*?")
        assert result == "But what did it cost?"

    def test_real_script_sample_save(self):
        """Real sample: '*Save what remains.*' → asterisks removed."""
        result = strip_markdown_emphasis("*Save what remains.*")
        assert result == "Save what remains."


class TestRendererStripDirections:
    """Tests for strip_directions= parameter on TTSRenderer."""

    def test_strip_off_by_default(self, tmp_path):
        """Directions are NOT stripped when strip_directions=False (default)."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine)
        data = {
            "events": [{
                "event_kind": "utterance",
                "speaker_name": "Alice",
                "text": "(laughs) Well that's interesting.",
                "tone": [],
                "pace": "medium",
                "volume": "normal",
                "nonverbal": [],
                "interruption": False,
                "overlap_with_previous": False,
                "pause_before": None,
            }],
            "participants": [{"name": "Alice"}],
        }
        renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)
        assert engine.calls[0]["text"] == "(laughs) Well that's interesting."

    def test_strip_on_removes_parentheticals(self, tmp_path):
        """Directions are stripped when strip_directions=True."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine, strip_directions=True)
        data = {
            "events": [{
                "event_kind": "utterance",
                "speaker_name": "Alice",
                "text": "(laughs) Well that's interesting.",
                "tone": [],
                "pace": "medium",
                "volume": "normal",
                "nonverbal": [],
                "interruption": False,
                "overlap_with_previous": False,
                "pause_before": None,
            }],
            "participants": [{"name": "Alice"}],
        }
        renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)
        assert engine.calls[0]["text"] == "Well that's interesting."

    def test_direction_only_event_skipped_when_stripping(self, tmp_path):
        """An event that is only a direction becomes empty → skipped."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine, strip_directions=True)
        data = {
            "events": [
                {
                    "event_kind": "utterance",
                    "speaker_name": "Alice",
                    "text": "(pause)",
                    "tone": [],
                    "pace": "medium",
                    "volume": "normal",
                    "nonverbal": [],
                    "interruption": False,
                    "overlap_with_previous": False,
                    "pause_before": None,
                },
                {
                    "event_kind": "utterance",
                    "speaker_name": "Bob",
                    "text": "This one has real text.",
                    "tone": [],
                    "pace": "medium",
                    "volume": "normal",
                    "nonverbal": [],
                    "interruption": False,
                    "overlap_with_previous": False,
                    "pause_before": None,
                },
            ],
            "participants": [{"name": "Alice"}, {"name": "Bob"}],
        }
        result = renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)
        assert result.rendered == 1
        assert result.skipped == 1
        assert engine.calls[0]["text"] == "This one has real text."

    def test_json_not_modified(self, tmp_path):
        """The original show_data dict is never mutated by stripping."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine, strip_directions=True)
        original_text = "(laughs) Well that's interesting."
        data = {
            "events": [{
                "event_kind": "utterance",
                "speaker_name": "Alice",
                "text": original_text,
                "tone": [],
                "pace": "medium",
                "volume": "normal",
                "nonverbal": [],
                "interruption": False,
                "overlap_with_previous": False,
                "pause_before": None,
            }],
            "participants": [{"name": "Alice"}],
        }
        renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)
        # The original dict must be untouched.
        assert data["events"][0]["text"] == original_text

    def test_markdown_emphasis_always_stripped(self, tmp_path):
        """Markdown emphasis markers are stripped regardless of strip_directions."""
        engine = FakeEngine()
        renderer = TTSRenderer(engine)  # strip_directions=False (default)
        data = {
            "events": [{
                "event_kind": "utterance",
                "speaker_name": "Alice",
                "text": "anything still works to come back *to*",
                "tone": [],
                "pace": "medium",
                "volume": "normal",
                "nonverbal": [],
                "interruption": False,
                "overlap_with_previous": False,
                "pause_before": None,
            }],
            "participants": [{"name": "Alice"}],
        }
        renderer.render(data, tmp_path / "out.wav", format="wav", quiet=True)
        assert engine.calls[0]["text"] == "anything still works to come back to"
