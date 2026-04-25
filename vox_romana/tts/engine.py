"""The TTS engine interface — the universal socket for speech synthesis.

Every TTS engine is different under the hood. Piper uses ONNX models on CPU.
Kokoro uses MLX on Apple Silicon. OpenAI is a cloud API. ElevenLabs is another
cloud API with different auth, different voices, and different pricing.

But from the renderer's perspective, they all do the same thing:

    text + voice + emotion → audio bytes

That's what BaseTTSEngine defines. Any engine that implements this interface
can be plugged into the renderer, no questions asked.

THE ELECTRICAL OUTLET ANALOGY:

    Think of your house's electrical outlets. Your phone charger, your
    toaster, and your TV all have different purposes, different power
    needs, and different manufacturers. But they all use the same plug
    shape. The outlet doesn't care what's plugged into it.

    BaseTTSEngine is the outlet. PiperEngine, KokoroEngine, and
    OpenAITTSEngine are the appliances. The TTSRenderer is the house.
    It just sees outlets — it doesn't know or care about the wiring
    behind the wall.

    ┌──────────┐       ┌────────────────┐
    │ Renderer │──────▶│ BaseTTSEngine  │  ← the outlet (interface)
    └──────────┘       └───────┬────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
           ┌─────┴────┐ ┌─────┴────┐ ┌──────┴─────┐
           │  Piper    │ │  Kokoro  │ │  OpenAI    │
           │  (CPU)    │ │  (Metal) │ │  (cloud)   │
           └──────────┘ └──────────┘ └────────────┘

VOICE CONFIG:

    Each engine has its own voice naming convention:
        - Piper: "en_US-lessac-medium" (ONNX model name)
        - Kokoro: "af_heart", "am_adam" (built-in voice IDs)
        - OpenAI: "alloy", "echo", "nova" (API voice names)

    The VoiceConfig dataclass maps a SPEAKER NAME (from the debate) to an
    ENGINE-SPECIFIC VOICE ID. The renderer builds this mapping and passes
    it to the engine.

SYNTHESIZED AUDIO:

    Every engine returns SynthesizedAudio — a simple container holding
    the raw audio bytes and the sample rate. We use WAV format internally
    because it's uncompressed and pydub can slurp it up without any
    codec issues. The final output file can be any format (mp3, flac, etc.)
    — that's the mixer's job.

EMOTION HINTS:

    DebateEvents carry rich metadata: tone ("aggressive", "contemplative"),
    pace ("fast", "slow"), volume ("loud", "whisper"). These get passed to
    the engine as `emotion` and `**kwargs`.

    What the engine DOES with these hints is engine-specific:
        - Piper: mostly ignores them (no emotion control). We can nudge
          speed via length_scale, but that's about it.
        - Kokoro: has an `instruct` param for natural-language emotion
          control ("speak with excitement and urgency").
        - OpenAI TTS: has an `instructions` param (similar to Kokoro).
        - ElevenLabs: has stability/similarity_boost/style sliders.

    The engine does its best. If it can't handle a hint, it ignores it.
    No crashes, no errors — just neutral delivery.

ADDING A NEW ENGINE:

    1. Create a new file in vox_romana/tts/engines/ (e.g. elevenlabs.py)
    2. Subclass BaseTTSEngine
    3. Implement synthesize() and available_voices()
    4. That's it. Pass your engine to TTSRenderer and it just works.

EXAMPLE:
    from vox_romana.tts.engine import BaseTTSEngine, SynthesizedAudio

    class MyCustomEngine(BaseTTSEngine):
        def synthesize(self, text, *, voice="default", emotion="", **kwargs):
            audio_bytes = my_tts_library.speak(text, voice=voice)
            return SynthesizedAudio(audio=audio_bytes, sample_rate=22050)

        def available_voices(self):
            return ["default", "narrator", "villain"]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SynthesizedAudio:
    """A chunk of synthesized audio from a TTS engine.

    This is the universal "audio clip" object. Every engine returns these,
    regardless of how they generated the audio internally. Think of it as
    a labeled envelope: the audio bytes are the contents, and the sample
    rate is the stamp telling the mixer how to play them.

    Attributes:
        audio:       Raw audio bytes (WAV format, uncompressed PCM).
                     WAV is our internal lingua franca — every engine
                     produces it, every mixer consumes it.
        sample_rate: Samples per second (e.g. 22050 for Piper, 24000 for
                     Kokoro). The mixer needs this to correctly interpret
                     the raw bytes and resample if needed.
        format:      Audio format hint. Always "wav" for now. Exists so
                     engines could theoretically return other formats
                     (mp3, flac) if the mixer ever supports direct ingest.
    """

    audio: bytes
    sample_rate: int
    format: str = "wav"


@dataclass
class VoiceConfig:
    """Maps a debate speaker to an engine-specific voice.

    When the renderer encounters "Dr. Maya Chen says...", it looks up her
    VoiceConfig to know which engine voice to use and what speech parameters
    to apply.

    Think of it like a casting sheet:
        "Dr. Maya Chen → voice 'af_heart', speed 1.0, pitch default"

    Attributes:
        voice_id:  The engine-specific voice identifier.
                   - Piper: "en_US-lessac-medium"
                   - Kokoro: "af_heart"
                   - OpenAI: "nova"
        speed:     Speech speed multiplier. 1.0 = normal, 0.8 = slower,
                   1.2 = faster. Not all engines support this.
        pitch:     Pitch adjustment (engine-specific). Some engines ignore
                   this entirely. That's fine.
        extras:    A catch-all dict for engine-specific parameters that
                   don't fit the standard fields. For example:
                   - Kokoro: {"instruct": "speak warmly"}
                   - ElevenLabs: {"stability": 0.7, "style": 0.5}
                   The engine knows what to do with its own extras.
    """

    voice_id: str
    speed: float = 1.0
    pitch: float | None = None
    extras: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# THE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

class BaseTTSEngine(ABC):
    """The universal interface for text-to-speech engines.

    Subclasses must implement two methods:
        - synthesize(): text → audio bytes
        - available_voices(): what voices this engine offers

    Everything else (model loading, caching, GPU setup, API auth) is the
    engine's own business. The renderer never sees it.
    """

    @abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        voice: str = "default",
        emotion: str = "",
        speed: float = 1.0,
        language: str = "en",
        **kwargs,
    ) -> SynthesizedAudio:
        """Convert text to speech audio.

        This is the core method. Every engine must implement it.

        Args:
            text:    The text to speak. Can be a single sentence or a
                     full paragraph. The engine handles chunking internally
                     if the text is too long.
            voice:   Engine-specific voice ID (e.g. "af_heart", "nova",
                     "en_US-lessac-medium"). Use available_voices() to
                     see what's available.
            emotion: A natural-language emotion hint from the debate event
                     (e.g. "aggressive", "contemplative", "excited").
                     Engines that support emotion control will use this.
                     Engines that don't will ignore it gracefully.
            language: BCP-47 language code (e.g. "en", "es", "fr") for the
                     text. Engines validate that the requested voice can
                     speak this language and raise ValueError on mismatch.
                     Kokoro is English-only; Piper supports any language
                     for which an ONNX model is installed.
            speed:   Speech speed multiplier. 1.0 = normal.
            **kwargs: Engine-specific extras (e.g. pitch, stability).

        Returns:
            SynthesizedAudio with raw WAV bytes and sample rate.
        """

    @abstractmethod
    def available_voices(self) -> list[str]:
        """List the voices this engine can produce.

        The renderer calls this to auto-assign voices to speakers.
        Each engine returns its own voice IDs:
            - Piper: ["en_US-lessac-medium", "en_US-amy-low", ...]
            - Kokoro: ["af_heart", "am_adam", "bf_emma", ...]
            - OpenAI: ["alloy", "echo", "fable", "nova", ...]

        Returns:
            List of voice ID strings.
        """

    @property
    def name(self) -> str:
        """Human-readable name of this engine.

        Override this if you want a custom name. Defaults to the class
        name with common suffixes stripped:
            PiperEngine  → "Piper"
            KokoroEngine → "Kokoro"
            OpenAITTS    → "OpenAI"
        """
        cls_name = type(self).__name__
        for suffix in ("Engine", "TTS"):
            if cls_name.endswith(suffix):
                return cls_name[: -len(suffix)]
        return cls_name
