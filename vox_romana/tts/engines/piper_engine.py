"""Piper TTS Engine — local, free, CPU-only speech synthesis.

Piper is the Honda Civic of TTS engines. It's not flashy, it won't win
any awards for vocal range, and nobody's going to mistake it for a real
human. But it starts every time, it's dead simple, it runs on anything,
and it costs nothing. Perfect for testing the pipeline end-to-end before
investing in fancier engines.

HOW PIPER WORKS:

    Piper uses ONNX models (pre-trained neural networks) to convert text
    to speech. Each "voice" is a separate ONNX model file (~15-100 MB).
    The models run on CPU via ONNX Runtime — no GPU needed.

    The text goes through espeak-ng (a phonemizer) to convert words into
    phonemes, then the neural network turns those phonemes into audio
    waveforms. It's fast: a typical debate turn takes 0.1-0.5 seconds.

VOICE MODELS:

    Piper voices are downloaded as .onnx files (+ .json config).
    Each voice has a quality tier:
        - "low"    (~15MB): Fastest, lowest quality. Fine for testing.
        - "medium" (~60MB): Good balance. Recommended for production.
        - "high"   (~100MB): Best Piper quality. Still not Kokoro-level.

    Popular English voices:
        - en_US-lessac-medium    (clear American male, good default)
        - en_US-amy-medium       (American female)
        - en_US-ryan-medium      (American male, deeper)
        - en_GB-alba-medium      (British female)
        - en_GB-northern_english_male-medium  (Northern English male)

    Download voices from:
        https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0

    Or use the download helper:
        python -m piper --download-dir ./models --model en_US-lessac-medium

EMOTION SUPPORT:

    Piper has NO emotion control. The `emotion` parameter is accepted
    but ignored. The voice speaks the same way regardless of whether the
    debate turn is "aggressive" or "contemplative."

    What we CAN do:
        - Speed: adjust via length_scale (higher = slower, lower = faster)
        - Volume: adjust via SynthesisConfig.volume
    That's it. For emotional range, use Kokoro or OpenAI TTS.

MULTI-SPEAKER:

    Some Piper models support multiple speakers via speaker_id. The
    en_US-lessac model, for example, only has one speaker (id=0). But
    models like libritts have many speakers.

    For a debate, you typically want one model per speaker (different
    voice characteristics), not multiple speakers from one model.

SETUP:
    pip install piper-tts       # Already installed
    # Download a voice model:
    mkdir -p models/piper
    # Download en_US-lessac-medium.onnx + .json from HuggingFace

EXAMPLE:
    from vox_romana.tts.engines.piper import PiperEngine

    engine = PiperEngine(model_dir="piper-voices")
    result = engine.synthesize("Hello, world!", voice="en_US-lessac-medium")
    # result.audio is WAV bytes, result.sample_rate is 22050
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

from ..engine import BaseTTSEngine, SynthesizedAudio


class PiperEngine(BaseTTSEngine):
    """Piper TTS engine — local, free, CPU-only.

    Uses piper-tts (ONNX-based) for speech synthesis. Each voice is a
    separate ONNX model file. Multiple models can be loaded for
    multi-speaker debates.

    Attributes:
        _model_dir:      Directory containing .onnx voice model files.
        _voices:         Dict of voice_id → loaded PiperVoice objects.
        _default_voice:  The voice_id to use when none is specified.

    TIP: Piper models are lazy-loaded. The first call to synthesize()
         with a new voice will take a moment to load the ONNX model.
         Subsequent calls reuse the loaded model (fast).
    """

    def __init__(
        self,
        model_dir: str | Path = "piper-voices",
    ) -> None:
        """Initialize the Piper engine.

        Args:
            model_dir: Directory containing .onnx voice model files.
                       Each voice is a pair of files:
                           {voice_name}.onnx     (the model)
                           {voice_name}.onnx.json (the config)
                       You can have multiple voices in one directory.

        TIP: If you only have one voice model, that's fine. The renderer
             will use it for all speakers. The debate will sound like one
             person doing all the parts — not ideal, but functional for
             testing.
        """
        self._model_dir = Path(model_dir)
        self._voices: dict[str, object] = {}  # voice_id → PiperVoice
        self._default_voice: str | None = None

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "default",
        emotion: str = "",
        speed: float = 1.0,
        **kwargs,
    ) -> SynthesizedAudio:
        """Synthesize text to speech using Piper.

        Args:
            text:     The text to speak.
            voice:    Voice model name (e.g. "en_US-lessac-medium").
                      If "default", uses the first available voice.
            emotion:  Ignored by Piper (no emotion control). Accepted
                      to match the interface — the renderer doesn't need
                      to know that Piper can't do emotions.
            speed:    Speed multiplier. 1.0 = normal, <1 = faster, >1 = slower.
                      NOTE: Piper's length_scale is inverted from what you'd
                      expect. We handle the conversion internally.
            **kwargs: Passed to PiperVoice. Supports:
                      - speaker_id (int): For multi-speaker models.

        Returns:
            SynthesizedAudio with WAV bytes and sample rate.
        """
        # Lazy-load the voice model.
        piper_voice = self._load_voice(voice)

        # Build synthesis config.
        # Piper's length_scale: higher = slower. We invert speed so that
        # speed=1.2 (faster) becomes length_scale=0.83 (shorter phonemes).
        from piper.config import SynthesisConfig

        syn_config = SynthesisConfig()
        if speed != 1.0:
            syn_config.length_scale = 1.0 / speed

        # Handle speaker_id for multi-speaker models.
        speaker_id = kwargs.get("speaker_id")
        if speaker_id is not None:
            syn_config.speaker_id = speaker_id

        # Synthesize to a WAV buffer.
        # PiperVoice.synthesize_wav() writes to a wave.Wave_write object.
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            piper_voice.synthesize_wav(text, wav_file, syn_config=syn_config)

        wav_bytes = wav_buffer.getvalue()

        # Get the sample rate from the voice config.
        sample_rate = piper_voice.config.sample_rate

        return SynthesizedAudio(
            audio=wav_bytes,
            sample_rate=sample_rate,
        )

    def available_voices(self) -> list[str]:
        """List voice models found in the model directory (recursive).

        Recursively scans model_dir for .onnx files and returns their
        stems as voice IDs. Works with both flat directories and the
        nested structure of the piper-voices git clone:
            piper-voices/en/en_US/lessac/medium/en_US-lessac-medium.onnx
            → voice ID: "en_US-lessac-medium"

        Returns:
            List of voice ID strings (sorted alphabetically).
        """
        if not self._model_dir.exists():
            return []

        # Recursively find all .onnx files (voice models).
        # Each voice is a .onnx file; the config is .onnx.json — filter those out.
        voices = sorted(
            p.stem for p in self._model_dir.rglob("*.onnx")
            if not p.name.endswith(".onnx.json")
        )
        return voices

    def _load_voice(self, voice_id: str) -> object:
        """Lazy-load a Piper voice model.

        Models are loaded once and cached. The first call to synthesize()
        with a new voice takes ~1 second. Subsequent calls are instant.

        Args:
            voice_id: Voice name (e.g. "en_US-lessac-medium") or "default".

        Returns:
            A loaded PiperVoice object.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
        """
        # Resolve "default" to the first available voice.
        if voice_id == "default":
            voices = self.available_voices()
            if not voices:
                raise FileNotFoundError(
                    f"No .onnx voice models found in {self._model_dir}. "
                    f"Download one from https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0"
                )
            voice_id = voices[0]

        # Return cached voice if already loaded.
        if voice_id in self._voices:
            return self._voices[voice_id]

        # Load the model — search recursively to support nested piper-voices/ layout.
        # First try a flat path (fast), then fall back to recursive search.
        model_path = self._model_dir / f"{voice_id}.onnx"
        if not model_path.exists():
            found = list(self._model_dir.rglob(f"{voice_id}.onnx"))
            if not found:
                raise FileNotFoundError(
                    f"Piper voice model not found: {voice_id}.onnx\n"
                    f"Searched in: {self._model_dir}\n"
                    f"Available models: {self.available_voices()[:10]}..."
                )
            model_path = found[0]

        from piper import PiperVoice

        loaded = PiperVoice.load(str(model_path))
        self._voices[voice_id] = loaded

        # Track the first loaded voice as default.
        if self._default_voice is None:
            self._default_voice = voice_id

        return loaded
