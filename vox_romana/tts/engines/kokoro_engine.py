"""Kokoro TTS Engine — high-quality local speech synthesis via MLX on Apple Silicon.

If Piper is the Honda Civic, Kokoro is the Tesla. It sounds significantly
more natural, has built-in emotion control (the `instruct` parameter), and
runs on your M1 Max GPU via Apple's MLX framework. It's still free, still
local, and still fast — but the audio quality is in a different league.

HOW KOKORO WORKS:

    Kokoro uses a neural speech synthesis model that runs natively on
    Apple Silicon via the MLX framework (Apple's answer to PyTorch/JAX
    for their own chips). The model understands not just WHAT to say,
    but HOW to say it — you can tell it "speak angrily" or "whisper
    nervously" and it'll adjust its delivery.

    The model is auto-downloaded from HuggingFace on first use (~2GB).
    After that, it's cached locally and loads in seconds.

EMOTION SUPPORT (the killer feature):

    Unlike Piper, Kokoro understands natural-language emotion instructions
    via the `instruct` parameter. The DebateEvent metadata maps beautifully:

        Event: tone=["aggressive"], pace="fast", volume="loud"
        → instruct: "Speak aggressively with a fast pace and loud volume"

        Event: tone=["contemplative", "measured"], pace="slow"
        → instruct: "Speak in a contemplative, measured way at a slow pace"

    This is why Kokoro is the recommended engine for debates — it can
    actually PERFORM the emotional shifts that make a debate sound alive.

AVAILABLE VOICES:

    Kokoro ships with several built-in voices. The naming convention:
        - First letter:  a=American, b=British
        - Second letter: f=female, m=male
        - The rest: a name or identifier

    English voices (as of v0.1):
        af_heart, af_alloy, af_aoede, af_jessica, af_kore, af_nicole,
        af_nova, af_river, af_sarah, af_sky
        am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx
        bf_emma, bf_isabella, bf_lily
        bm_daniel, bm_fable, bm_george, bm_lewis

    TIP: For a debate, try contrasting voices:
         Dr. Maya Chen → af_heart (warm, clear female)
         Dr. Robert Vance → am_adam (deep, authoritative male)
         Moderator → bf_emma (British neutral, implies gravitas)

MODEL:
    Default model: "mlx-community/Kokoro-82M-bf16"
    Size: ~355 MB (downloaded once, cached in HuggingFace cache dir)
    Speed: ~2-3x realtime on M1 Max (first run is slower due to
           compilation; subsequent runs are faster)

SETUP:
    pip install mlx mlx-audio misaki[en] num2words phonemizer spacy espeakng_loader
    # Model auto-downloads on first use — no manual download needed.
    # First run also downloads en_core_web_sm spacy model (~13MB).

EXAMPLE:
    from vox_romana.tts.engines.kokoro_engine import KokoroEngine

    engine = KokoroEngine()  # Uses default model
    result = engine.synthesize(
        "Universal basic income is the future!",
        voice="af_heart",
        emotion="speak passionately and with conviction",
    )
    # result.audio is WAV bytes, result.sample_rate is 24000

APPLE SILICON ONLY:
    This engine requires an Apple Silicon Mac (M1/M2/M3/M4) with the
    MLX framework. It will NOT work on Intel Macs or Linux/Windows.
    For those platforms, use PiperEngine or OpenAI TTS.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

from pydub import AudioSegment

from ..engine import BaseTTSEngine, SynthesizedAudio


def _lazy_generate_audio(**kwargs):
    """Lazy wrapper that imports generate_audio on first call.

    We don't import mlx_audio at module level because:
    1. It would crash on non-Apple-Silicon systems at import time.
    2. It pulls in the full MLX framework, which is heavy.
    3. We want tests to be able to run without the real library.

    On first call, we import the real function and cache it.
    """
    from mlx_audio.tts.generate import generate_audio
    # Replace ourselves with the real function for future calls.
    globals()["_lazy_generate_audio"] = generate_audio
    return generate_audio(**kwargs)


# Saved reference to the original stub so teardown() can reset the global.
_LAZY_GENERATE_AUDIO_STUB = _lazy_generate_audio


# Default Kokoro model on HuggingFace. Auto-downloaded on first use.
# NOTE: The original prince-canuma repo went private/offline. The
# mlx-community fork is the canonical source (834K+ downloads).
DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"

# Kokoro's native sample rate.
KOKORO_SAMPLE_RATE = 24000

# Language codes accepted by mlx-audio's Kokoro wrapper.
#
# The values are mlx-audio pipeline codes passed via ``lang_code`` to
# ``generate_audio()``:
#   a=en, b=en-gb, z=zh, e=es, f=fr, h=hi, i=it, j=ja, p=pt / pt-br
KOKORO_LANG_CODES: dict[str, str] = {
    "en": "a",
    "en-gb": "b",
    "zh": "z",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt": "p",
    "pt-br": "p",
}
KOKORO_LANGUAGES: frozenset[str] = frozenset(KOKORO_LANG_CODES)


def _resolve_kokoro_lang_code(language: str) -> str | None:
    """Map a BCP-47-ish language string to mlx-audio's ``lang_code``.

    Exact mappings win first (e.g. ``en-gb`` → ``b``, ``pt-br`` → ``p``).
    For other region variants (e.g. ``zh-CN``), we fall back to the primary
    language subtag when Kokoro supports that base language.
    """
    normalized = language.strip().lower().replace("_", "-")
    direct = KOKORO_LANG_CODES.get(normalized)
    if direct is not None:
        return direct
    primary = normalized.split("-", 1)[0]
    return KOKORO_LANG_CODES.get(primary)

# All built-in Kokoro voices, organized by accent and gender.
# These are baked into the model — no separate downloads needed.
KOKORO_VOICES = [
    # ── American Female ──────────────────────────────────────────
    "af_heart", "af_alloy", "af_aoede", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    # ── American Male ────────────────────────────────────────────
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx",
    # ── British Female ───────────────────────────────────────────
    "bf_emma", "bf_isabella", "bf_lily",
    # ── British Male ─────────────────────────────────────────────
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]


class KokoroEngine(BaseTTSEngine):
    """Kokoro TTS engine — high-quality, local, Apple Silicon GPU.

    Uses mlx-audio to run the Kokoro model on your M-series chip.
    Auto-downloads the model on first use. Supports emotion/style
    control via the `instruct` parameter.

    Attributes:
        _model_name:  HuggingFace model identifier.
        _model:       The loaded model (lazy-loaded on first synthesize).
        _sample_rate: Audio sample rate (24000 Hz for Kokoro).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """Initialize the Kokoro engine.

        Args:
            model: HuggingFace model identifier or local path.
                   Default: "mlx-community/Kokoro-82M-bf16"
                   The model auto-downloads on first use (~355 MB).

        TIP: The first call to synthesize() will take a minute while
             the model downloads. After that, it's cached and loads
             in ~2-5 seconds.
        """
        self._model_name = model
        self._model = None  # Lazy-loaded on first use.
        self._sample_rate = KOKORO_SAMPLE_RATE

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "af_heart",
        emotion: str = "",
        speed: float = 1.0,
        language: str = "en",
        **kwargs,
    ) -> SynthesizedAudio:
        """Synthesize text to speech using Kokoro via MLX.

        Args:
            text:     The text to speak.
            voice:    Kokoro voice ID (e.g. "af_heart", "am_adam").
                      See KOKORO_VOICES for the full list.
            emotion:  Natural-language emotion instruction for the model.
                      Examples: "speak angrily", "whisper nervously",
                      "say this with excitement and conviction".
                      This maps to Kokoro's `instruct` parameter.
            speed:    Speed multiplier. 1.0 = normal.
            language: BCP-47 language code. Kokoro supports a small set of
                      languages exposed by mlx-audio via ``lang_code``.
            **kwargs: Extra kwargs passed to generate_audio().

        Returns:
            SynthesizedAudio with WAV bytes and sample rate (24000 Hz).

        Raises:
            ValueError: If ``language`` is not one of Kokoro's supported
                        languages.

        NOTE ON HOW THIS WORKS:
            mlx-audio's generate_audio() is designed to save files to disk.
            We work around this by writing to a temp directory, then reading
            the WAV back into memory. Not elegant, but reliable. The temp
            files are cleaned up automatically.

            Why not call the model directly? Because generate_audio() handles
            text chunking, model loading, and a bunch of edge cases we don't
            want to reimplement. The file roundtrip costs <10ms on an SSD.
        """
        lang_code = _resolve_kokoro_lang_code(language)
        if lang_code is None:
            supported = ", ".join(sorted(KOKORO_LANGUAGES))
            raise ValueError(
                f"KokoroEngine does not support language={language!r}. "
                f"Supported language codes: {supported}."
            )

        # Create a temp directory for the audio output.
        # generate_audio() writes files here; we read them back.
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Build kwargs for generate_audio().
            gen_kwargs = dict(
                text=text,
                model=self._model_name,
                voice=voice,
                speed=speed,
                audio_format="wav",
                output_path=tmp_dir,
                file_prefix="clip",
                save=True,
                verbose=False,    # Don't spam the console.
                play=False,       # Don't play audio out loud.
                join_audio=True,  # Join segments into one file.
                stream=False,     # We want the full audio at once.
            )

            # Pass emotion as the `instruct` parameter.
            # This is Kokoro's natural-language style control.
            # If emotion is empty, don't pass instruct (model uses default style).
            if emotion:
                gen_kwargs["instruct"] = emotion

            # Merge any extra kwargs (e.g. temperature, max_tokens).
            gen_kwargs.update(kwargs)
            gen_kwargs["lang_code"] = lang_code

            # Generate! This is where the GPU does its thing.
            # Suppress the bare print() fired by mlx_audio's
            # KokoroModel._get_pipeline() on every language-pipeline
            # creation — it's not a logger and cannot be silenced via
            # verbose=False.
            with contextlib.redirect_stdout(io.StringIO()):
                _lazy_generate_audio(**gen_kwargs)

            # Read back the generated WAV file.
            # generate_audio with join_audio=True writes: {prefix}.wav
            wav_path = Path(tmp_dir) / "clip.wav"

            if not wav_path.exists():
                # Fallback: check for numbered segments (clip_000.wav, etc.)
                # join_audio=True should produce a single clip.wav, but some
                # Kokoro versions write clip_000.wav, clip_001.wav, etc. for
                # long texts.  Concatenate ALL segments to avoid truncation.
                wav_files = sorted(Path(tmp_dir).glob("clip*.wav"))
                if not wav_files:
                    raise RuntimeError(
                        f"Kokoro generated no audio files in {tmp_dir}. "
                        f"Text was: {text[:100]!r}"
                    )
                combined = AudioSegment.from_wav(str(wav_files[0]))
                for wf in wav_files[1:]:
                    combined += AudioSegment.from_wav(str(wf))
                buf = io.BytesIO()
                combined.export(buf, format="wav")
                wav_bytes = buf.getvalue()
            else:
                wav_bytes = wav_path.read_bytes()

        return SynthesizedAudio(
            audio=wav_bytes,
            sample_rate=self._sample_rate,
        )

    def available_voices(self) -> list[str]:
        """List all built-in Kokoro voices.

        These are baked into the model — no downloads needed.
        The list is hardcoded here because the model doesn't expose
        a programmatic way to query voices.

        Returns:
            List of voice ID strings (e.g. ["af_heart", "am_adam", ...]).
        """
        return list(KOKORO_VOICES)

    def teardown(self) -> None:
        """Release Kokoro model state to free RAM.

        Resets the module-level generate_audio reference back to the lazy
        stub so that the real mlx_audio function (which keeps model weights
        resident in MLX unified memory) can be garbage-collected.  Also
        attempts to clear the MLX metal heap cache.
        """
        import vox_romana.tts.engines.kokoro_engine as _m
        _m._lazy_generate_audio = _m._LAZY_GENERATE_AUDIO_STUB
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass
