"""MusicGen Engine — AI-powered music generation for show bumpers and theme cues.

This engine turns text prompts into short music clips using Meta's MusicGen
model, accessed through HuggingFace Transformers. No audiocraft package is
required — the model is available directly via the `transformers` library
already present in this project.

THE BUMPER ANALOGY:

    Every radio show has a sound board. The producer hits a button and a
    five-second "stinger" plays before the next segment. Another button
    and the theme music swells. This engine IS that sound board — except
    instead of pre-recorded stingers, you describe what you want and the
    model generates it on the fly.

    "upbeat podcast intro with jazz piano"
    "somber orchestral transition, strings only"
    "news alert sting, punchy, 2 seconds"
    "lo-fi hip hop background, podcast vibe"

WHERE IT FITS IN THE PIPELINE:

    The renderer intercepts events with event_kind == "bumper" (or events
    with [music: ...] cues in the text) and routes them here instead of
    to the speech TTS engine. The output is a SynthesizedAudio — the same
    container the speech engines use — so the mixer handles it identically.

    ┌────────────────────────────────────────────────────────────────────┐
    │  Script events                                                     │
    │      utterance  ──▶  KokoroEngine / PiperEngine  ──▶  AudioMixer  │
    │      bumper     ──▶  MusicGenEngine              ──▶  AudioMixer  │
    └────────────────────────────────────────────────────────────────────┘

LAZY LOADING:

    Like KokoroEngine, the model is NOT loaded at import time. It loads
    on the first call to generate(). This means:
    - Tests that mock this class don't pull in PyTorch or the model.
    - The ~300 MB download only happens when music is actually needed.
    - Import-time is instant even in environments without GPU support.

MODEL OPTIONS:

    facebook/musicgen-small   (~300 MB, fast, good quality, default)
    facebook/musicgen-medium  (~1.5 GB, slower, better quality)
    facebook/musicgen-large   (~3.3 GB, slowest, best quality)

    On an M1 Max with MPS, small generates an 8-second clip in ~10 seconds.
    medium is ~30s. All three run on Apple Silicon via MPS.

DEVICE SELECTION:

    The engine auto-selects the best available device:
    1. MPS  — Apple Silicon GPU (fastest on Mac)
    2. CUDA — NVIDIA GPU (fastest on Linux/Windows)
    3. CPU  — universal fallback (slow but always works)

    Pass device="cpu" explicitly to force CPU (useful for testing or if
    MPS gives unexpected results on a particular model version).

EXAMPLE:
    from vox_romana.tts.music_engine import MusicGenEngine

    engine = MusicGenEngine()  # loads model on first generate() call
    result = engine.generate(
        "upbeat jazz podcast intro, piano and bass",
        duration=8.0,
    )
    # result.audio is a valid WAV file (bytes)
    # result.sample_rate is 32000 (MusicGen's native rate)

    with open("intro.wav", "wb") as f:
        f.write(result.audio)

REQUIREMENTS:
    - transformers>=4.31  (already in requirements.txt)
    - torch               (already in requirements.txt)
    - scipy               (already in requirements.txt, used for WAV export)
    No new pip installs needed.
"""

from __future__ import annotations

import io
import logging

import numpy as np

from .engine import SynthesizedAudio

try:
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
except ImportError:  # pragma: no cover
    AutoProcessor = None  # type: ignore[assignment]
    MusicgenForConditionalGeneration = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Default model — small is fast and good enough for bumpers/stingers.
# Users can pass model_name="facebook/musicgen-medium" for higher quality.
DEFAULT_MODEL = "facebook/musicgen-small"

# MusicGen's native audio sample rate (Hz).
# This is fixed by the model architecture — not configurable.
MUSICGEN_SAMPLE_RATE = 32000

# Convenience shorthands accepted by the CLI (--music-model small).
# Maps to the full HuggingFace model identifier.
MODEL_ALIASES: dict[str, str] = {
    "small":  "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large":  "facebook/musicgen-large",
}


def _resolve_model_name(name: str) -> str:
    """Resolve a model alias or full HF identifier to the canonical name.

    Args:
        name: Either a shorthand ("small", "medium", "large") or a full
              HuggingFace model identifier like "facebook/musicgen-small".

    Returns:
        The canonical HuggingFace model identifier.
    """
    return MODEL_ALIASES.get(name, name)


def _select_device(device: str | None) -> str:
    """Auto-select the best available compute device.

    Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU.

    Args:
        device: Explicit device string, or None to auto-select.

    Returns:
        A torch device string: "mps", "cuda", or "cpu".
    """
    if device is not None:
        return device

    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover
        pass

    return "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class MusicGenEngine:
    """Generates short music clips from text prompts using MusicGen.

    This is NOT a BaseTTSEngine subclass — music generation has a different
    interface (no voice, no speed, no emotion hint — just a prompt and a
    duration). The renderer handles it separately from speech synthesis.

    Attributes:
        _model_name:  Canonical HuggingFace model identifier.
        _device:      Compute device string ("mps", "cuda", "cpu").
        _model:       The loaded MusicGen model (None until first generate()).
        _processor:   The loaded MusicGen processor (None until first generate()).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the MusicGen engine.

        The model is NOT loaded here — it loads lazily on the first call
        to generate(). This mirrors the pattern used by KokoroEngine and
        keeps import time instant.

        Args:
            model_name: HuggingFace model identifier or shorthand alias
                        ("small", "medium", "large"). Defaults to
                        "facebook/musicgen-small" (~300 MB on first use).
            device:     Compute device. None = auto-select (MPS > CUDA > CPU).
                        Pass "cpu" to force CPU, e.g. for reproducible tests.

        TIP: On an M1 Max, musicgen-small generates 8 seconds of audio
             in roughly 10–15 seconds on MPS. The first run takes longer
             because the model is downloaded (~300 MB) and compiled.
        """
        self._model_name = _resolve_model_name(model_name)
        self._device = _select_device(device)
        self._cache_dir = cache_dir
        self._model = None       # Lazy-loaded on first generate() call.
        self._processor = None   # Lazy-loaded on first generate() call.

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load the model and processor if not already loaded.

        Called at the start of generate(). After the first call, this is
        a no-op. The model stays in memory for the lifetime of this object,
        so subsequent generate() calls are fast.

        DOWNLOAD STRATEGY:
            Large models (musicgen-large is ~13.7 GB) can stall indefinitely
            when using from_pretrained() directly because requests' streaming
            mode has no read-stall timeout — if the server stops sending bytes
            the download freezes forever.

            We pre-download via huggingface_hub.snapshot_download(), which:
            1. Uses .incomplete lock files so interrupted downloads resume
               from where they left off (Ctrl-C and re-run is safe).
            2. Respects HF_HUB_ENABLE_HF_TRANSFER=1, activating the optional
               hf_transfer backend (Rust-based multi-part download with
               built-in stall detection and retries).
            3. Sets HF_HUB_HTTP_TIMEOUT to 120 s so a connection that stops
               sending data is detected and retried rather than hanging.

            Once the files are local we load from that path, which is
            instantaneous (no network I/O).
        """
        if self._model is not None:
            return

        import os

        # Raise HTTP read timeout so a stalled connection triggers a retry
        # instead of hanging indefinitely.  Only set if the caller hasn't
        # already configured it.
        os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "120")

        # Enable hf_transfer if the package is installed.  It uses a
        # Rust-based multi-part downloader that handles stalls far better
        # than plain requests.  Silently ignored when not installed.
        try:
            import hf_transfer  # noqa: F401
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
            logger.debug("MusicGenEngine: hf_transfer available — using fast download")
        except ImportError:
            pass  # fine — fall back to requests-based download

        logger.info(
            "MusicGenEngine: downloading/verifying %s (first use — large models "
            "can take a while; download is resumable if interrupted)",
            self._model_name,
        )

        from huggingface_hub import snapshot_download

        local_path = snapshot_download(
            repo_id=self._model_name,
            cache_dir=self._cache_dir,
        )

        logger.info("MusicGenEngine: loading model on %s from %s", self._device, local_path)

        self._processor = AutoProcessor.from_pretrained(
            local_path, local_files_only=True
        )
        self._model = MusicgenForConditionalGeneration.from_pretrained(
            local_path, local_files_only=True
        )
        self._model.to(self._device)  # type: ignore[union-attr]

        logger.info("MusicGenEngine: model loaded — ready to generate")

    # ── Public API ────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        duration: float = 8.0,
    ) -> SynthesizedAudio:
        """Generate a music clip from a text prompt.

        Args:
            prompt:   Natural-language description of the desired music.
                      Examples:
                          "upbeat jazz podcast intro, piano and bass"
                          "somber orchestral strings, transitional"
                          "news alert sting, punchy, 2 seconds"
                          "lo-fi hip hop, chill, background podcast vibe"
            duration: Length of the generated clip in seconds.
                      MusicGen supports up to ~30 seconds per call.
                      Default is 8 seconds (good for bumpers/stingers).

        Returns:
            SynthesizedAudio with WAV bytes at MUSICGEN_SAMPLE_RATE (32000 Hz).
            The audio field is a fully valid WAV file — drop it straight
            into AudioMixer.add_music_clip() or write to disk.

        NOTE ON CONVERSION:
            MusicGen outputs a float32 tensor. We convert to int16 PCM and
            write a WAV file via scipy.io.wavfile — the same scipy that
            powers DataResearcher and is already in requirements.txt.
            No torchaudio import needed.
        """
        self._ensure_loaded()

        logger.debug("MusicGenEngine: generating %.1fs clip — %r", duration, prompt)

        import torch
        from scipy.io import wavfile

        # Tokenize the prompt and move inputs to the compute device.
        inputs = self._processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Tell the model how many audio tokens to generate.
        # MusicGen encodes audio at 50 tokens/second internally.
        max_new_tokens = int(duration * 50)

        # Generate — disable gradient tracking (we're doing inference only).
        with torch.no_grad():
            audio_values = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        # audio_values shape: [batch, channels, samples]
        # We took one prompt → batch=1, channels=1 → extract [0, 0].
        audio_np = audio_values[0, 0].cpu().numpy()

        # Convert float32 → int16 PCM (standard WAV sample format).
        # Clip first to [-1, 1] to avoid wrap-around distortion.
        audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)

        # Write to an in-memory WAV file using scipy (already in requirements).
        buffer = io.BytesIO()
        wavfile.write(buffer, MUSICGEN_SAMPLE_RATE, audio_int16)
        wav_bytes = buffer.getvalue()

        logger.debug(
            "MusicGenEngine: generated %d bytes of audio (%.1fs @ %d Hz)",
            len(wav_bytes),
            duration,
            MUSICGEN_SAMPLE_RATE,
        )

        return SynthesizedAudio(audio=wav_bytes, sample_rate=MUSICGEN_SAMPLE_RATE)

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """The canonical HuggingFace model identifier in use."""
        return self._model_name

    @property
    def device(self) -> str:
        """The compute device this engine will use (mps/cuda/cpu)."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """True once the model has been loaded (i.e. after the first generate() call)."""
        return self._model is not None

    def teardown(self) -> None:
        """Release the MusicGen model and processor to free RAM.

        Drops the Python references to the HuggingFace model and processor
        objects, then runs gc.collect() and optionally empties the torch
        device cache so that the freed memory is available before the next
        pipeline phase (e.g. animation/diffusion).
        """
        import gc
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
