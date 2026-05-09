"""VoiceCloneEngine — synthesize speech using a cloned voice profile.

This engine reads a registered voice profile (created by
``vox_romana.voice_prep``) and uses **f5-tts-mlx** to clone that voice
when synthesising text.  The cloned voice can speak any language that
f5-tts supports — making it the primary bridge between a real speaker's
timbre and a multilingual show.

HOW IT WORKS:

    f5-tts (Flow-Matching Text-to-Speech) conditions generation on a
    short reference audio clip.  Given ~5-15 seconds of a target voice,
    it reconstructs that speaker's timbre for any new text.  The MLX
    port (f5-tts-mlx) runs natively on Apple Silicon via Metal, with no
    PyTorch dependency — so it coexists cleanly with the existing
    torch 2.11 installation.

VOICE PROFILES:

    A profile is the directory written by ``voice_prep.profiler``:

        {profiles_dir}/vc_{name}/
            reference.wav    — 24 kHz mono WAV reference clip
            metadata.json    — descriptor (gender, tone, tags, …)

    The engine is constructed with one *profiles_dir*; it loads profiles
    lazily and caches them by ``voice_id`` (e.g. ``vc_marcus_deep``).

AVAILABLE VOICES:

    ``available_voices()`` scans ``profiles_dir`` for any sub-directory
    whose name starts with the ``vc_`` prefix and contains a valid
    ``reference.wav``.  This means newly registered profiles are
    automatically picked up without restarting the engine.

EMOTION HINTS:

    f5-tts-mlx does not have an explicit emotion-control API.  The
    ``emotion`` parameter is silently accepted and ignored — consistent
    with PiperEngine's behaviour.

LANGUAGE:

    f5-tts-mlx can synthesize text in any language as long as the
    reference clip phoneme-matches reasonably well.  We do NOT validate
    ``language`` (unlike KokoroEngine) so the renderer can route
    non-English utterances here freely.

ADDING BACKENDS:

    The internal ``_synth_f5`` method is intentionally isolated.  To add
    an alternative backend (``"chatterbox"``, ``"elevenlabs"``), add a
    new ``_synth_*`` method and dispatch from ``synthesize()`` based on a
    ``backend`` constructor argument.

EXAMPLE:

    from vox_romana.voice_prep.profiler import register_voice
    from vox_romana.tts.engines.voice_clone_engine import VoiceCloneEngine

    engine = VoiceCloneEngine(profiles_dir="/data/voice_profiles")
    result = engine.synthesize(
        "Il senato ha votato questa mattina.",
        voice="vc_marcus_deep",
        language="it",
    )
    # result.audio is WAV bytes at VOICE_CLONE_SAMPLE_RATE

APPLE SILICON NOTE:

    f5-tts-mlx requires an Apple Silicon Mac.  On other platforms, the
    import will fail; the engine will still construct but ``synthesize()``
    will raise ``RuntimeError`` at call time, not at import time.

SETUP:

    pip install f5-tts-mlx>=0.2.6
    # No separate model download step — the model fetches automatically
    # from HuggingFace on first use (~800 MB, cached thereafter).
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

from ..engine import BaseTTSEngine, SynthesizedAudio
from ...voice_prep.profiler import VOICE_CLONE_PREFIX, list_profiles

logger = logging.getLogger(__name__)

# f5-tts-mlx generates 24 kHz audio.
VOICE_CLONE_SAMPLE_RATE = 24_000


class VoiceCloneEngine(BaseTTSEngine):
    """TTS engine that synthesizes speech conditioned on a cloned voice profile.

    Attributes:
        _profiles_dir: Root directory containing ``vc_*`` profile subdirs.
        _ref_cache:    Lazy cache mapping voice_id → absolute Path of reference.wav.
    """

    def __init__(self, profiles_dir: str | Path) -> None:
        """Construct the engine.

        Args:
            profiles_dir:
                Root directory written by ``voice_prep.profiler``.  Must
                exist and contain at least one ``vc_*`` profile for
                ``synthesize()`` to succeed.  The directory is re-scanned
                on each ``available_voices()`` call so new profiles are
                picked up without restarting.
        """
        self._profiles_dir = Path(profiles_dir).resolve()
        self._ref_cache: dict[str, Path] = {}

    # ── BaseTTSEngine interface ──────────────────────────────────────

    @property
    def name(self) -> str:
        return "voice_clone"

    def available_voices(self) -> list[str]:
        """Scan profiles_dir and return voice_ids for all valid profiles.

        A profile is valid when its directory starts with ``vc_`` **and**
        contains a readable ``reference.wav``.  Malformed directories
        (missing WAV, no metadata) are silently skipped.

        This method updates the internal reference cache so subsequent
        ``synthesize()`` calls find the correct WAV without re-scanning.
        """
        if not self._profiles_dir.exists():
            return []

        voices: list[str] = []
        for entry in sorted(self._profiles_dir.iterdir()):
            if not entry.is_dir():
                continue
            if not entry.name.startswith(VOICE_CLONE_PREFIX):
                continue
            ref = entry / "reference.wav"
            if not ref.exists():
                logger.debug(
                    "VoiceCloneEngine: skipping profile %s (no reference.wav)",
                    entry.name,
                )
                continue
            voice_id = entry.name  # already has vc_ prefix
            self._ref_cache[voice_id] = ref
            voices.append(voice_id)

        return voices

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "",
        emotion: str = "",
        speed: float = 1.0,
        language: str = "en",
        **kwargs,
    ) -> SynthesizedAudio:
        """Synthesize text using the cloned voice identified by ``voice``.

        Args:
            text:     The text to speak.  May be any language.
            voice:    A ``vc_`` prefixed voice_id previously registered
                      via ``voice_prep.profiler.register_voice()``.
            emotion:  Accepted for interface compatibility; ignored by
                      f5-tts-mlx (no explicit emotion API).
            speed:    Speed multiplier.  f5-tts-mlx does not expose a
                      real-time speed knob so values other than 1.0 are
                      accepted but currently ignored.
            language: BCP-47 language code.  f5-tts-mlx supports any
                      language; this value is passed through for
                      informational purposes and future use.
            **kwargs: Forwarded to the f5-tts backend (e.g. ``nfe_step``,
                      ``cfg_strength`` for f5-tts-mlx).

        Returns:
            SynthesizedAudio with raw WAV bytes and 24 000 Hz sample rate.

        Raises:
            ValueError:  If ``voice`` is not a known ``vc_`` voice_id.
            RuntimeError: If the f5-tts backend is not installed or fails.
        """
        if not voice:
            raise ValueError(
                "VoiceCloneEngine.synthesize(): voice must be a registered "
                "vc_* voice_id (e.g. 'vc_marcus_deep')."
            )
        if not voice.startswith(VOICE_CLONE_PREFIX):
            raise ValueError(
                f"VoiceCloneEngine.synthesize(): voice_id {voice!r} does not "
                f"start with '{VOICE_CLONE_PREFIX}'. Only cloned voices are "
                "handled by this engine."
            )

        ref_wav = self._resolve_reference(voice)
        return self._synth_f5(text, ref_wav=ref_wav, **kwargs)

    # ── Internal: reference resolution ──────────────────────────────

    def _resolve_reference(self, voice_id: str) -> Path:
        """Return the reference.wav path for voice_id, refreshing cache if needed."""
        if voice_id in self._ref_cache:
            ref = self._ref_cache[voice_id]
            if ref.exists():
                return ref

        # Profile dir not in cache (or WAV was removed): re-scan.
        self.available_voices()

        ref = self._ref_cache.get(voice_id)
        if ref is None or not ref.exists():
            raise ValueError(
                f"VoiceCloneEngine: no profile found for voice_id {voice_id!r}. "
                f"Register the voice first with voice_prep.profiler.register_voice(), "
                f"or check that {self._profiles_dir / voice_id} exists."
            )
        return ref

    # ── Internal: f5-tts-mlx backend ────────────────────────────────

    def _synth_f5(
        self,
        text: str,
        *,
        ref_wav: Path,
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        **kwargs,
    ) -> SynthesizedAudio:
        """Synthesize using f5-tts-mlx conditioned on ref_wav.

        Args:
            text:         Target text to synthesize.
            ref_wav:      Absolute path to the 24 kHz mono reference WAV.
            nfe_step:     Number of flow-matching steps.  Higher = better
                          quality but slower.  32 is a good default.
            cfg_strength: Classifier-free guidance strength.  Higher =
                          closer to the reference voice.  2.0 is default.
            **kwargs:     Any remaining kwargs are silently accepted.

        Returns:
            SynthesizedAudio at VOICE_CLONE_SAMPLE_RATE.
        """
        try:
            from f5_tts_mlx.generate import generate as f5_generate  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "VoiceCloneEngine requires f5-tts-mlx. "
                "Install with: pip install f5-tts-mlx>=0.2.6"
            ) from exc

        import numpy as np  # noqa: F401 — imported here to fail fast if absent

        with tempfile.TemporaryDirectory(prefix="vp_synth_") as tmp:
            out_path = Path(tmp) / "out.wav"

            f5_generate(
                generation_text=text,
                ref_audio_path=str(ref_wav),
                output_path=str(out_path),
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                speed=1.0,  # f5-tts speed param — separate from VoiceConfig.speed
                seed=-1,    # random seed for reproducibility control (-1 = random)
            )

            if not out_path.exists():
                raise RuntimeError(
                    f"VoiceCloneEngine: f5-tts-mlx did not produce output at {out_path}. "
                    "Check that the reference.wav is a valid 24 kHz mono WAV."
                )

            audio_bytes = out_path.read_bytes()

        return SynthesizedAudio(audio=audio_bytes, sample_rate=VOICE_CLONE_SAMPLE_RATE)
