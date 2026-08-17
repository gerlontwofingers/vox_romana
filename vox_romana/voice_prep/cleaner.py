"""voice_prep.cleaner — audio cleaning pipeline for voice clone reference files.

Three-stage pipeline:

    Stage 1 — Vocal separation (demucs, optional)
        Strips music beds and ambient noise. Skipped gracefully if demucs
        is not installed or ``skip_demucs=True`` is passed.

    Stage 2 — Silence trim + loudness normalisation
        Uses pydub to trim leading/trailing silence and normalise the
        cleaned waveform to a target loudness (default −16 dBFS, which
        approximates −16 LUFS for spoken-word material).

    Stage 3 — Resample to mono 24 kHz WAV
        f5-tts-mlx expects mono 24 kHz WAV input. This stage guarantees
        the correct format regardless of the source file.

OUTPUT:

    A single WAV file at ``{output_dir}/{stem}_cleaned.wav``.  If
    ``output_dir`` is not supplied, the file is written next to the
    source file in a sub-directory called ``_vp_cleaned``.

USAGE:

    from vox_romana.voice_prep.cleaner import clean_audio

    wav_path = clean_audio(
        "/path/to/interview.mp3",
        output_dir="/tmp/cleaned",
    )
    # → Path("/tmp/cleaned/interview_cleaned.wav")
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Target sample rate required by f5-tts-mlx
_TARGET_SR = 24_000
# Target loudness (dBFS peak after normalise)
_TARGET_DBFS = -16.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_audio(
    source_path: str | Path,
    *,
    skip_demucs: bool = False,
    output_dir: str | Path | None = None,
) -> Path:
    """Clean a voice reference audio file and return the path to the output WAV.

    Parameters
    ----------
    source_path:
        Path to any audio file readable by pydub (MP3, WAV, FLAC, …).
    skip_demucs:
        If True, skip vocal separation even if demucs is installed.
    output_dir:
        Directory in which to write the cleaned WAV.  Created if absent.
        Defaults to ``{source_parent}/_vp_cleaned/``.

    Returns
    -------
    Path
        Absolute path to the cleaned, resampled mono 24 kHz WAV file.

    Raises
    ------
    FileNotFoundError
        If ``source_path`` does not exist.
    RuntimeError
        If pydub (required stage) fails to process the audio.
    """
    source = Path(source_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"voice_prep.cleaner: source not found: {source}")

    if output_dir is None:
        out_dir = source.parent / "_vp_cleaned"
    else:
        out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = source.stem
    final_path = out_dir / f"{stem}_cleaned.wav"

    # Use a temp dir for intermediate files so we can clean up on failure
    with tempfile.TemporaryDirectory(prefix="vp_clean_") as tmp:
        working = Path(tmp) / source.name
        shutil.copy2(source, working)

        # Stage 1: vocal separation via demucs (optional)
        working = _stage_demucs(working, tmp_dir=Path(tmp), skip=skip_demucs)

        # Stage 2: silence trim + normalise (pydub)
        working = _stage_normalise(working, tmp_dir=Path(tmp))

        # Stage 3: resample to mono 24 kHz WAV
        working = _stage_resample(working, tmp_dir=Path(tmp))

        shutil.copy2(working, final_path)

    logger.info("voice_prep.cleaner: cleaned file written to %s", final_path)
    return final_path


# ---------------------------------------------------------------------------
# Internal stages
# ---------------------------------------------------------------------------


def _stage_demucs(audio_path: Path, *, tmp_dir: Path, skip: bool) -> Path:
    """Run demucs vocal separation; return path to vocals-only file (or input)."""
    if skip:
        logger.debug("voice_prep.cleaner: demucs skipped (skip_demucs=True)")
        return audio_path

    try:
        import demucs.separate  # noqa: F401  (presence check)
        import torch
    except ImportError:
        logger.warning(
            "voice_prep.cleaner: demucs not installed — skipping vocal separation. "
            "Install with: pip install demucs"
        )
        return audio_path

    vocals_dir = tmp_dir / "demucs_out"
    vocals_dir.mkdir(exist_ok=True)

    try:
        import demucs.separate as _sep

        _sep.main(
            [
                "--two-stems=vocals",
                "--out",
                str(vocals_dir),
                str(audio_path),
            ]
        )
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "voice_prep.cleaner: demucs failed (%s) — falling back to raw audio.",
            exc,
        )
        return audio_path

    # demucs writes to: {vocals_dir}/htdemucs/{stem}/vocals.wav
    # Walk and find the first file named "vocals.wav"
    vocals_candidates = list(vocals_dir.rglob("vocals.wav"))
    if not vocals_candidates:
        logger.warning(
            "voice_prep.cleaner: demucs ran but produced no vocals.wav — "
            "falling back to raw audio."
        )
        return audio_path

    result = vocals_candidates[0]
    logger.debug("voice_prep.cleaner: demucs vocals at %s", result)
    return result


def _stage_normalise(audio_path: Path, *, tmp_dir: Path) -> Path:
    """Trim silence and normalise loudness; return path to processed file."""
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_leading_silence
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "voice_prep.cleaner: pydub is required. pip install pydub"
        ) from exc

    seg: AudioSegment = AudioSegment.from_file(str(audio_path))

    # Trim leading silence
    start_trim = detect_leading_silence(seg)
    # Trim trailing silence (reverse trick)
    end_trim = detect_leading_silence(seg.reverse())
    trimmed = seg[start_trim : len(seg) - end_trim]

    if len(trimmed) == 0:
        logger.warning(
            "voice_prep.cleaner: silence trim produced an empty segment — "
            "using original audio."
        )
        trimmed = seg

    # Normalise to target dBFS
    change_db = _TARGET_DBFS - trimmed.dBFS
    normalised = trimmed.apply_gain(change_db)

    out = tmp_dir / f"{audio_path.stem}_norm.wav"
    fh = normalised.export(str(out), format="wav")
    fh.close()
    return out


def _stage_resample(audio_path: Path, *, tmp_dir: Path) -> Path:
    """Convert to mono 24 kHz WAV using pydub."""
    try:
        from pydub import AudioSegment
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "voice_prep.cleaner: pydub is required. pip install pydub"
        ) from exc

    seg: AudioSegment = AudioSegment.from_file(str(audio_path))
    mono = seg.set_channels(1).set_frame_rate(_TARGET_SR).set_sample_width(2)

    out = tmp_dir / f"{audio_path.stem}_24k.wav"
    fh = mono.export(str(out), format="wav")
    fh.close()
    return out
