"""voice_prep.profiler — register a cleaned voice WAV as a reusable voice profile.

A voice profile is a directory containing:

    {profiles_dir}/vc_{name}/
        reference.wav    — the cleaned 24 kHz mono WAV reference clip
        metadata.json    — structured descriptor (gender, tone, tags, …)

The ``vc_`` prefix keeps cloned voice IDs in a distinct namespace, separate
from Kokoro (``af_*``, ``am_*``, ``bf_*``, ``bm_*``) and Piper (``en_US-*``)
voice IDs.  This prevents silent collisions in the catalog and registry.

USAGE:

    from vox_romana.voice_prep.profiler import register_voice

    profile_dir = register_voice(
        cleaned_wav="/tmp/cleaned/interview_cleaned.wav",
        voice_name="marcus_deep",
        profiles_dir="/data/voice_profiles",
        gender="male",
        tone="deep, measured",
        best_for="narration, debate",
        tags=["deep", "authoritative", "measured", "male"],
    )
    # → Path("/data/voice_profiles/vc_marcus_deep")
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Only lowercase letters, digits, and underscores are allowed in voice names.
_NAME_RE = re.compile(r"^[a-z0-9_]+$")

# Prefix that every cloned voice ID carries
VOICE_CLONE_PREFIX = "vc_"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_voice(
    cleaned_wav: str | Path,
    voice_name: str,
    *,
    profiles_dir: str | Path,
    gender: str,
    tone: str,
    best_for: str,
    tags: list[str],
    force: bool = False,
) -> Path:
    """Register a cleaned WAV as a named voice profile.

    Parameters
    ----------
    cleaned_wav:
        Path to the cleaned 24 kHz mono WAV produced by ``clean_audio()``.
    voice_name:
        Human-friendly name for the voice (e.g. ``"marcus_deep"``).
        Must consist only of lowercase ASCII letters, digits, and underscores.
        The final ``voice_id`` will be ``vc_{voice_name}``.
    profiles_dir:
        Root directory where voice profiles are stored.  The directory
        ``{profiles_dir}/vc_{voice_name}/`` will be created.
    gender:
        ``"male"`` or ``"female"`` (used for catalog matching).
    tone:
        Short tonal description, e.g. ``"warm, measured"``.
    best_for:
        Comma-separated use-cases, e.g. ``"narration, debate"``.
    tags:
        Flat list of searchable keyword strings for voice matching.
    force:
        If True, overwrite an existing profile with the same name.
        If False (default), raise ``ValueError`` on collision.

    Returns
    -------
    Path
        Absolute path to the created profile directory.

    Raises
    ------
    FileNotFoundError
        If ``cleaned_wav`` does not exist.
    ValueError
        If ``voice_name`` contains disallowed characters, or if a profile
        with the same name already exists and ``force=False``.
    """
    wav = Path(cleaned_wav).resolve()
    if not wav.exists():
        raise FileNotFoundError(
            f"voice_prep.profiler: cleaned_wav not found: {wav}"
        )

    _validate_name(voice_name)
    voice_id = f"{VOICE_CLONE_PREFIX}{voice_name}"

    root = Path(profiles_dir).resolve()
    profile_dir = root / voice_id

    if profile_dir.exists():
        if not force:
            raise ValueError(
                f"voice_prep.profiler: profile '{voice_id}' already exists at "
                f"{profile_dir}. Pass force=True to overwrite."
            )
        logger.warning(
            "voice_prep.profiler: overwriting existing profile at %s", profile_dir
        )
        shutil.rmtree(profile_dir)

    profile_dir.mkdir(parents=True, exist_ok=True)

    # Copy reference wav
    reference_wav = profile_dir / "reference.wav"
    shutil.copy2(wav, reference_wav)

    # Write metadata
    metadata = {
        "voice_id": voice_id,
        "voice_name": voice_name,
        "gender": gender,
        "tone": tone,
        "best_for": best_for,
        "tags": tags,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": wav.name,
    }
    (profile_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "voice_prep.profiler: registered voice '%s' at %s", voice_id, profile_dir
    )
    return profile_dir


def load_profile_metadata(profile_dir: str | Path) -> dict:
    """Load and return the metadata dict for an existing voice profile.

    Parameters
    ----------
    profile_dir:
        Path to a profile directory containing ``metadata.json``.

    Raises
    ------
    FileNotFoundError
        If ``metadata.json`` is missing from ``profile_dir``.
    """
    meta_path = Path(profile_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"voice_prep.profiler: metadata.json not found in {profile_dir}"
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def list_profiles(profiles_dir: str | Path) -> list[dict]:
    """Return metadata for all registered voice profiles in ``profiles_dir``.

    Profiles without a readable ``metadata.json`` are silently skipped.
    """
    root = Path(profiles_dir)
    if not root.exists():
        return []

    results: list[dict] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name.startswith(VOICE_CLONE_PREFIX):
            try:
                results.append(load_profile_metadata(entry))
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                logger.warning(
                    "voice_prep.profiler: skipping malformed profile %s: %s",
                    entry.name,
                    exc,
                )
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_name(voice_name: str) -> None:
    if not voice_name:
        raise ValueError("voice_prep.profiler: voice_name must not be empty.")
    if not _NAME_RE.match(voice_name):
        raise ValueError(
            f"voice_prep.profiler: voice_name '{voice_name}' is invalid. "
            "Use only lowercase letters (a-z), digits (0-9), and underscores (_)."
        )
