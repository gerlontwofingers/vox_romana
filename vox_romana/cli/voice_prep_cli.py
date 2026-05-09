"""Voice-prep mode runner: clean a reference audio file and register it as a
voice-clone profile.

Extracted from run.py so the logic is importable and unit-testable
independently of the master CLI's argparse layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_voice_prep(args: argparse.Namespace) -> Path:
    """Clean a reference audio file and register it as a named voice profile.

    Reads ``args.source`` (Path), ``args.name`` (str), ``args.gender``,
    ``args.tone``, ``args.best_for``, ``args.tags``, ``args.force``, and
    ``args.skip_demucs`` from the parsed namespace.

    Resolves ``voice_profiles_dir`` from ``args.voice_profiles_dir`` (set by
    run.py from ``VOICE_PROFILES_DIR``).

    Prints the registered profile directory path on success.

    Args:
        args: Parsed namespace from run.py's parse_args().

    Returns:
        Path to the registered profile directory.
    """
    from vox_romana.voice_prep.cleaner import clean_audio
    from vox_romana.voice_prep.profiler import register_voice

    source: Path = args.vp_source
    if not source.exists():
        print(f"Error: source file not found: {source}", file=sys.stderr)
        sys.exit(1)

    profiles_dir: Path = args.voice_profiles_dir

    if not getattr(args, "quiet", False):
        print(f"Cleaning audio: {source}")

    cleaned_wav = clean_audio(source, skip_demucs=getattr(args, "skip_demucs", False))

    if not getattr(args, "quiet", False):
        print(f"Cleaned: {cleaned_wav}")
        print(f"Registering voice '{args.vp_name}' …")

    profile_dir = register_voice(
        cleaned_wav,
        args.vp_name,
        profiles_dir=profiles_dir,
        gender=getattr(args, "vp_gender", "neutral"),
        tone=getattr(args, "vp_tone", ""),
        best_for=getattr(args, "vp_best_for", ""),
        tags=getattr(args, "vp_tags", None) or [],
        force=getattr(args, "vp_force", False),
    )

    print(f"Registered vc_{args.vp_name} at {profile_dir}")
    return profile_dir
