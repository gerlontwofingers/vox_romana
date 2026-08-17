"""Voice-assign mode runner: lock a cloned voice to a series character and
record it in the content catalog.

Extracted from run.py so the logic is importable and unit-testable
independently of the master CLI's argparse layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_voice_assign(args: argparse.Namespace) -> None:
    """Assign a cloned voice ID to a named character in a series.

    Reads ``args.voice_id``, ``args.character``, ``args.series_id``, and
    ``args.voice_profiles_dir`` from the parsed namespace.

    * Registers the character→voice lock in ``VoiceRegistry`` (SQLite).
    * Records the cloned-voice asset in the content ``Catalog``.

    Args:
        args: Parsed namespace from run.py's parse_args().
    """
    from vox_romana.core.catalog import Catalog
    from vox_romana.core.paths import CATALOG_DB_PATH
    from vox_romana.tts.voice_registry import VoiceRegistry

    voice_id: str = args.voice_id
    character: str = args.va_character
    series_id: str = args.series_id
    profiles_dir: Path = args.voice_profiles_dir

    if not voice_id.startswith("vc_"):
        print(
            f"Error: voice ID must start with 'vc_' (got: {voice_id!r})",
            file=sys.stderr,
        )
        sys.exit(1)

    profile_dir = profiles_dir / voice_id
    if not profile_dir.exists():
        print(
            f"Warning: profile directory not found: {profile_dir}",
            file=sys.stderr,
        )

    # Lock character → voice in VoiceRegistry.
    registry = VoiceRegistry()
    registry.register(character, voice_id, series_id)

    # Record in content catalog.
    catalog = Catalog(CATALOG_DB_PATH)
    catalog.register_cloned_asset(
        series_id=series_id,
        voice_id=voice_id,
        character_name=character,
        profile_dir=str(profile_dir),
    )

    print(f"Assigned {voice_id} to '{character}' in series {series_id}")
