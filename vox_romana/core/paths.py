"""Centralised data-directory resolution for vox_romana.

All runtime data (catalog DB, memory index, audio output, TTS models) lives
under a single configurable root.  Resolution order:

1. ``VOX_ROMANA_DATA_DIR`` environment variable  (highest priority)
2. ``vox_romana.yaml`` in the project root  (copy from vox_romana.example.yaml)
3. ``~/.vox_romana/config.yaml``  (user-level, backward compatible)
4. Default: ``~/.vox_romana/``

To customise, copy ``vox_romana.example.yaml`` → ``vox_romana.yaml`` in the
project root (next to ``run.py``) and edit the values there::

    data_dir: /Volumes/BigDisk/vox_romana_data
    # Optional — only needed if Piper voices live elsewhere:
    # piper_voices_dir: /Volumes/BigDisk/piper-voices

Every module that previously hard-coded a ``data/…`` path now imports the
relevant constant from here instead.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root = the directory containing this file's package root (3 levels up:
# paths.py → core/ → vox_romana/ → repo root).
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def _load_config() -> dict:
    """Load config from project-root yaml, then user-level yaml, else empty dict.

    Resolution order:
        1. vox_romana.yaml  (project root — copy from vox_romana.example.yaml)
        2. ~/.vox_romana/config.yaml  (user-level, backward compatible)
    """
    candidates = [
        _PROJECT_ROOT / "vox_romana.yaml",
        Path.home() / ".vox_romana" / "config.yaml",
    ]
    for config_path in candidates:
        if config_path.exists():
            try:
                import yaml  # pyyaml
                with config_path.open() as fh:
                    result = yaml.safe_load(fh) or {}
                if result:
                    return result
            except Exception:
                pass
    return {}


def _data_root() -> Path:
    """Resolve the data root from env var, config file, or default."""
    # 1. Env var wins (useful for CI and scripting).
    env = os.environ.get("VOX_ROMANA_DATA_DIR")
    if env:
        root = Path(env).expanduser()
    else:
        cfg = _load_config()
        data_dir = cfg.get("data_dir")
        root = Path(data_dir).expanduser() if data_dir else Path.home() / ".vox_romana"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _piper_voices_dir(data_root: Path) -> Path:
    """Resolve the Piper voices directory."""
    cfg = _load_config()
    override = cfg.get("piper_voices_dir")
    if override:
        return Path(override).expanduser()
    return data_root / "piper-voices"


# ── Public path constants ─────────────────────────────────────────────────────
# Evaluated once at import time.

DATA_ROOT:        Path = _data_root()

CATALOG_DB_PATH:  Path = DATA_ROOT / "vox_romana.db"
OUTPUT_DIR:       Path = DATA_ROOT / "output"
MEMORY_ROOT:      Path = DATA_ROOT / "memory"
DOCUMENT_DIR:     Path = MEMORY_ROOT / "documents"
CHROMA_DIR:            Path = MEMORY_ROOT / "chroma"
ENCODER_DIR:           Path = MEMORY_ROOT / "encoder"
FINGERPRINT_STORE_PATH: Path = MEMORY_ROOT / "fingerprints.npz"
LM_DIR:           Path = MEMORY_ROOT / "lm"
CORPUS_ROOT:      Path = MEMORY_ROOT / "corpus"
HF_CACHE_DIR:     Path = DATA_ROOT / "hf_cache"
PIPER_VOICES_DIR: Path = _piper_voices_dir(DATA_ROOT)
UNIVERSITY_ROOT:  Path = DATA_ROOT / "university"
WIKIPEDIA_CACHE_DIR: Path = DATA_ROOT / "wikipedia_cache"
