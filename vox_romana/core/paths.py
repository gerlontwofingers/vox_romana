"""Centralised data-directory resolution for vox_romana.

All runtime data (catalog DB, memory index, audio output, TTS models) lives
under a single configurable root.  Resolution order:

1. ``VOX_ROMANA_DATA_DIR`` environment variable
2. ``data_dir`` key in ``~/.vox_romana/config.yaml``
3. Default: ``~/.vox_romana/``

To customise, create ``~/.vox_romana/config.yaml``::

    data_dir: /Volumes/BigDisk/vox_romana_data
    # Optional — only needed if Piper models live elsewhere:
    # piper_voices_dir: /Volumes/BigDisk/piper-voices

Every module that previously hard-coded a ``data/…`` path now imports the
relevant constant from here instead.
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_config() -> dict:
    """Load ~/.vox_romana/config.yaml if it exists, else return empty dict."""
    config_path = Path.home() / ".vox_romana" / "config.yaml"
    if config_path.exists():
        try:
            import yaml  # pyyaml
            with config_path.open() as fh:
                return yaml.safe_load(fh) or {}
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
CHROMA_DIR:       Path = MEMORY_ROOT / "chroma"
ENCODER_DIR:      Path = MEMORY_ROOT / "encoder"
LM_DIR:           Path = MEMORY_ROOT / "lm"
CORPUS_ROOT:      Path = MEMORY_ROOT / "corpus"
PIPER_VOICES_DIR: Path = _piper_voices_dir(DATA_ROOT)
