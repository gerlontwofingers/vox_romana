"""Reproducibility utilities — seed fixing and environment manifest capture.

All stochastic components in the emotion_flow pipeline call
:func:`freeze_randomness` as their first action to pin the global random
state. :func:`build_manifest` records every version that could affect results,
enabling exact reproduction of any published finding.

Reproducibility is organised into three tiers:

* **Tier 1 — Seed control:** :func:`freeze_randomness` pins the Python,
  NumPy, and (optionally) PyTorch random seeds to the requested value.
  Guarantees identical results from the same input in the same process.

* **Tier 2 — Environment capture:** :func:`build_manifest` snapshots all
  package versions, Python version, and platform details into a
  :class:`~.models.ReproducibilityManifest`.  Sharing the manifest alongside
  results lets a third party reconstruct the exact software environment.

* **Tier 3 — Deterministic models:** Individual pipeline components
  (tagger, semantic flow, BERTScore) are responsible for passing
  determinism flags to their underlying models.  The manifest records the
  seed and model versions so discrepancies can be debugged.
"""

from __future__ import annotations

import logging
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from .models import ReproducibilityManifest

logger = logging.getLogger(__name__)

_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "scikit-learn",
    "torch",
    "transformers",
    "bert-score",
    "langdetect",
    "chromadb",
    "sentence-transformers",
    "statsmodels",
)


def freeze_randomness(seed: int = 42) -> None:
    """Fix all random seeds and enable deterministic algorithm selection.

    Affects: Python random, NumPy, PyTorch (if installed).
    Also sets torch.backends.cudnn.deterministic=True when CUDA is available.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("torch not available; skipping torch seed fixing.")


def build_manifest(
    seed: int,
    model_versions: dict[str, str] | None = None,
) -> ReproducibilityManifest:
    """Capture environment metadata for reproducibility."""
    pkg_versions = _collect_package_versions()
    hardware = _detect_hardware()
    git_commit = _git_short_sha()
    deterministic = _is_deterministic_mode()
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    return ReproducibilityManifest(
        seed=seed,
        python_version=sys.version.split()[0],
        package_versions=pkg_versions,
        model_versions=model_versions or {},
        hardware=hardware,
        timestamp=timestamp,
        git_commit=git_commit,
        deterministic_mode=deterministic,
    )


def _collect_package_versions() -> dict[str, str]:
    """Return installed versions for all tracked packages."""
    from importlib.metadata import PackageNotFoundError, version

    result: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            result[pkg] = version(pkg)
        except PackageNotFoundError:
            result[pkg] = "not-installed"
    return result


def _detect_hardware() -> str:
    """Return a human-readable hardware description string."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.version.cuda or "unknown CUDA"
            return f"{name}, CUDA {cap}"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return f"{platform.processor() or 'Apple Silicon'}, MPS"
    except ImportError:
        pass
    return f"{platform.processor() or platform.machine()}, CPU"


def _git_short_sha() -> str:
    """Return the current git short commit SHA, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _is_deterministic_mode() -> bool:
    """Return True if torch deterministic algorithms are currently enabled."""
    try:
        import torch
        return torch.are_deterministic_algorithms_enabled()
    except (ImportError, AttributeError):
        return False
