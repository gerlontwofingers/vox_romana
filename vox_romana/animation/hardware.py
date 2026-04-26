"""HardwareScanner — detect compute device and recommend animation settings.

This module probes the current machine and returns a HardwareProfile that
tells the rest of the animation pipeline what it can do.

DEVICE PRIORITY (mirrors music_engine.py):

    1. MPS   — Apple Silicon GPU (M-series chips, unified memory)
    2. CUDA  — NVIDIA GPU (discrete VRAM, Linux/Windows)
    3. CPU   — Universal fallback (slow but always works)

TIER TABLE:

    The scanner maps hardware capabilities to a recommended SD model,
    output resolution, and frame rate:

    ┌──────────────────────────────┬──────────────────────────────┬───────────┬─────┐
    │ Condition                    │ SD model                     │ Res       │ FPS │
    ├──────────────────────────────┼──────────────────────────────┼───────────┼─────┤
    │ MPS + unified ≥ 16 GB        │ SDXL base 1.0                │ 1920×1080 │  30 │
    │ CUDA ≥ 10 GB VRAM            │ SDXL base 1.0                │ 1920×1080 │  30 │
    │ CUDA 4–9.9 GB VRAM           │ stable-diffusion-2-1         │ 1280×720  │  24 │
    │ CUDA < 4 GB / MPS < 16 GB   │ stable-diffusion-v1-5        │ 512×512   │  24 │
    │ CPU only                     │ stable-diffusion-v1-5        │ 512×512   │  24 │
    └──────────────────────────────┴──────────────────────────────┴───────────┴─────┘

    devstation_01 (M1 Max, 32 GB unified, MPS):
        → SDXL + 1920×1080 + 30fps  ✓

DISK WARNING:

    SD model caches are 4–7 GB on first download (SDXL is ~7 GB).
    The scanner emits ``low_disk_warning = True`` when free disk space is
    below LOW_DISK_THRESHOLD_GB (default: 10 GB) so callers can warn the
    user before attempting a download.

GRACEFUL DEGRADATION:

    If torch is not installed, the scanner falls back to CPU mode without
    crashing. This mirrors the pattern in music_engine.py — tests and
    environments without GPU deps work correctly.

USAGE:

    from vox_romana.animation.hardware import HardwareScanner

    profile = HardwareScanner.scan()
    print(profile.device)                     # "mps"
    print(profile.recommended_sd_model)       # "stabilityai/stable-diffusion-xl-base-1.0"
    print(profile.recommended_resolution)     # (1920, 1080)
    print(profile.recommended_fps)            # 30
    if profile.low_disk_warning:
        print(f"Warning: only {profile.free_disk_gb:.1f} GB free — SD download may fail")
"""

from __future__ import annotations

import logging
import shutil

from .models import HardwareProfile

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# Warn when free disk space is below this threshold (GB).
# SDXL weights are ~7 GB; leave headroom for outputs too.
LOW_DISK_THRESHOLD_GB: float = 10.0

# SD model identifiers — HuggingFace canonical IDs
_SD_MODELS = {
    "sdxl":  "stabilityai/stable-diffusion-xl-base-1.0",
    "sd21":  "stabilityai/stable-diffusion-2-1",
    "sd15":  "runwayml/stable-diffusion-v1-5",
}

# VRAM threshold (GB) for CUDA tier decisions
_CUDA_HIGH_VRAM_GB: float = 10.0
_CUDA_MID_VRAM_GB: float  =  4.0

# Unified memory threshold (GB) for MPS tier decisions
# M1 Max / M2 Max / M1 Ultra all ship with ≥ 16 GB, so 16 GB is the cutoff.
_MPS_HIGH_UNIFIED_GB: float = 16.0


# ═══════════════════════════════════════════════════════════════════════════
# HARDWARE SCANNER
# ═══════════════════════════════════════════════════════════════════════════

class HardwareScanner:
    """Detect compute device and return recommended animation settings.

    All logic is in the class method ``scan()`` — no instance state needed.
    The class is a namespace rather than a true singleton; it exists for
    extension (subclass and override ``scan()`` in tests or custom configs).
    """

    @classmethod
    def scan(cls) -> HardwareProfile:
        """Probe the current machine and return a HardwareProfile.

        No side effects — safe to call multiple times. The first call may
        take a moment if torch needs to initialize its device backend.

        Returns:
            A populated HardwareProfile with device, VRAM, recommended
            SD model, resolution, FPS, and disk status.
        """
        device = cls._select_device()
        vram_gb = cls._probe_vram(device)
        total_ram_gb = cls._probe_total_ram()
        free_disk_gb = cls._probe_free_disk()

        sd_model, resolution, fps = cls._choose_tier(
            device, vram_gb, total_ram_gb
        )

        low_disk = free_disk_gb < LOW_DISK_THRESHOLD_GB

        if device == "cpu":
            logger.warning(
                "HardwareScanner: no GPU detected — animation will use CPU. "
                "Image generation will be slow. Consider installing PyTorch "
                "with MPS (Apple Silicon) or CUDA (NVIDIA) support."
            )

        if low_disk:
            logger.warning(
                "HardwareScanner: only %.1f GB disk free (threshold: %.1f GB). "
                "SD model downloads (~4–7 GB) may fail. Free up space first.",
                free_disk_gb,
                LOW_DISK_THRESHOLD_GB,
            )

        profile = HardwareProfile(
            device=device,
            vram_gb=vram_gb,
            total_ram_gb=total_ram_gb,
            recommended_sd_model=sd_model,
            recommended_resolution=resolution,
            recommended_fps=fps,
            low_disk_warning=low_disk,
            free_disk_gb=free_disk_gb,
        )

        logger.info(
            "HardwareScanner: device=%s vram=%.1f GB → %s %s@%dfps",
            device,
            vram_gb or 0.0,
            sd_model.split("/")[-1],
            "x".join(str(d) for d in resolution),
            fps,
        )

        return profile

    # ── Device selection ──────────────────────────────────────────────────

    @staticmethod
    def _select_device() -> str:
        """Return the best available device string: "mps", "cuda", or "cpu".

        Mirrors ``_select_device()`` in tts/music_engine.py.
        Gracefully returns "cpu" if torch is not installed.
        """
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # pragma: no cover
            pass
        return "cpu"

    # ── VRAM / memory probing ─────────────────────────────────────────────

    @staticmethod
    def _probe_vram(device: str) -> float | None:
        """Probe GPU memory in GB.

        For MPS (Apple Silicon unified memory), returns the full system
        RAM as the effective VRAM budget — the GPU and CPU share the pool.

        For CUDA, returns the total VRAM of the first GPU device.

        Returns None when torch is not installed or probing fails.
        """
        try:
            import torch

            if device == "mps":
                # On Apple Silicon, unified memory = system RAM.
                # Return total RAM as the effective GPU budget.
                return HardwareScanner._probe_total_ram()

            if device == "cuda":
                props = torch.cuda.get_device_properties(0)
                return props.total_memory / (1024 ** 3)

        except Exception as exc:
            logger.debug("HardwareScanner: could not probe VRAM: %s", exc)

        return None

    @staticmethod
    def _probe_total_ram() -> float | None:
        """Return total system RAM in GB, or None if not detectable."""
        try:
            import psutil  # type: ignore[import]
            return psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            pass

        # Fallback: read /proc/meminfo on Linux
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
        except (FileNotFoundError, ValueError):
            pass

        return None

    # ── Disk probing ──────────────────────────────────────────────────────

    @staticmethod
    def _probe_free_disk(path: str = "/") -> float:
        """Return free disk space in GB at ``path``."""
        try:
            _, _, free = shutil.disk_usage(path)
            return free / (1024 ** 3)
        except Exception as exc:
            logger.debug("HardwareScanner: could not probe disk: %s", exc)
            return 999.0  # assume fine if we can't check

    # ── Tier selection ────────────────────────────────────────────────────

    @classmethod
    def _choose_tier(
        cls,
        device: str,
        vram_gb: float | None,
        total_ram_gb: float | None,
    ) -> tuple[str, tuple[int, int], int]:
        """Map hardware capabilities to (sd_model, resolution, fps).

        Args:
            device:       "mps", "cuda", or "cpu"
            vram_gb:      GPU VRAM in GB (or unified RAM for MPS)
            total_ram_gb: Total system RAM in GB

        Returns:
            (sd_model_id, (width, height), fps)
        """
        if device == "mps":
            # Apple Silicon: use unified RAM as the effective budget.
            # If we couldn't probe RAM, assume high-tier (M-series ships ≥ 16 GB).
            effective_gb = vram_gb or total_ram_gb or _MPS_HIGH_UNIFIED_GB
            if effective_gb >= _MPS_HIGH_UNIFIED_GB:
                return _SD_MODELS["sdxl"], (1920, 1080), 30
            else:
                return _SD_MODELS["sd15"], (512, 512), 24

        if device == "cuda":
            effective_gb = vram_gb or 0.0
            if effective_gb >= _CUDA_HIGH_VRAM_GB:
                return _SD_MODELS["sdxl"], (1920, 1080), 30
            if effective_gb >= _CUDA_MID_VRAM_GB:
                return _SD_MODELS["sd21"], (1280, 720), 24
            return _SD_MODELS["sd15"], (512, 512), 24

        # CPU fallback — lowest settings
        return _SD_MODELS["sd15"], (512, 512), 24
