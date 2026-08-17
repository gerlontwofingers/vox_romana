"""Continuous disk and RAM monitoring for long-running pipelines.

The pipeline has several stages that can consume resources gradually rather
than all at startup: model downloads, audio rendering, image generation,
moviepy compositing, and OS-level swap growth. A one-time preflight check
is not sufficient on its own, so this module provides a background monitor
that samples resources continuously and raises at explicit checkpoints.

Design goals:
    * Portable: only stdlib disk probing; RAM probing degrades gracefully.
    * Scalable: watch multiple relevant volumes, not just one output path.
    * Non-invasive: long-running code stays synchronous; a daemon thread does
      the sampling and callers decide where checkpoints belong.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .duration import parse_duration_from_text
from .paths import DATA_ROOT, HF_CACHE_DIR, OUTPUT_DIR, VOICE_PROFILES_DIR

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_SEC = 20.0
_DEFAULT_DURATION_SEC = 10.0 * 60.0
_PLATFORM_DEFAULTS: dict[str, tuple[tuple[int, int], int]] = {
    "youtube": ((1920, 1080), 30),
    "tiktok": ((1080, 1920), 30),
    "both": ((1920, 1080), 30),
    "fast": ((512, 512), 15),
}
_MODE_DEFAULTS: dict[str, tuple[float, float]] = {
    "generate": (1.0, 2.0),
    "render": (3.0, 4.0),
    "full": (4.0, 5.0),
    "video": (15.0, 10.0),
    "music": (5.0, 6.0),
    "voice-prep": (3.0, 4.0),
    "publish": (2.0, 2.0),
    "translate": (2.0, 3.0),
    "ingest": (2.0, 3.0),
}


class ResourceError(RuntimeError):
    """Raised when a resource budget is exceeded."""


def _resolve_probe_path(path: str | Path) -> Path:
    """Return an existing path suitable for disk_usage probing."""
    candidate = Path(path).expanduser().resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else Path("/")


def free_disk_gb(path: str | Path) -> float:
    """Return free disk space in GB for the volume backing ``path``."""
    probe_path = _resolve_probe_path(path)
    try:
        _, _, free = shutil.disk_usage(probe_path)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.debug("ResourceMonitor: disk probe failed for %s: %s", probe_path, exc)
        return 9999.0
    return free / (1024 ** 3)


def available_ram_gb() -> float | None:
    """Return currently available system RAM in GB, or None if unavailable."""
    try:
        import psutil  # type: ignore[import]
    except ImportError:
        return None

    try:
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.debug("ResourceMonitor: RAM probe failed: %s", exc)
        return None


@dataclass(slots=True, frozen=True)
class ResourceSnapshot:
    """Latest sampled resource state."""

    disk_path: Path
    free_disk_gb: float
    available_ram_gb: float | None
    sampled_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class ResourceBudget:
    """Thresholds and paths for a single monitor instance."""

    disk_paths: tuple[Path, ...]
    disk_warn_gb: float
    disk_abort_gb: float
    ram_warn_gb: float | None = None
    ram_abort_gb: float | None = None
    poll_interval_sec: float = _DEFAULT_POLL_INTERVAL_SEC

    def __post_init__(self) -> None:
        normalized = []
        seen: set[str] = set()
        for path in self.disk_paths:
            probe_path = _resolve_probe_path(path)
            key = str(probe_path)
            if key not in seen:
                normalized.append(probe_path)
                seen.add(key)
        if not normalized:
            raise ValueError("disk_paths must contain at least one path")
        if self.disk_abort_gb <= 0:
            raise ValueError("disk_abort_gb must be > 0")
        if self.disk_warn_gb < self.disk_abort_gb:
            raise ValueError("disk_warn_gb must be >= disk_abort_gb")
        if self.ram_abort_gb is not None and self.ram_abort_gb <= 0:
            raise ValueError("ram_abort_gb must be > 0 when provided")
        if (
            self.ram_warn_gb is not None
            and self.ram_abort_gb is not None
            and self.ram_warn_gb < self.ram_abort_gb
        ):
            raise ValueError("ram_warn_gb must be >= ram_abort_gb")
        if self.poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be > 0")
        self.disk_paths = tuple(normalized)

    @classmethod
    def for_mode(
        cls,
        mode: str,
        args: argparse.Namespace | None = None,
    ) -> "ResourceBudget":
        """Derive a budget from CLI mode and known runtime hints."""
        normalized_mode = (mode or "render").strip().lower()
        disk_paths = tuple(_collect_disk_paths(args))

        if normalized_mode == "video":
            platform = str(getattr(args, "video_platform", "youtube") or "youtube").lower()
            resolution, fps = _PLATFORM_DEFAULTS.get(platform, _PLATFORM_DEFAULTS["youtube"])
            duration_sec = _estimate_duration_seconds(args, normalized_mode)
            with_music = bool(getattr(args, "music", False))
            disk_abort_gb = _estimate_video_disk_abort_gb(
                duration_sec=duration_sec,
                resolution=resolution,
                fps=fps,
                platform=platform,
                with_music=with_music,
            )
            ram_abort_gb = _estimate_video_ram_abort_gb(
                resolution=resolution,
                fps=fps,
                platform=platform,
                with_music=with_music,
            )
        else:
            disk_abort_gb, ram_abort_gb = _MODE_DEFAULTS.get(normalized_mode, (3.0, 4.0))

        disk_warn_gb = max(disk_abort_gb * 1.5, disk_abort_gb + 2.0)
        ram_warn_gb = None
        if ram_abort_gb is not None:
            ram_warn_gb = max(ram_abort_gb * 1.4, ram_abort_gb + 2.0)

        return cls(
            disk_paths=disk_paths,
            disk_warn_gb=round(disk_warn_gb, 1),
            disk_abort_gb=round(disk_abort_gb, 1),
            ram_warn_gb=None if ram_warn_gb is None else round(ram_warn_gb, 1),
            ram_abort_gb=None if ram_abort_gb is None else round(ram_abort_gb, 1),
            poll_interval_sec=float(getattr(args, "resource_poll_interval", _DEFAULT_POLL_INTERVAL_SEC)),
        )


class ResourceMonitor:
    """Background sampler for disk and RAM that aborts at checkpoints."""

    def __init__(self, budget: ResourceBudget) -> None:
        self.budget = budget
        self._stop_event = threading.Event()
        self._warn_event = threading.Event()
        self._abort_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_snapshot: ResourceSnapshot | None = None
        self._warning_message: str | None = None
        self._abort_message: str | None = None
        self._warning_emitted = False

    @classmethod
    def for_mode(
        cls,
        mode: str,
        args: argparse.Namespace | None = None,
    ) -> "ResourceMonitor":
        """Construct a monitor from a CLI mode and argument namespace."""
        return cls(ResourceBudget.for_mode(mode, args))

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False

    @property
    def should_abort(self) -> bool:
        return self._abort_event.is_set()

    @property
    def abort_reason(self) -> str | None:
        return self._abort_message

    @property
    def warning_reason(self) -> str | None:
        return self._warning_message

    def start(self) -> None:
        """Start background sampling if it is not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._sample()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="vox-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop background sampling."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def snapshot(self) -> ResourceSnapshot:
        """Return the latest resource snapshot, sampling immediately if needed."""
        self._sample()
        assert self._latest_snapshot is not None
        return self._latest_snapshot

    def checkpoint(self, label: str = "") -> ResourceSnapshot:
        """Raise if a previously sampled abort threshold has been crossed.

        When RAM appears low on the first sample, a single confirmation
        re-sample is taken after a short delay before deciding to warn or
        abort.  This prevents false positives immediately after a child
        process exits with a signal (e.g. MLX SIGBUS on teardown), where
        the OS may not have finished returning wired pages to the free pool
        yet.  The 1-second wait is only incurred when RAM is in the
        warning/abort zone, so normal runs pay nothing.

        Note: the background thread's ``_warn_event`` and ``_abort_event``
        are sticky (set once, never cleared), so the confirmation re-sample
        evaluates the fresh snapshot directly rather than trusting those
        flags.
        """
        snapshot = self.snapshot()

        # If RAM looks borderline, wait 1 s and re-sample once to confirm.
        # Guards against transient dips right after a subprocess signal exit.
        if (
            snapshot.available_ram_gb is not None
            and self.budget.ram_warn_gb is not None
            and snapshot.available_ram_gb <= self.budget.ram_warn_gb
        ):
            time.sleep(1.0)
            snapshot = self._sample_fresh()

        # Re-evaluate warn/abort against the (possibly updated) snapshot.
        warn_message = self._build_warning(snapshot)
        abort_message = self._build_abort(snapshot)

        if warn_message and not self._warning_emitted:
            logger.warning(self._prefix(label, warn_message))
            self._warning_emitted = True
        if abort_message:
            raise ResourceError(self._prefix(label, abort_message))

        ram_str = f"{snapshot.available_ram_gb:.1f} GB" if snapshot.available_ram_gb is not None else "N/A"
        logger.info(
            "ResourceMonitor: checkpoint %r \u2014 disk %.1f GB free (%s) | RAM %s",
            label, snapshot.free_disk_gb, snapshot.disk_path, ram_str,
        )
        return snapshot

    def _run(self) -> None:
        while not self._stop_event.wait(self.budget.poll_interval_sec):
            self._sample()

    def _sample(self) -> None:
        snapshot = self._sample_fresh()
        warning_message = self._build_warning(snapshot)
        abort_message = self._build_abort(snapshot)

        with self._lock:
            self._latest_snapshot = snapshot
            if warning_message and self._warning_message is None:
                self._warning_message = warning_message
                self._warn_event.set()
            if abort_message and self._abort_message is None:
                self._abort_message = abort_message
                self._abort_event.set()

    def _sample_fresh(self) -> ResourceSnapshot:
        """Take an immediate resource sample without updating sticky state.

        Used by ``checkpoint()`` for the confirmation re-sample so that a
        transient low reading from the background thread cannot permanently
        arm the sticky abort/warn events before the OS finishes reclaiming
        subprocess memory.
        """
        disk_path, free_gb = self._most_constrained_disk()
        ram_gb = available_ram_gb()
        snapshot = ResourceSnapshot(
            disk_path=disk_path,
            free_disk_gb=free_gb,
            available_ram_gb=ram_gb,
        )
        with self._lock:
            self._latest_snapshot = snapshot
        return snapshot

    def _most_constrained_disk(self) -> tuple[Path, float]:
        probe_paths = self.budget.disk_paths
        best_path = probe_paths[0]
        best_free = free_disk_gb(best_path)
        for path in probe_paths[1:]:
            free = free_disk_gb(path)
            if free < best_free:
                best_path = path
                best_free = free
        return best_path, best_free

    def _build_warning(self, snapshot: ResourceSnapshot) -> str | None:
        parts = []
        if snapshot.free_disk_gb <= self.budget.disk_warn_gb:
            parts.append(
                f"low disk: {snapshot.free_disk_gb:.1f} GB free at {snapshot.disk_path} "
                f"(warn at {self.budget.disk_warn_gb:.1f} GB)"
            )
        if (
            snapshot.available_ram_gb is not None
            and self.budget.ram_warn_gb is not None
            and snapshot.available_ram_gb <= self.budget.ram_warn_gb
        ):
            parts.append(
                f"low RAM: {snapshot.available_ram_gb:.1f} GB available "
                f"(warn at {self.budget.ram_warn_gb:.1f} GB)"
            )
        if not parts:
            return None
        return "; ".join(parts)

    def _build_abort(self, snapshot: ResourceSnapshot) -> str | None:
        parts = []
        if snapshot.free_disk_gb <= self.budget.disk_abort_gb:
            parts.append(
                f"disk exhausted: {snapshot.free_disk_gb:.1f} GB free at {snapshot.disk_path} "
                f"(needs > {self.budget.disk_abort_gb:.1f} GB)"
            )
        if (
            snapshot.available_ram_gb is not None
            and self.budget.ram_abort_gb is not None
            and snapshot.available_ram_gb <= self.budget.ram_abort_gb
        ):
            parts.append(
                f"RAM exhausted: {snapshot.available_ram_gb:.1f} GB available "
                f"(needs > {self.budget.ram_abort_gb:.1f} GB)"
            )
        if not parts:
            return None
        return "; ".join(parts)

    @staticmethod
    def _prefix(label: str, message: str) -> str:
        if not label:
            return message
        return f"[{label}] {message}"


def _collect_disk_paths(args: argparse.Namespace | None) -> tuple[Path, ...]:
    candidates = [
        Path("/"),
        Path(tempfile.gettempdir()),
        DATA_ROOT,
        OUTPUT_DIR,
        HF_CACHE_DIR,
        VOICE_PROFILES_DIR,
    ]
    if args is None:
        return tuple(_resolve_probe_path(path) for path in candidates)

    for attr in (
        "audio_output",
        "video_output",
        "script_output",
        "script_input",
        "prompt",
        "vp_source",
    ):
        value = getattr(args, attr, None)
        if value:
            candidates.append(_disk_path_for_value(value))

    return tuple(_resolve_probe_path(path) for path in candidates)


def _disk_path_for_value(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    return path.parent if path.suffix else path


def _estimate_duration_seconds(
    args: argparse.Namespace | None,
    mode: str,
) -> float:
    if args is None:
        return _DEFAULT_DURATION_SEC

    if mode == "music":
        seconds = getattr(args, "music_duration", None)
        return float(seconds) if seconds else 30.0

    minutes = getattr(args, "duration", None)
    if minutes:
        return float(minutes) * 60.0

    prompt_path = getattr(args, "prompt", None)
    if prompt_path:
        try:
            prompt_text = Path(prompt_path).read_text(encoding="utf-8")
            inferred_minutes = parse_duration_from_text(prompt_text)
            if inferred_minutes:
                return float(inferred_minutes) * 60.0
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("ResourceMonitor: could not infer duration from %s: %s", prompt_path, exc)

    return _DEFAULT_DURATION_SEC


def _estimate_video_disk_abort_gb(
    *,
    duration_sec: float,
    resolution: tuple[int, int],
    fps: int,
    platform: str,
    with_music: bool,
) -> float:
    width, height = resolution
    platform_floor = {
        "fast": 4.0,
        "youtube": 15.0,
        "tiktok": 15.0,
        "both": 20.0,
    }.get(platform, 15.0)
    resolution_scale = max(1.0, (width * height) / (1920 * 1080))
    fps_scale = max(1.0, fps / 30.0)
    duration_scale = max(1.0, (max(duration_sec, 60.0) / _DEFAULT_DURATION_SEC) ** 0.5)
    estimate = platform_floor * resolution_scale * fps_scale * duration_scale
    if with_music:
        estimate += 2.0
    return max(platform_floor, estimate)


def _estimate_video_ram_abort_gb(
    *,
    resolution: tuple[int, int],
    fps: int,
    platform: str,
    with_music: bool,
) -> float:
    width, height = resolution
    # RAM needed by the animation stack (diffusion model + moviepy).
    # MusicGen has already finished by the time animation runs, so
    # with_music does NOT add overhead here — it was relevant during
    # TTS render, not during diffusion.
    # fast = SD 1.5 @ 512x512, 4 steps (~2 GB peak).
    platform_floor = {
        "fast": 3.0,
        "youtube": 10.0,
        "tiktok": 10.0,
        "both": 12.0,
    }.get(platform, 10.0)
    resolution_scale = max(1.0, (width * height) / (1920 * 1080))
    fps_scale = max(1.0, fps / 30.0)
    estimate = platform_floor * resolution_scale * fps_scale
    return max(platform_floor, estimate)


__all__ = [
    "ResourceBudget",
    "ResourceError",
    "ResourceMonitor",
    "ResourceSnapshot",
    "available_ram_gb",
    "free_disk_gb",
]