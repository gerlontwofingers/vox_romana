"""Duration math and speech-rate utilities.

Central place for words-per-minute defaults, duration estimation,
tolerance windows, and the SpeechRateAdvisor helper that optionally
learns from historical catalog data.

The numbers here are genre-level defaults calibrated from broadcast
standards.  The SpeechRateAdvisor can refine them using past run stats
stored in the Catalog's ``wpm_stats`` table.

USAGE:
    from vox_romana.core.duration import (
        estimate_total_words,
        tolerance_window,
        seconds_to_mmss,
        words_to_seconds,
        SpeechRateAdvisor,
    )

    # Quick math — no catalog needed
    words = estimate_total_words(duration_minutes=30, content_type="debate")
    lo, hi = tolerance_window(target_words=words, tolerance_pct=10.0)
    print(seconds_to_mmss(1800))  # "30:00"

    # With learning from historical data
    advisor = SpeechRateAdvisor(catalog)
    wpm = advisor.effective_wpm("podcast")
    words = advisor.estimate_words(duration_minutes=30, content_type="podcast")
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .catalog import Catalog


# ─────────────────────────────────────────────────────────────────────────────
# Genre-level WPM defaults
# ─────────────────────────────────────────────────────────────────────────────
# These are calibrated from broadcast/podcast standards:
#   - Debates & podcasts: conversational pace, ~150 wpm
#   - News: anchor reads are faster and tighter, ~165 wpm
#   - Audiobooks: slower, deliberate narration, ~140 wpm

WPM_DEFAULTS: dict[str, int] = {
    "debate": 150,
    "news": 165,
    "podcast": 150,
    "audiobook": 140,
}

# Fallback for unknown content types.
_WPM_FALLBACK = 150


# ─────────────────────────────────────────────────────────────────────────────
# Pure-function helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_wpm(content_type: str) -> int:
    """Return the default WPM for a content type."""
    return WPM_DEFAULTS.get(content_type, _WPM_FALLBACK)


def words_to_seconds(word_count: int, wpm: int | None = None, content_type: str = "debate") -> float:
    """Convert a word count to estimated seconds of audio.

    Args:
        word_count:   Number of words.
        wpm:          Words per minute.  Falls back to genre default.
        content_type: Used to select the genre default when *wpm* is None.

    Returns:
        Estimated duration in seconds.
    """
    effective_wpm = wpm or get_wpm(content_type)
    if effective_wpm <= 0:
        effective_wpm = _WPM_FALLBACK
    return (word_count / effective_wpm) * 60.0


def seconds_to_words(seconds: float, wpm: int | None = None, content_type: str = "debate") -> int:
    """Convert seconds to a word count estimate."""
    effective_wpm = wpm or get_wpm(content_type)
    if effective_wpm <= 0:
        effective_wpm = _WPM_FALLBACK
    return round((seconds / 60.0) * effective_wpm)


def seconds_to_mmss(seconds: float) -> str:
    """Format seconds as ``MM:SS``."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def estimate_total_words(
    duration_minutes: float,
    content_type: str = "debate",
    wpm: int | None = None,
) -> int:
    """Estimate how many words are needed for a target duration.

    Args:
        duration_minutes: Target show length in minutes.
        content_type:     Genre (selects default WPM).
        wpm:              Explicit WPM override.

    Returns:
        Estimated word count (rounded to nearest integer).
    """
    effective_wpm = wpm or get_wpm(content_type)
    return round(duration_minutes * effective_wpm)


def tolerance_window(
    target_words: int,
    tolerance_pct: float = 10.0,
) -> tuple[int, int]:
    """Return (low, high) word-count bounds for a ± tolerance.

    Args:
        target_words:  Ideal word count.
        tolerance_pct: Acceptable deviation as a percentage (default 10%).

    Returns:
        Tuple of (min_words, max_words).
    """
    margin = target_words * (tolerance_pct / 100.0)
    return (round(target_words - margin), round(target_words + margin))


def parse_duration_from_text(text: str) -> float | None:
    """Extract a duration in minutes from natural language.

    Looks for patterns like "30 minute podcast", "1 hour debate",
    "45-minute show", "a 20 min episode", "1.5 hour audiobook".

    Returns:
        Duration in minutes, or None if nothing found.
    """
    # Try "N hour(s)" first (may also have minutes after it)
    hour_pat = re.compile(
        r"(\d+(?:\.\d+)?)\s*[-–]?\s*hours?\b",
        re.IGNORECASE,
    )
    min_pat = re.compile(
        r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:minutes?|mins?)\b",
        re.IGNORECASE,
    )

    hours: float = 0.0
    minutes: float = 0.0

    h_match = hour_pat.search(text)
    if h_match:
        hours = float(h_match.group(1))

    m_match = min_pat.search(text)
    if m_match:
        minutes = float(m_match.group(1))

    total = hours * 60.0 + minutes
    return total if total > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# SpeechRateAdvisor — learns from historical run data
# ─────────────────────────────────────────────────────────────────────────────

class SpeechRateAdvisor:
    """Wraps WPM logic with optional learning from the Catalog.

    If a Catalog is provided and has ``wpm_stats`` data, the advisor uses
    the running average WPM from past runs instead of the static defaults.
    Without a Catalog (or when no history exists), it falls back to
    ``WPM_DEFAULTS``.

    Usage:
        advisor = SpeechRateAdvisor()             # static defaults only
        advisor = SpeechRateAdvisor(catalog)       # learns from history

        wpm = advisor.effective_wpm("podcast")
        words = advisor.estimate_words(30, "podcast")
        lo, hi = advisor.tolerance_window(words, tolerance_pct=10.0)
    """

    def __init__(self, catalog: Catalog | None = None) -> None:
        self._catalog = catalog

    def effective_wpm(self, content_type: str, word_count: int | None = None) -> int:
        """Return the best WPM estimate for a content type.

        Priority order:
            1. WPMModel prediction (if sklearn installed, model trained,
               and ≥5 observations exist) — ML-calibrated estimate.
            2. Catalog running average (if catalog has ≥3 observations) —
               simple historical mean.
            3. Static genre defaults (WPM_DEFAULTS) — safe starting point.

        The word_count parameter only matters for the WPMModel path,
        which uses it as a feature.  For avg/default, it's ignored.

        Args:
            content_type: Genre label (e.g. "debate", "news").
            word_count:   Optional word count for ML-based prediction.

        Returns:
            Best available WPM estimate (integer).
        """
        # Try ML model first (requires sklearn + enough data).
        if self._catalog is not None and word_count is not None:
            try:
                from vox_romana.data.wpm_model import WPMModel
                model = WPMModel.load()
                if model is not None and model.n_samples >= 5:
                    prediction = model.predict(content_type, word_count)
                    if prediction is not None:
                        return round(prediction)
            except Exception:
                pass  # Graceful degradation — fall through to averages.

        # Try catalog running average.
        if self._catalog is not None:
            stats = self._catalog.get_wpm_stats(content_type)
            if stats and stats["count"] >= 3:
                # Use the running average once we have enough samples.
                return round(stats["avg_wpm"])
        return get_wpm(content_type)

    def estimate_words(
        self,
        duration_minutes: float,
        content_type: str = "debate",
    ) -> int:
        """Estimate word count using the best available WPM."""
        wpm = self.effective_wpm(content_type)
        return round(duration_minutes * wpm)

    def tolerance_window(
        self,
        target_words: int,
        tolerance_pct: float = 10.0,
    ) -> tuple[int, int]:
        """Convenience passthrough to the module-level function."""
        return tolerance_window(target_words, tolerance_pct)

    def log_observation(
        self,
        content_type: str,
        word_count: int,
        duration_seconds: float,
    ) -> None:
        """Record a WPM observation to the catalog for future learning.

        Only writes if a catalog is available. Safe to call unconditionally.
        """
        if self._catalog is None or duration_seconds <= 0:
            return
        wpm = (word_count / duration_seconds) * 60.0
        self._catalog.log_wpm_stat(
            content_type=content_type,
            word_count=word_count,
            duration_seconds=duration_seconds,
            wpm=wpm,
        )
