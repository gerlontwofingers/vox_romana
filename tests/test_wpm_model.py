"""Tests for the WPMModel — data-driven speech rate predictor.

Tests cover:
    1. Feature encoding (one-hot + word count)
    2. fit() with enough observations
    3. fit() with insufficient observations (< MIN_OBSERVATIONS)
    4. predict() before fitting (returns None)
    5. predict() after fitting (returns reasonable WPM)
    6. predict() clamps to [80, 300] range
    7. save/load round-trip via joblib
    8. load() when file doesn't exist (returns None)
    9. update_from_catalog() with mocked catalog
    10. update_from_catalog() with empty catalog
    11. Graceful degradation when sklearn is not installed

All tests use in-memory data — no real catalog or model files on disk
(tmp_path fixtures handle any file I/O).

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_wpm_model.py -v
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.data.wpm_model import (
    KNOWN_CONTENT_TYPES,
    MIN_OBSERVATIONS,
    WPMModel,
    _HAS_SKLEARN,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

def _make_records(n: int = 10, content_type: str = "debate", base_wpm: float = 150.0) -> list[dict]:
    """Generate synthetic WPM observation records for testing.

    Creates N records with slight WPM variation around base_wpm
    and word counts between 2000 and 6000.

    Args:
        n:            Number of records to generate.
        content_type: Genre for all records.
        base_wpm:     Center of the WPM distribution.

    Returns:
        List of dicts with content_type, word_count, wpm keys.
    """
    records = []
    for i in range(n):
        word_count = 2000 + i * 400
        wpm = base_wpm + (i % 5) * 2 - 4  # range: base-4 to base+4
        records.append({
            "content_type": content_type,
            "word_count": word_count,
            "wpm": wpm,
        })
    return records


def _make_mixed_records() -> list[dict]:
    """Generate records spanning multiple content types."""
    records = []
    for ct, base_wpm in [("debate", 150), ("news", 165), ("podcast", 148)]:
        records.extend(_make_records(n=8, content_type=ct, base_wpm=base_wpm))
    return records


@pytest.fixture
def model():
    """Fresh, unfitted WPMModel."""
    return WPMModel()


@pytest.fixture
def fitted_model(tmp_path):
    """A WPMModel trained on synthetic debate data, saving to tmp_path."""
    m = WPMModel()
    records = _make_records(n=15, content_type="debate", base_wpm=150.0)
    # Patch the save path so we don't write to real data/models/.
    with patch("vox_romana.data.wpm_model.DEFAULT_MODEL_PATH", tmp_path / "wpm_model.pkl"):
        m.fit(records)
    return m


@pytest.fixture
def mock_catalog():
    """A mock Catalog that returns synthetic wpm_stats rows."""
    cat = MagicMock()
    # update_from_catalog now calls the public get_wpm_stats_all() method.
    rows = _make_mixed_records()  # plain dicts — no Row wrapper needed
    cat.get_wpm_stats_all.return_value = rows
    return cat


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENCODING
# ═══════════════════════════════════════════════════════════════════════════

class TestEncoding:
    """Verify the one-hot + word_count feature encoding."""

    def test_debate_encoding(self, model):
        """Debate should have [1, 0, 0, 0, 0, word_count]."""
        features = model._encode("debate", 4500)
        assert features == [1.0, 0.0, 0.0, 0.0, 0.0, 4500.0]

    def test_news_encoding(self, model):
        """News should have [0, 1, 0, 0, 0, word_count]."""
        features = model._encode("news", 3000)
        assert features == [0.0, 1.0, 0.0, 0.0, 0.0, 3000.0]

    def test_unknown_type_encoding(self, model):
        """Unknown content types get all-zero one-hot."""
        features = model._encode("mystery_genre", 2000)
        assert features == [0.0, 0.0, 0.0, 0.0, 0.0, 2000.0]

    def test_encoding_length(self, model):
        """Feature vector should be len(KNOWN_CONTENT_TYPES) + 1."""
        features = model._encode("debate", 1000)
        assert len(features) == len(KNOWN_CONTENT_TYPES) + 1


# ═══════════════════════════════════════════════════════════════════════════
# FIT
# ═══════════════════════════════════════════════════════════════════════════

class TestFit:
    """Verify training behavior."""

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_fit_sets_is_fitted(self, tmp_path):
        """After fit() with enough data, is_fitted should be True."""
        m = WPMModel()
        records = _make_records(n=10)
        with patch("vox_romana.data.wpm_model.DEFAULT_MODEL_PATH", tmp_path / "wpm_model.pkl"):
            m.fit(records)
        assert m.is_fitted is True
        assert m.n_samples == 10

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_fit_insufficient_data(self, model):
        """fit() with fewer than MIN_OBSERVATIONS should not train."""
        records = _make_records(n=MIN_OBSERVATIONS - 1)
        model.fit(records)
        assert model.is_fitted is False
        assert model.n_samples == 0

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_fit_expands_content_types(self, tmp_path):
        """fit() should add new content types to the encoding."""
        m = WPMModel()
        records = _make_records(n=10, content_type="debate")
        # Add a few records with a new content type.
        records.extend(_make_records(n=3, content_type="interview"))
        with patch("vox_romana.data.wpm_model.DEFAULT_MODEL_PATH", tmp_path / "wpm_model.pkl"):
            m.fit(records)
        assert "interview" in m._content_types

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_fit_saves_model_file(self, tmp_path):
        """fit() should persist the model to disk."""
        m = WPMModel()
        records = _make_records(n=10)
        model_path = tmp_path / "wpm_model.pkl"
        with patch("vox_romana.data.wpm_model.DEFAULT_MODEL_PATH", model_path):
            m.fit(records)
        assert model_path.exists()


# ═══════════════════════════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════════════════════════

class TestPredict:
    """Verify prediction behavior."""

    def test_predict_before_fit_returns_none(self, model):
        """Predicting on an unfitted model should return None."""
        result = model.predict("debate", 4500)
        assert result is None

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_predict_returns_reasonable_wpm(self, fitted_model):
        """Prediction should be within a reasonable WPM range."""
        wpm = fitted_model.predict("debate", 4500)
        assert wpm is not None
        # Trained on ~150 WPM data, prediction should be in the ballpark.
        assert 100 < wpm < 200

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_predict_clamps_to_range(self, tmp_path):
        """Extreme inputs shouldn't produce WPM outside [80, 300]."""
        m = WPMModel()
        # Train on extreme data to provoke out-of-range predictions.
        records = [
            {"content_type": "debate", "word_count": wc, "wpm": 50.0}
            for wc in range(100, 700, 60)
        ]
        with patch("vox_romana.data.wpm_model.DEFAULT_MODEL_PATH", tmp_path / "wpm_model.pkl"):
            m.fit(records)
        result = m.predict("debate", 100)
        assert result is not None
        assert result >= 80.0


# ═══════════════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════

class TestPersistence:
    """Verify model save/load round-trip."""

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_save_load_roundtrip(self, tmp_path):
        """A saved model should load back with the same predictions."""
        model_path = tmp_path / "wpm_model.pkl"

        # Train and save.
        m1 = WPMModel()
        records = _make_records(n=15, content_type="debate", base_wpm=150.0)
        with patch("vox_romana.data.wpm_model.DEFAULT_MODEL_PATH", model_path):
            m1.fit(records)

        # Load from disk.
        with patch("vox_romana.data.wpm_model.DEFAULT_MODEL_PATH", model_path):
            m2 = WPMModel.load()

        assert m2 is not None
        assert m2.is_fitted is True
        assert m2.n_samples == m1.n_samples

        # Predictions should match.
        p1 = m1.predict("debate", 4500)
        p2 = m2.predict("debate", 4500)
        assert p1 is not None and p2 is not None
        assert abs(p1 - p2) < 0.01

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading from a path that doesn't exist should return None."""
        with patch("vox_romana.data.wpm_model.DEFAULT_MODEL_PATH", tmp_path / "nope.pkl"):
            result = WPMModel.load()
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# CATALOG INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

class TestCatalogIntegration:
    """Verify update_from_catalog() behavior."""

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_update_from_catalog(self, mock_catalog, tmp_path):
        """update_from_catalog() should train the model from catalog data."""
        m = WPMModel()
        with patch("vox_romana.data.wpm_model.DEFAULT_MODEL_PATH", tmp_path / "wpm_model.pkl"):
            m.update_from_catalog(mock_catalog)
        assert m.is_fitted is True
        assert m.n_samples == 24  # 8 records × 3 content types

    def test_update_from_empty_catalog(self):
        """Empty catalog should leave model unfitted."""
        cat = MagicMock()
        cat.get_wpm_stats_all.return_value = []
        m = WPMModel()
        m.update_from_catalog(cat)
        assert m.is_fitted is False


# ═══════════════════════════════════════════════════════════════════════════
# GRACEFUL DEGRADATION (no sklearn)
# ═══════════════════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    """Verify behavior when sklearn is NOT installed."""

    def test_fit_noop_without_sklearn(self, model):
        """fit() should be a no-op returning self when sklearn is absent."""
        records = _make_records(n=20)
        with patch("vox_romana.data.wpm_model._HAS_SKLEARN", False):
            result = model.fit(records)
        assert result is model
        assert model.is_fitted is False

    def test_predict_none_without_sklearn(self, model):
        """predict() returns None when sklearn is absent."""
        with patch("vox_romana.data.wpm_model._HAS_SKLEARN", False):
            result = model.predict("debate", 4500)
        assert result is None

    def test_load_none_without_sklearn(self, tmp_path):
        """load() returns None when sklearn is absent."""
        with patch("vox_romana.data.wpm_model._HAS_SKLEARN", False):
            result = WPMModel.load(tmp_path / "model.pkl")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# REPR
# ═══════════════════════════════════════════════════════════════════════════

class TestRepr:
    """Verify __repr__ output."""

    def test_repr_unfitted(self, model):
        assert "not fitted" in repr(model)
        assert "n=0" in repr(model)

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
    def test_repr_fitted(self, fitted_model):
        assert "fitted" in repr(fitted_model)
        assert "n=15" in repr(fitted_model)
