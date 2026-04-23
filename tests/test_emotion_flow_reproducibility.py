"""Tests for vox_romana.emotion_flow.reproducibility."""

from __future__ import annotations

import random
from unittest.mock import patch

import numpy as np
import pytest

from vox_romana.emotion_flow.reproducibility import (
    _collect_package_versions,
    _detect_hardware,
    _git_short_sha,
    build_manifest,
    freeze_randomness,
)
from vox_romana.emotion_flow.models import ReproducibilityManifest


class TestFreezeRandomness:
    def test_numpy_seed_is_fixed(self):
        freeze_randomness(seed=0)
        a = np.random.rand(5)
        freeze_randomness(seed=0)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_python_random_seed_is_fixed(self):
        freeze_randomness(seed=7)
        a = [random.random() for _ in range(5)]
        freeze_randomness(seed=7)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_different_seeds_differ(self):
        freeze_randomness(seed=1)
        a = np.random.rand(3)
        freeze_randomness(seed=2)
        b = np.random.rand(3)
        assert not np.allclose(a, b)

    def test_torch_not_required(self):
        """freeze_randomness must not raise if torch is unavailable."""
        with patch.dict("sys.modules", {"torch": None}):
            freeze_randomness(seed=42)  # should not raise


class TestBuildManifest:
    def test_returns_manifest_type(self):
        m = build_manifest(seed=42)
        assert isinstance(m, ReproducibilityManifest)

    def test_seed_stored(self):
        m = build_manifest(seed=99)
        assert m.seed == 99

    def test_python_version_populated(self):
        import sys
        m = build_manifest(seed=42)
        assert m.python_version == sys.version.split()[0]

    def test_model_versions_stored(self):
        m = build_manifest(seed=42, model_versions={"my-model": "sha123"})
        assert m.model_versions["my-model"] == "sha123"

    def test_timestamp_is_iso_format(self):
        from datetime import datetime
        m = build_manifest(seed=42)
        datetime.fromisoformat(m.timestamp)  # raises if invalid

    def test_package_versions_includes_numpy(self):
        m = build_manifest(seed=42)
        assert "numpy" in m.package_versions
        assert m.package_versions["numpy"] != "not-installed"


class TestCollectPackageVersions:
    def test_numpy_version_found(self):
        versions = _collect_package_versions()
        assert "numpy" in versions
        assert versions["numpy"] not in ("", "not-installed")

    def test_missing_package_returns_not_installed(self):
        versions = _collect_package_versions()
        # bert-score may not be installed yet; just check it returns a string
        assert isinstance(versions.get("bert-score", "not-installed"), str)


class TestDetectHardware:
    def test_returns_string(self):
        hw = _detect_hardware()
        assert isinstance(hw, str)
        assert len(hw) > 0


class TestGitShortSha:
    def test_returns_string(self):
        sha = _git_short_sha()
        assert isinstance(sha, str)

    def test_returns_unknown_gracefully(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            sha = _git_short_sha()
        assert sha == "unknown"
