"""Tests for spatial organization metrics and branch null model.

Tests:
- Clark-Evans: clustered → R≈0, equally-spaced → R>1, <5 points → NaN
- Interval CV: equal-spacing → CV≈0, clustered → CV>1
- Geodesic compactness: all-same-point → max compactness
- Branch null z-scores: approximately N(0,1) under uniform placement
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_organization import clark_evans_1d, interval_cv, pairwise_compactness
from src.branch_null import branch_null_distribution, compute_z_score


class TestClarkEvans:
    def test_equally_spaced(self):
        """Equally spaced points → R > 1 (more regular than random)."""
        positions = np.linspace(10, 90, 10)  # evenly spaced on [0, 100]
        R = clark_evans_1d(positions, 100.0)
        assert R > 1.0, f"Equally-spaced should be regular (R>1), got {R}"

    def test_highly_clustered(self):
        """All points near same location → R ≈ 0."""
        positions = np.array([50.0, 50.01, 50.02, 50.03, 50.04])
        R = clark_evans_1d(positions, 100.0)
        assert R < 0.3, f"Clustered should have small R, got {R}"

    def test_minimum_points(self):
        """Fewer than 5 points → NaN."""
        positions = np.array([10.0, 20.0, 30.0, 40.0])
        R = clark_evans_1d(positions, 100.0, min_points=5)
        assert np.isnan(R)

    def test_exactly_minimum(self):
        """Exactly 5 points → should compute."""
        positions = np.linspace(10, 90, 5)
        R = clark_evans_1d(positions, 100.0, min_points=5)
        assert not np.isnan(R)

    def test_zero_length(self):
        positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        R = clark_evans_1d(positions, 0.0)
        assert np.isnan(R)


class TestIntervalCV:
    def test_equal_spacing(self):
        """Equal spacing → CV ≈ 0."""
        positions = np.linspace(0, 100, 11)  # 10 equal gaps
        cv = interval_cv(positions)
        assert cv < 0.01, f"Equal spacing should give CV≈0, got {cv}"

    def test_clustered(self):
        """Clustered points → CV > 1."""
        # 8 points tightly clustered, 2 far away
        positions = np.array([10, 10.1, 10.2, 10.3, 10.4, 10.5, 80, 90])
        cv = interval_cv(positions)
        assert cv > 1.0, f"Clustered should give CV>1, got {cv}"

    def test_minimum_points(self):
        """Fewer than 3 points → NaN."""
        positions = np.array([10.0, 50.0])
        cv = interval_cv(positions, min_points=3)
        assert np.isnan(cv)

    def test_single_point(self):
        positions = np.array([50.0])
        cv = interval_cv(positions)
        assert np.isnan(cv)


class TestPairwiseCompactness:
    def test_all_same_point(self):
        """All at same location → max compactness (≈0)."""
        positions = np.array([50.0, 50.0, 50.0, 50.0])
        pw = pairwise_compactness(positions, 100.0)
        assert pw == pytest.approx(0.0, abs=1e-10)

    def test_endpoints(self):
        """Points at 0 and L → compactness = 1.0."""
        positions = np.array([0.0, 100.0, 0.0])
        pw = pairwise_compactness(positions, 100.0)
        # Mean of |0-100|, |0-0|, |100-0| / 100 = (100 + 0 + 100) / 3 / 100 = 0.667
        expected = (100 + 0 + 100) / 3.0 / 100.0
        assert pw == pytest.approx(expected, rel=1e-4)

    def test_minimum_points(self):
        positions = np.array([10.0, 50.0])
        pw = pairwise_compactness(positions, 100.0, min_points=3)
        assert np.isnan(pw)

    def test_compactness_range(self):
        """Compactness should be in [0, 1] for positions in [0, L]."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(5, 50)
            L = rng.uniform(1000, 100000)
            positions = rng.uniform(0, L, size=n)
            pw = pairwise_compactness(positions, L)
            assert 0 <= pw <= 1.0, f"Compactness {pw} outside [0,1]"


class TestBranchNull:
    def test_null_distribution_shape(self):
        null = branch_null_distribution(10000.0, 10, n_draws=100, seed=42)
        assert null["clark_evans"].shape == (100,)
        assert null["interval_cv"].shape == (100,)
        assert null["pairwise_compactness"].shape == (100,)

    def test_null_ce_near_one(self):
        """Under uniform placement, Clark-Evans R should be near 1.0."""
        null = branch_null_distribution(50000.0, 20, n_draws=500, seed=42)
        valid = null["clark_evans"][~np.isnan(null["clark_evans"])]
        assert len(valid) > 400
        mean_R = np.mean(valid)
        assert 0.8 < mean_R < 1.2, f"Mean null R should be near 1, got {mean_R}"

    def test_z_score_approximately_standard(self):
        """Z-scores of uniform samples should be approximately N(0,1)."""
        rng = np.random.default_rng(123)
        z_scores = []

        for trial in range(100):
            L = 50000.0
            n_syn = 15
            positions = rng.uniform(0, L, size=n_syn)
            observed_cv = interval_cv(positions)

            null = branch_null_distribution(L, n_syn, n_draws=200, seed=trial)
            z = compute_z_score(observed_cv, null["interval_cv"])
            if not np.isnan(z):
                z_scores.append(z)

        z_arr = np.array(z_scores)
        assert len(z_arr) > 50, f"Too few valid z-scores: {len(z_arr)}"
        # Under null, mean should be near 0, std near 1
        assert abs(np.mean(z_arr)) < 0.5, f"Mean z = {np.mean(z_arr):.2f}, expected ~0"
        assert 0.5 < np.std(z_arr) < 2.0, f"Std z = {np.std(z_arr):.2f}, expected ~1"

    def test_z_score_nan_handling(self):
        """Z-score should be NaN if observed is NaN."""
        null = np.random.randn(100)
        assert np.isnan(compute_z_score(np.nan, null))

    def test_z_score_constant_null(self):
        """Z-score should be NaN if null has zero variance."""
        null = np.ones(100)
        assert np.isnan(compute_z_score(1.5, null))
