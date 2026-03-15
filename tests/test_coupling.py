"""Tests for coupling analysis.

Tests:
- Mixed-effects model converges on synthetic data with known regime effect
- Permutation test: p-values are small when coupling exists
- Within-order stratification works when coupling exists within strata
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.coupling_tests import (
    fit_mixed_model,
    kruskal_wallis_per_neuron,
    permutation_test_within_neuron,
    within_order_stratified_test,
    residual_analysis,
    regime_shuffle_within_order,
)
from src.structural_regimes import (
    classify_by_electrotonic_length,
    classify_by_branch_order,
    check_regime_agreement,
    REGIME_NAMES,
)


def make_synthetic_data(n_neurons=6, n_branches_per=50, regime_effect=1.0, seed=42):
    """Create synthetic branch data with known regime-metric coupling.

    Regime 0 (summation): metric ~ N(0, 1)
    Regime 1 (nonlinear): metric ~ N(regime_effect/2, 1)
    Regime 2 (compartmentalized): metric ~ N(regime_effect, 1)
    """
    rng = np.random.default_rng(seed)
    records = []
    for n in range(n_neurons):
        label = f"neuron_{n}"
        for b in range(n_branches_per):
            regime = rng.choice([0, 1, 2])
            offset = regime * regime_effect / 2.0
            metric = rng.normal(offset, 1.0)
            records.append({
                "neuron_label": label,
                "branch_idx": b,
                "regime": regime,
                "regime_name": REGIME_NAMES[regime],
                "branch_order": regime + rng.integers(0, 3),  # correlated but noisy
                "synapse_count": rng.integers(5, 50),
                "total_length_nm": rng.uniform(5000, 50000),
                "exc_fraction": rng.uniform(0.3, 0.8),
                "electrotonic_length": [0.05, 0.3, 0.8][regime] + rng.normal(0, 0.02),
                "test_metric_z": metric,
            })
    return pd.DataFrame(records)


class TestMixedModel:
    def test_converges_with_effect(self):
        df = make_synthetic_data(regime_effect=2.0)
        result = fit_mixed_model(df, "test_metric_z",
                                  covariates=["synapse_count", "total_length_nm", "exc_fraction"])
        assert result["converged"], "Model should converge"
        assert result["n_obs"] > 100

    def test_detects_effect(self):
        df = make_synthetic_data(regime_effect=2.0)
        result = fit_mixed_model(df, "test_metric_z")
        assert result["converged"]
        # Should find significant regime effect
        coefs = result["regime_coefficients"]
        assert len(coefs) > 0, "Should have regime coefficients"
        max_coef = max(abs(v) for v in coefs.values())
        assert max_coef > 0.3, f"Regime coefficient should be substantial, got {max_coef}"

    def test_null_effect(self):
        df = make_synthetic_data(regime_effect=0.0)
        result = fit_mixed_model(df, "test_metric_z")
        assert result["converged"]
        # Coefficient should be near zero
        coefs = result["regime_coefficients"]
        if coefs:
            max_coef = max(abs(v) for v in coefs.values())
            assert max_coef < 1.0, "Under no effect, coefficient should be small"

    def test_too_few_observations(self):
        df = make_synthetic_data(n_neurons=1, n_branches_per=5, regime_effect=1.0)
        result = fit_mixed_model(df, "test_metric_z")
        # May or may not converge with very few data, but shouldn't crash
        assert "n_obs" in result


class TestKruskalWallis:
    def test_per_neuron(self):
        df = make_synthetic_data(regime_effect=2.0)
        result = kruskal_wallis_per_neuron(df, "test_metric_z")
        assert len(result) == 6, "Should have 6 neurons"
        assert "p_bh" in result.columns

    def test_detects_effect(self):
        df = make_synthetic_data(regime_effect=3.0, n_branches_per=100)
        result = kruskal_wallis_per_neuron(df, "test_metric_z")
        # At least some neurons should show significant effect
        sig_count = (result["p_bh"] < 0.05).sum()
        assert sig_count > 0, "Some neurons should show significant coupling"


class TestPermutation:
    def test_detects_coupling(self):
        df = make_synthetic_data(regime_effect=3.0, n_branches_per=100)
        result = permutation_test_within_neuron(df, "test_metric_z", n_perms=1000, seed=42)
        assert result["p_value"] < 0.05, f"Should detect strong coupling, got p={result['p_value']}"

    def test_null_is_non_significant(self):
        df = make_synthetic_data(regime_effect=0.0, n_branches_per=50)
        result = permutation_test_within_neuron(df, "test_metric_z", n_perms=1000, seed=42)
        # Under null, p should usually be > 0.01 (not always > 0.05 due to randomness)
        assert result["p_value"] > 0.001, f"Under null, should not be highly significant, got p={result['p_value']}"


class TestWithinOrderStratification:
    def test_produces_results(self):
        df = make_synthetic_data(regime_effect=2.0)
        result = within_order_stratified_test(df, "test_metric_z")
        assert len(result) > 0
        assert "branch_order" in result.columns
        assert "p_value" in result.columns


class TestResidualAnalysis:
    def test_converges(self):
        df = make_synthetic_data(regime_effect=2.0)
        result = residual_analysis(df, "test_metric_z")
        assert result["converged"]
        assert "order_r2" in result

    def test_detects_effect_after_residualization(self):
        df = make_synthetic_data(regime_effect=3.0, n_branches_per=100)
        result = residual_analysis(df, "test_metric_z")
        assert result["converged"]
        assert result["residual_p"] < 0.05, f"Effect should survive residualization, got p={result['residual_p']}"


class TestRegimeShuffleWithinOrder:
    def test_detects_coupling(self):
        df = make_synthetic_data(regime_effect=3.0, n_branches_per=100)
        result = regime_shuffle_within_order(df, "test_metric_z", n_perms=1000, seed=42)
        assert result["p_value"] < 0.05, f"Should detect coupling, got p={result['p_value']}"


class TestRegimeClassification:
    def test_electrotonic_thresholds_explicit(self):
        """Test with explicitly provided thresholds (biophysical reference)."""
        e_lengths = np.array([0.05, 0.08, 0.15, 0.3, 0.6, 1.0, np.nan])
        regimes = classify_by_electrotonic_length(e_lengths, threshold_low=0.1, threshold_high=0.5)
        assert regimes[0] == 0  # < 0.1
        assert regimes[1] == 0  # < 0.1
        assert regimes[2] == 1  # 0.1-0.5
        assert regimes[3] == 1  # 0.1-0.5
        assert regimes[4] == 2  # >= 0.5
        assert regimes[5] == 2  # >= 0.5
        assert regimes[6] == -1  # NaN

    def test_electrotonic_thresholds_tertile(self):
        """Test with tertile thresholds (auto-computed from data)."""
        e_lengths = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30])
        regimes = classify_by_electrotonic_length(e_lengths)  # None thresholds → tertiles
        # Should produce ~equal groups
        assert (regimes == 0).sum() == 3
        assert (regimes == 1).sum() == 3
        assert (regimes == 2).sum() == 3

    def test_agreement_perfect(self):
        e = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        o = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        result = check_regime_agreement(e, o)
        assert result["kappa"] == pytest.approx(1.0)
        assert result["agreement_fraction"] == pytest.approx(1.0)

    def test_agreement_random(self):
        rng = np.random.default_rng(42)
        e = rng.choice([0, 1, 2], size=100)
        o = rng.choice([0, 1, 2], size=100)
        result = check_regime_agreement(e, o)
        assert result["kappa"] < 0.3, "Random labels should have low kappa"
