"""Coupling tests: does structural regime predict spatial organization?

Primary: Mixed-effects model with covariates and neuron random intercept.
Confound controls: within-order stratification, residual analysis, label shuffle.
Secondary: Kruskal-Wallis, permutation test, BH-FDR.
Sensitivity: threshold sweep for effect size stability.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from .structural_regimes import (
    classify_by_electrotonic_length,
    quantile_threshold_sweep,
    REGIME_NAMES,
)


def fit_mixed_model(df, metric_col, covariates=None):
    """Fit mixed-effects model: metric ~ C(regime) + covariates + (1|neuron).

    Parameters
    ----------
    df : DataFrame
        Must contain metric_col, 'regime', 'neuron_label', and covariate columns.
    metric_col : str
        Z-scored spatial metric column name.
    covariates : list of str or None
        Additional covariate column names.

    Returns
    -------
    dict with 'model', 'converged', 'regime_coefficients', 'regime_pvalues',
    'aic', 'bic', 'n_obs'.
    """
    work = df.dropna(subset=[metric_col, "regime"]).copy()
    work = work[work["regime"] >= 0].copy()

    if len(work) < 20:
        return {"converged": False, "n_obs": len(work)}

    # Build formula
    terms = ["C(regime)"]
    if covariates:
        for cov in covariates:
            if cov in work.columns:
                terms.append(cov)
    formula = f"{metric_col} ~ {' + '.join(terms)}"

    try:
        model = smf.mixedlm(formula, work, groups=work["neuron_label"])
        result = model.fit(reml=True, maxiter=200)

        # Extract regime coefficients
        regime_coefs = {}
        regime_pvals = {}
        for key in result.params.index:
            if "regime" in key.lower() or "C(regime)" in key:
                regime_coefs[key] = result.params[key]
                regime_pvals[key] = result.pvalues[key]

        return {
            "model": result,
            "converged": True,
            "formula": formula,
            "regime_coefficients": regime_coefs,
            "regime_pvalues": regime_pvals,
            "aic": result.aic,
            "bic": result.bic,
            "n_obs": len(work),
        }
    except Exception as e:
        return {"converged": False, "error": str(e), "n_obs": len(work)}


def kruskal_wallis_per_neuron(df, metric_col):
    """Kruskal-Wallis test of metric by regime, per neuron.

    Returns DataFrame with per-neuron H, p-value, and BH-corrected p-value.
    """
    results = []
    for label, grp in df.groupby("neuron_label"):
        valid = grp.dropna(subset=[metric_col])
        valid = valid[valid["regime"] >= 0]

        groups = [g[metric_col].values for _, g in valid.groupby("regime")]
        groups = [g for g in groups if len(g) >= 2]

        if len(groups) < 2:
            results.append({
                "neuron_label": label,
                "H": np.nan,
                "p_value": np.nan,
                "n_branches": len(valid),
            })
            continue

        H, p = stats.kruskal(*groups)
        results.append({
            "neuron_label": label,
            "H": H,
            "p_value": p,
            "n_branches": len(valid),
        })

    out = pd.DataFrame(results)
    valid_p = out["p_value"].dropna()
    if len(valid_p) > 0:
        _, p_adj, _, _ = multipletests(valid_p, method="fdr_bh")
        out.loc[valid_p.index, "p_bh"] = p_adj
    else:
        out["p_bh"] = np.nan
    return out


def permutation_test_within_neuron(df, metric_col, n_perms=10000, seed=42):
    """Permutation test: shuffle regime labels within each neuron.

    Test statistic: absolute difference in mean metric between regime 0 and regime 2.

    Returns
    -------
    dict with 'observed_stat', 'p_value', 'null_distribution'.
    """
    rng = np.random.default_rng(seed)
    work = df.dropna(subset=[metric_col]).copy()
    work = work[work["regime"].isin([0, 1, 2])].copy()

    def compute_stat(frame):
        means = frame.groupby("regime")[metric_col].mean()
        if 0 in means.index and 2 in means.index:
            return means[2] - means[0]
        return np.nan

    observed = compute_stat(work)
    if np.isnan(observed):
        return {"observed_stat": np.nan, "p_value": np.nan}

    null_stats = np.empty(n_perms)
    for p in range(n_perms):
        shuffled = work.copy()
        for _, grp in shuffled.groupby("neuron_label"):
            perm_idx = grp.index
            shuffled.loc[perm_idx, "regime"] = rng.permutation(shuffled.loc[perm_idx, "regime"].values)
        null_stats[p] = compute_stat(shuffled)

    valid_null = null_stats[~np.isnan(null_stats)]
    p_value = np.mean(np.abs(valid_null) >= np.abs(observed)) if len(valid_null) > 0 else np.nan

    return {
        "observed_stat": observed,
        "p_value": p_value,
        "null_distribution": null_stats,
    }


def within_order_stratified_test(df, metric_col):
    """Test coupling within each branch order stratum.

    For each branch order value, run Kruskal-Wallis on metric by regime.
    If coupling persists within strata, it's not an order artifact.

    Returns DataFrame with per-stratum results.
    """
    results = []
    work = df.dropna(subset=[metric_col, "regime", "branch_order"]).copy()
    work = work[work["regime"] >= 0]

    for order_val, stratum in work.groupby("branch_order"):
        groups = [g[metric_col].values for _, g in stratum.groupby("regime")]
        groups = [g for g in groups if len(g) >= 2]

        if len(groups) < 2:
            results.append({
                "branch_order": order_val,
                "H": np.nan,
                "p_value": np.nan,
                "n_branches": len(stratum),
                "n_regimes": len(groups),
            })
            continue

        H, p = stats.kruskal(*groups)
        results.append({
            "branch_order": order_val,
            "H": H,
            "p_value": p,
            "n_branches": len(stratum),
            "n_regimes": len(groups),
        })

    out = pd.DataFrame(results)
    valid_p = out["p_value"].dropna()
    if len(valid_p) > 0:
        _, p_adj, _, _ = multipletests(valid_p, method="fdr_bh")
        out.loc[valid_p.index, "p_bh"] = p_adj
    else:
        out["p_bh"] = np.nan
    return out


def residual_analysis(df, metric_col):
    """Regress metric on branch_order, test regime effect on residuals.

    Returns dict with residual OLS result and regime effect test.
    """
    work = df.dropna(subset=[metric_col, "branch_order", "regime"]).copy()
    work = work[work["regime"] >= 0]

    if len(work) < 20:
        return {"converged": False, "n_obs": len(work)}

    # Step 1: Regress metric on branch_order
    X = sm.add_constant(work["branch_order"].values)
    y = work[metric_col].values
    ols_order = sm.OLS(y, X).fit()
    residuals = ols_order.resid

    # Step 2: Kruskal-Wallis on residuals by regime
    work = work.copy()
    work["_residual"] = residuals

    groups = [g["_residual"].values for _, g in work.groupby("regime")]
    groups = [g for g in groups if len(g) >= 2]

    if len(groups) < 2:
        return {
            "converged": True,
            "order_r2": ols_order.rsquared,
            "residual_H": np.nan,
            "residual_p": np.nan,
            "n_obs": len(work),
        }

    H, p = stats.kruskal(*groups)

    return {
        "converged": True,
        "order_r2": ols_order.rsquared,
        "residual_H": H,
        "residual_p": p,
        "n_obs": len(work),
    }


def regime_shuffle_within_order(df, metric_col, n_perms=10000, seed=42):
    """Shuffle regime labels within branch order bins.

    Most rigorous confound control: preserves order distribution.

    Returns dict with observed_stat, p_value, null_distribution.
    """
    rng = np.random.default_rng(seed)
    work = df.dropna(subset=[metric_col, "regime", "branch_order"]).copy()
    work = work[work["regime"].isin([0, 1, 2])].copy()

    def compute_stat(frame):
        means = frame.groupby("regime")[metric_col].mean()
        if 0 in means.index and 2 in means.index:
            return means[2] - means[0]
        return np.nan

    observed = compute_stat(work)
    if np.isnan(observed):
        return {"observed_stat": np.nan, "p_value": np.nan}

    null_stats = np.empty(n_perms)
    for p in range(n_perms):
        shuffled = work.copy()
        for _, grp in shuffled.groupby("branch_order"):
            if len(grp) < 2:
                continue
            perm_idx = grp.index
            shuffled.loc[perm_idx, "regime"] = rng.permutation(shuffled.loc[perm_idx, "regime"].values)
        null_stats[p] = compute_stat(shuffled)

    valid_null = null_stats[~np.isnan(null_stats)]
    p_value = np.mean(np.abs(valid_null) >= np.abs(observed)) if len(valid_null) > 0 else np.nan

    return {
        "observed_stat": observed,
        "p_value": p_value,
        "null_distribution": null_stats,
    }


def sensitivity_sweep(df, metric_col, covariates=None):
    """Sweep electrotonic length thresholds and report effect size stability.

    Uses both biophysical (0.1, 0.5) and quantile-based thresholds.

    Returns DataFrame with threshold, regime sizes, and mixed-model coefficient.
    """
    e_valid = df["electrotonic_length"].dropna()
    e_valid = e_valid[~np.isinf(e_valid)]

    from .structural_regimes import compute_tertile_thresholds, BIOPHYSICAL_THRESHOLD_LOW, BIOPHYSICAL_THRESHOLD_HIGH
    # Primary (tertile) thresholds
    t_lo_primary, t_hi_primary = compute_tertile_thresholds(e_valid.values)
    threshold_sets = [(t_lo_primary, t_hi_primary, "primary_tertile")]
    # Biophysical reference
    threshold_sets.append((BIOPHYSICAL_THRESHOLD_LOW, BIOPHYSICAL_THRESHOLD_HIGH, "biophysical"))
    # Quantile sweeps
    for t_lo, t_hi in quantile_threshold_sweep(e_valid.values):
        threshold_sets.append((t_lo, t_hi, "quantile"))

    results = []
    for t_lo, t_hi, source in threshold_sets:
        work = df.copy()
        work["regime"] = classify_by_electrotonic_length(
            work["electrotonic_length"].values, t_lo, t_hi
        )

        # Count regimes
        regime_counts = work[work["regime"] >= 0]["regime"].value_counts().to_dict()

        # Fit mixed model
        mm = fit_mixed_model(work, metric_col, covariates=covariates)

        # Extract max absolute regime coefficient
        max_coef = np.nan
        if mm.get("converged"):
            coefs = mm.get("regime_coefficients", {})
            if coefs:
                max_coef = max(abs(v) for v in coefs.values())

        results.append({
            "threshold_low": t_lo,
            "threshold_high": t_hi,
            "source": source,
            "n_regime_0": regime_counts.get(0, 0),
            "n_regime_1": regime_counts.get(1, 0),
            "n_regime_2": regime_counts.get(2, 0),
            "max_abs_coefficient": max_coef,
            "converged": mm.get("converged", False),
        })

    return pd.DataFrame(results)


def run_all_coupling_tests(df, metric_col, covariates=None, n_perms=10000, seed=42):
    """Run complete coupling analysis for one spatial metric.

    Returns dict with all test results.
    """
    default_covariates = ["synapse_count", "total_length_nm", "exc_fraction"]
    if covariates is None:
        covariates = default_covariates

    return {
        "mixed_model": fit_mixed_model(df, metric_col, covariates=covariates),
        "kruskal_wallis": kruskal_wallis_per_neuron(df, metric_col),
        "permutation": permutation_test_within_neuron(df, metric_col, n_perms=n_perms, seed=seed),
        "within_order": within_order_stratified_test(df, metric_col),
        "residual": residual_analysis(df, metric_col),
        "order_shuffle": regime_shuffle_within_order(df, metric_col, n_perms=n_perms, seed=seed),
        "sensitivity": sensitivity_sweep(df, metric_col, covariates=covariates),
    }
