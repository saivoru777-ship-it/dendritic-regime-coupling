"""Structural regime classification based on electrotonic length.

Primary classification uses tertile-based thresholds on the observed L/lambda
distribution. In this MICrONS cortical EM dataset, electrotonic lengths range
from ~0.001 to ~0.5, with a median of ~0.06 — well below textbook thresholds
(0.1, 0.5) that assume idealized passive cable parameters.

Tertile thresholds produce balanced groups and the upper threshold (~0.12)
falls near the literature compact/non-compact boundary (Rall 1967).

Sensitivity analysis sweeps across quantile thresholds and includes the
biophysical (0.1, 0.5) case to confirm effect size robustness.

Branch order serves as a robustness check only.
K-means classification is supplementary.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score


# Data-grounded tertile thresholds (from MICrONS L/lambda distribution)
# These correspond to 33rd/67th percentiles of observed electrotonic lengths.
# The upper threshold (~0.12) aligns with the classic compact/non-compact
# boundary; the lower threshold (~0.03) separates ultracompact proximal
# segments from the intermediate range.
DEFAULT_THRESHOLD_LOW = None   # Computed from data tertiles
DEFAULT_THRESHOLD_HIGH = None  # Computed from data tertiles

# Biophysical reference thresholds (used in sensitivity sweep)
BIOPHYSICAL_THRESHOLD_LOW = 0.1
BIOPHYSICAL_THRESHOLD_HIGH = 0.5

REGIME_NAMES = {
    0: "summation-like",
    1: "nonlinear-prone",
    2: "compartmentalized",
}


def compute_tertile_thresholds(e_lengths):
    """Compute tertile thresholds from observed electrotonic length distribution.

    Returns (threshold_low, threshold_high) at 33rd and 67th percentiles.
    """
    e = np.asarray(e_lengths, dtype=float)
    e = e[~np.isnan(e) & ~np.isinf(e)]
    return float(np.percentile(e, 33.3)), float(np.percentile(e, 66.7))


def classify_by_electrotonic_length(e_lengths, threshold_low=DEFAULT_THRESHOLD_LOW,
                                     threshold_high=DEFAULT_THRESHOLD_HIGH):
    """Classify branches into structural regimes by electrotonic length.

    Parameters
    ----------
    e_lengths : array-like
        Electrotonic length (L/lambda) per branch.
    threshold_low : float or None
        Below this → summation-like. If None, uses 33rd percentile.
    threshold_high : float or None
        At or above this → compartmentalized. If None, uses 67th percentile.

    Returns
    -------
    regimes : np.ndarray of int
        0=summation-like, 1=nonlinear-prone, 2=compartmentalized.
        NaN electrotonic lengths → -1.
    """
    e_lengths = np.asarray(e_lengths, dtype=float)

    # Compute tertile thresholds from data if not provided
    if threshold_low is None or threshold_high is None:
        t_low, t_high = compute_tertile_thresholds(e_lengths)
        if threshold_low is None:
            threshold_low = t_low
        if threshold_high is None:
            threshold_high = t_high

    regimes = np.full(len(e_lengths), -1, dtype=int)
    valid = ~np.isnan(e_lengths)
    regimes[valid & (e_lengths < threshold_low)] = 0
    regimes[valid & (e_lengths >= threshold_low) & (e_lengths < threshold_high)] = 1
    regimes[valid & (e_lengths >= threshold_high)] = 2
    return regimes


def classify_by_branch_order(orders, n_regimes=3):
    """Classify branches by branch order using quantile boundaries.

    Used only as a robustness check against electrotonic classification.

    Returns
    -------
    regimes : np.ndarray of int (0, 1, 2)
    """
    orders = np.asarray(orders, dtype=float)
    valid = ~np.isnan(orders) & (orders >= 0)
    regimes = np.full(len(orders), -1, dtype=int)

    valid_orders = orders[valid]
    if len(valid_orders) == 0:
        return regimes

    q_low = np.percentile(valid_orders, 33.3)
    q_high = np.percentile(valid_orders, 66.7)

    regimes[valid & (orders <= q_low)] = 0
    regimes[valid & (orders > q_low) & (orders <= q_high)] = 1
    regimes[valid & (orders > q_high)] = 2
    return regimes


def classify_by_kmeans(features_df, feature_cols=None, k=3, seed=42):
    """K-means classification on branch features (supplementary).

    Parameters
    ----------
    features_df : DataFrame
        Branch features.
    feature_cols : list of str
        Columns to cluster on. Defaults to electrotonic_length, mean_diameter_nm, soma_distance_nm.
    k : int
        Number of clusters.
    seed : int

    Returns
    -------
    labels : np.ndarray of int
        Cluster labels (0 to k-1), -1 for invalid rows.
    """
    if feature_cols is None:
        feature_cols = ["electrotonic_length", "mean_diameter_nm", "soma_distance_nm"]

    X = features_df[feature_cols].values.copy()
    valid = ~np.any(np.isnan(X), axis=1) & ~np.any(np.isinf(X), axis=1)

    labels = np.full(len(X), -1, dtype=int)
    if valid.sum() < k:
        return labels

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[valid])

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels[valid] = km.fit_predict(X_scaled)

    # Sort clusters by mean electrotonic length (0=shortest, k-1=longest)
    e_col_idx = feature_cols.index("electrotonic_length") if "electrotonic_length" in feature_cols else 0
    cluster_means = []
    for c in range(k):
        mask = labels == c
        cluster_means.append(X[mask, e_col_idx].mean() if mask.any() else np.inf)

    sort_order = np.argsort(cluster_means)
    remap = {sort_order[i]: i for i in range(k)}
    labels[valid] = np.array([remap[l] for l in labels[valid]])

    return labels


def check_regime_agreement(electrotonic_regimes, order_regimes):
    """Compute Cohen's kappa between electrotonic and order-based classifications.

    Parameters
    ----------
    electrotonic_regimes : array-like
    order_regimes : array-like

    Returns
    -------
    dict with 'kappa', 'agreement_fraction', 'n_valid'
    """
    e = np.asarray(electrotonic_regimes)
    o = np.asarray(order_regimes)

    valid = (e >= 0) & (o >= 0)
    if valid.sum() < 10:
        return {"kappa": np.nan, "agreement_fraction": np.nan, "n_valid": valid.sum()}

    e_valid = e[valid]
    o_valid = o[valid]
    kappa = cohen_kappa_score(e_valid, o_valid)
    agreement = np.mean(e_valid == o_valid)

    return {
        "kappa": kappa,
        "agreement_fraction": agreement,
        "n_valid": int(valid.sum()),
    }


def classify_branches(features_df, threshold_low=DEFAULT_THRESHOLD_LOW,
                      threshold_high=DEFAULT_THRESHOLD_HIGH):
    """Add regime classifications to branch features DataFrame.

    Adds columns: regime, regime_name, order_regime, kmeans_regime, is_bpc,
    and stores thresholds used as DataFrame attributes.

    Parameters
    ----------
    features_df : DataFrame
        Output of compute_all_branch_features().

    Returns
    -------
    DataFrame with added columns.
    """
    df = features_df.copy()

    # Resolve thresholds (compute from data if None)
    if threshold_low is None or threshold_high is None:
        t_low, t_high = compute_tertile_thresholds(df["electrotonic_length"].values)
        if threshold_low is None:
            threshold_low = t_low
        if threshold_high is None:
            threshold_high = t_high

    # Store thresholds used
    df.attrs["threshold_low"] = threshold_low
    df.attrs["threshold_high"] = threshold_high

    # Primary: electrotonic length
    df["regime"] = classify_by_electrotonic_length(
        df["electrotonic_length"].values, threshold_low, threshold_high
    )
    df["regime_name"] = df["regime"].map(REGIME_NAMES).fillna("unclassified")

    # Robustness check: branch order
    df["order_regime"] = classify_by_branch_order(df["branch_order"].values)

    # Supplementary: k-means
    df["kmeans_regime"] = classify_by_kmeans(df)

    # BPC flag
    df["is_bpc"] = df["subtype"].str.upper() == "BPC"

    return df


def regime_size_table(df):
    """Summary table of regime sizes."""
    valid = df["regime"] >= 0
    counts = df.loc[valid, "regime_name"].value_counts()
    fracs = counts / counts.sum()
    return pd.DataFrame({"count": counts, "fraction": fracs})


def quantile_threshold_sweep(e_lengths, quantiles=None):
    """Generate thresholds from quantiles for sensitivity analysis.

    Parameters
    ----------
    e_lengths : array-like
        Valid electrotonic lengths.
    quantiles : list of (q_low, q_high) tuples

    Returns
    -------
    List of (threshold_low, threshold_high) tuples.
    """
    e = np.asarray(e_lengths, dtype=float)
    e = e[~np.isnan(e) & ~np.isinf(e)]

    if quantiles is None:
        quantiles = [
            (0.20, 0.60), (0.25, 0.65), (0.30, 0.70),
            (0.33, 0.67), (0.35, 0.75), (0.40, 0.80),
        ]

    thresholds = []
    for q_lo, q_hi in quantiles:
        t_lo = np.quantile(e, q_lo)
        t_hi = np.quantile(e, q_hi)
        thresholds.append((t_lo, t_hi))

    return thresholds
