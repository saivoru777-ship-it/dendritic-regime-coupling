"""Per-branch uniform null model for spatial organization metrics.

For each branch with k synapses on length L:
1. Generate n_draws of k positions from Uniform(0, L)
2. Compute all three metrics for each draw → null distributions
3. Z-score = (observed - null_mean) / null_std

Controls for branch geometry (length, synapse count) determining expected metric values.
"""

import numpy as np
from .spatial_organization import clark_evans_1d, interval_cv, pairwise_compactness


def branch_null_distribution(branch_length, n_synapses, n_draws=1000, seed=None,
                              min_points_ce=5, min_points_cv=3, min_points_pw=3):
    """Generate null distributions of spatial metrics for a single branch.

    Parameters
    ----------
    branch_length : float
        Total length of the branch.
    n_synapses : int
        Number of synapses on this branch.
    n_draws : int
        Number of null realizations.
    seed : int or None
    min_points_ce, min_points_cv, min_points_pw : int
        Minimum synapse counts for each metric.

    Returns
    -------
    dict with keys 'clark_evans', 'interval_cv', 'pairwise_compactness',
    each a np.ndarray of shape (n_draws,).
    """
    rng = np.random.default_rng(seed)

    null_ce = np.full(n_draws, np.nan)
    null_cv = np.full(n_draws, np.nan)
    null_pw = np.full(n_draws, np.nan)

    for d in range(n_draws):
        pos = rng.uniform(0, branch_length, size=n_synapses)
        null_ce[d] = clark_evans_1d(pos, branch_length, min_points=min_points_ce)
        null_cv[d] = interval_cv(pos, min_points=min_points_cv)
        null_pw[d] = pairwise_compactness(pos, branch_length, min_points=min_points_pw)

    return {
        "clark_evans": null_ce,
        "interval_cv": null_cv,
        "pairwise_compactness": null_pw,
    }


def compute_z_score(observed, null_dist):
    """Compute z-score of observed value against null distribution.

    Returns np.nan if null_std is 0 or too few valid null samples.
    """
    valid = ~np.isnan(null_dist)
    if valid.sum() < 10 or np.isnan(observed):
        return np.nan

    null_mean = np.mean(null_dist[valid])
    null_std = np.std(null_dist[valid], ddof=1)

    if null_std < 1e-12:
        return np.nan

    return (observed - null_mean) / null_std


def compute_branch_z_scores(snap_result, branches, n_draws=1000, seed=42,
                             min_points_ce=5, min_points_cv=3, min_points_pw=3):
    """Compute z-scored spatial metrics for all branches.

    Parameters
    ----------
    snap_result : SnapResult
    branches : list of Branch
    n_draws : int
        Number of null draws per branch.
    seed : int
    min_points_ce, min_points_cv, min_points_pw : int

    Returns
    -------
    dict with keys 'clark_evans_z', 'interval_cv_z', 'pairwise_compactness_z',
    each a np.ndarray of shape (n_branches,).
    Also includes 'clark_evans_raw', 'interval_cv_raw', 'pairwise_compactness_raw'
    for the observed values.
    """
    from .spatial_organization import compute_branch_spatial_metrics

    n_branches = len(branches)

    # Observed metrics
    observed = compute_branch_spatial_metrics(
        snap_result, branches,
        min_points_ce=min_points_ce,
        min_points_cv=min_points_cv,
        min_points_pw=min_points_pw,
    )

    ce_z = np.full(n_branches, np.nan)
    cv_z = np.full(n_branches, np.nan)
    pw_z = np.full(n_branches, np.nan)

    rng = np.random.default_rng(seed)

    for i, br in enumerate(branches):
        mask = snap_result.branch_ids == i
        n_syn = mask.sum()
        if n_syn < 2:
            continue

        # Per-branch seed for reproducibility
        branch_seed = rng.integers(0, 2**31)
        null_dists = branch_null_distribution(
            br.total_length, n_syn, n_draws=n_draws, seed=branch_seed,
            min_points_ce=min_points_ce,
            min_points_cv=min_points_cv,
            min_points_pw=min_points_pw,
        )

        ce_z[i] = compute_z_score(observed["clark_evans"][i], null_dists["clark_evans"])
        cv_z[i] = compute_z_score(observed["interval_cv"][i], null_dists["interval_cv"])
        pw_z[i] = compute_z_score(observed["pairwise_compactness"][i], null_dists["pairwise_compactness"])

    return {
        "clark_evans_raw": observed["clark_evans"],
        "interval_cv_raw": observed["interval_cv"],
        "pairwise_compactness_raw": observed["pairwise_compactness"],
        "clark_evans_z": ce_z,
        "interval_cv_z": cv_z,
        "pairwise_compactness_z": pw_z,
    }
