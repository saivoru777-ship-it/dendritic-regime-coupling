"""Spatial organization metrics for synapses on individual branches.

Three independent metrics:
1. Clark-Evans ratio (1D nearest-neighbor)
2. Interval CV (inter-synapse gap variability)
3. Pairwise geodesic compactness (mean pairwise distance / branch length)
"""

import numpy as np


def clark_evans_1d(positions, branch_length, min_points=5):
    """Clark-Evans ratio for 1D point process on a branch.

    R = observed_mean_NN / expected_mean_NN
    R < 1 = clustered, R = 1 = random, R > 1 = regular.

    Parameters
    ----------
    positions : array-like
        Synapse positions along the branch (0 to branch_length).
    branch_length : float
        Total branch length.
    min_points : int
        Minimum number of points required.

    Returns
    -------
    float or np.nan
    """
    positions = np.sort(np.asarray(positions, dtype=float))
    n = len(positions)
    if n < min_points or branch_length <= 0:
        return np.nan

    # Observed: mean nearest-neighbor distance in 1D
    # For each point, NN is the closer of left and right neighbors
    nn_dists = np.empty(n)
    for i in range(n):
        left = positions[i] - positions[i - 1] if i > 0 else positions[i]  # distance to boundary
        right = positions[i + 1] - positions[i] if i < n - 1 else branch_length - positions[i]
        # NN is minimum of neighbor distances (not boundary distances for interior points)
        candidates = []
        if i > 0:
            candidates.append(positions[i] - positions[i - 1])
        if i < n - 1:
            candidates.append(positions[i + 1] - positions[i])
        # Edge points: also consider boundary distance
        if i == 0:
            candidates.append(positions[i])  # distance to left boundary
        if i == n - 1:
            candidates.append(branch_length - positions[i])  # distance to right boundary
        nn_dists[i] = min(candidates)

    observed_mean = np.mean(nn_dists)

    # Expected mean NN for 1D CSR with edge correction (Donnelly approximation)
    # E[NN] = L/(2n) + 0.0514 * L / n^2
    expected_mean = branch_length / (2 * n) + 0.0514 * branch_length / (n * n)

    if expected_mean <= 0:
        return np.nan

    return observed_mean / expected_mean


def interval_cv(positions, min_points=3):
    """Coefficient of variation of inter-synapse intervals.

    CV > 1 = clustered, CV ~ 1 = random (exponential), CV < 1 = regular.

    Parameters
    ----------
    positions : array-like
        Synapse positions along the branch.
    min_points : int
        Minimum number of points (need at least 2 intervals → 3 points).

    Returns
    -------
    float or np.nan
    """
    positions = np.sort(np.asarray(positions, dtype=float))
    n = len(positions)
    if n < min_points:
        return np.nan

    gaps = np.diff(positions)
    if len(gaps) < 2:
        return np.nan

    mean_gap = np.mean(gaps)
    if mean_gap <= 0:
        return np.nan

    return np.std(gaps, ddof=0) / mean_gap


def pairwise_compactness(positions, branch_length, min_points=3):
    """Mean pairwise distance normalized by branch length.

    Lower values = more compact/clustered.

    Parameters
    ----------
    positions : array-like
        Synapse positions along branch.
    branch_length : float
        Total branch length.
    min_points : int

    Returns
    -------
    float or np.nan
    """
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    if n < min_points or branch_length <= 0:
        return np.nan

    # Mean pairwise distance
    dists = np.abs(positions[:, None] - positions[None, :])
    # Upper triangle only
    triu_idx = np.triu_indices(n, k=1)
    mean_pw = np.mean(dists[triu_idx])

    return mean_pw / branch_length


def compute_branch_spatial_metrics(snap_result, branches, min_points_ce=5,
                                    min_points_cv=3, min_points_pw=3):
    """Compute all three spatial metrics for every branch.

    Parameters
    ----------
    snap_result : SnapResult
        Snapped synapses.
    branches : list of Branch
        Branch objects.
    min_points_ce : int
        Minimum synapses for Clark-Evans.
    min_points_cv : int
        Minimum synapses for interval CV.
    min_points_pw : int
        Minimum synapses for pairwise compactness.

    Returns
    -------
    dict with keys 'clark_evans', 'interval_cv', 'pairwise_compactness',
    each a np.ndarray of shape (n_branches,).
    """
    n_branches = len(branches)
    ce = np.full(n_branches, np.nan)
    cv = np.full(n_branches, np.nan)
    pw = np.full(n_branches, np.nan)

    for i, br in enumerate(branches):
        mask = snap_result.branch_ids == i
        if not mask.any():
            continue
        positions = snap_result.branch_positions[mask]
        bl = br.total_length

        ce[i] = clark_evans_1d(positions, bl, min_points=min_points_ce)
        cv[i] = interval_cv(positions, min_points=min_points_cv)
        pw[i] = pairwise_compactness(positions, bl, min_points=min_points_pw)

    return {
        "clark_evans": ce,
        "interval_cv": cv,
        "pairwise_compactness": pw,
    }
