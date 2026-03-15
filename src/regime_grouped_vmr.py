"""Regime-grouped VMR: pool branches by structural regime and compute VMR curves.

For each regime:
1. Collect all branches assigned to that regime
2. Create filtered SnapResult (only synapses on those branches)
3. Compute VMR curves via compute_curves()
4. Generate regime-specific null envelope
"""

import numpy as np
import pandas as pd

from neurostat.io.swc import SnapResult
from neurostat.core.tree_statistics import compute_curves, ScaleRange
from neurostat.core.null_models import DendriteConstrainedNull

from .structural_regimes import REGIME_NAMES


def filter_snap_by_regime(snap_result, branches, regime_labels, target_regime):
    """Filter SnapResult to only include synapses on branches of given regime.

    Parameters
    ----------
    snap_result : SnapResult
    branches : list of Branch
    regime_labels : array-like
        Regime label per branch (0, 1, 2, or -1).
    target_regime : int

    Returns
    -------
    filtered_snap : SnapResult
        Only synapses on branches in target_regime.
    filtered_branches : list of Branch
        Branches in target_regime.
    branch_remap : dict
        Old branch_idx -> new branch_idx in filtered_branches.
    """
    regime_labels = np.asarray(regime_labels)
    target_mask = regime_labels == target_regime
    target_branch_indices = np.where(target_mask)[0]

    if len(target_branch_indices) == 0:
        return None, [], {}

    # Remap branch indices
    branch_remap = {old: new for new, old in enumerate(target_branch_indices)}
    filtered_branches = [branches[i] for i in target_branch_indices]

    # Filter synapses
    synapse_mask = np.isin(snap_result.branch_ids, target_branch_indices)
    if not synapse_mask.any():
        return None, filtered_branches, branch_remap

    new_branch_ids = np.array([branch_remap[bid] for bid in snap_result.branch_ids[synapse_mask]])

    filtered_snap = SnapResult(
        branch_ids=new_branch_ids,
        branch_positions=snap_result.branch_positions[synapse_mask],
        distances=snap_result.distances[synapse_mask],
        valid=np.ones(synapse_mask.sum(), dtype=bool),
    )

    return filtered_snap, filtered_branches, branch_remap


def compute_regime_vmr_curves(snap_result, branches, regime_labels, n_scales=12):
    """Compute VMR curves for each structural regime.

    Parameters
    ----------
    snap_result : SnapResult
    branches : list of Branch
    regime_labels : array-like
        Regime per branch.
    n_scales : int

    Returns
    -------
    dict mapping regime_int -> dict with 'curves', 'n_branches', 'n_synapses', 'regime_name'.
    """
    results = {}
    for regime in [0, 1, 2]:
        filt_snap, filt_branches, _ = filter_snap_by_regime(
            snap_result, branches, regime_labels, regime
        )

        if filt_snap is None or len(filt_branches) < 3:
            results[regime] = {
                "curves": None,
                "n_branches": 0,
                "n_synapses": 0,
                "regime_name": REGIME_NAMES.get(regime, "unknown"),
            }
            continue

        scales = ScaleRange.for_dendrite(filt_branches, n_scales=n_scales)
        curves = compute_curves(filt_snap, filt_branches, scales)

        results[regime] = {
            "curves": curves,
            "n_branches": len(filt_branches),
            "n_synapses": len(filt_snap.branch_ids),
            "regime_name": REGIME_NAMES.get(regime, "unknown"),
            "scales": scales,
        }

    return results


def compute_regime_null_envelopes(skeleton, branches, regime_labels,
                                    n_mocks=200, n_scales=12, seed=42):
    """Generate null VMR envelopes for each regime.

    Uses DendriteConstrainedNull restricted to branches of each regime.

    Parameters
    ----------
    skeleton : NeuronSkeleton
    branches : list of Branch
    regime_labels : array-like
    n_mocks : int
    n_scales : int
    seed : int

    Returns
    -------
    dict mapping regime -> dict with 'mock_curves', 'median', 'ci_low', 'ci_high'.
    """
    results = {}

    for regime in [0, 1, 2]:
        target_mask = np.asarray(regime_labels) == regime
        target_indices = np.where(target_mask)[0]

        if len(target_indices) < 3:
            results[regime] = None
            continue

        filt_branches = [branches[i] for i in target_indices]
        branch_remap = {old: new for new, old in enumerate(target_indices)}

        # Compute scales for this regime's branches
        scales = ScaleRange.for_dendrite(filt_branches, n_scales=n_scales)

        # Generate null samples using uniform placement on these branches only
        rng = np.random.default_rng(seed + regime)
        branch_lengths = np.array([br.total_length for br in filt_branches])
        branch_probs = branch_lengths / branch_lengths.sum()

        # Estimate n_synapses as typical count on these branches
        # (will be set per-neuron in the actual pipeline)
        total_length = branch_lengths.sum()

        mock_vmr_matrix = []
        for m in range(n_mocks):
            # Sample n_points proportional to total length ratio
            # Use a representative count (will be overridden in pipeline)
            n_points = max(20, int(total_length / 10000))
            chosen_branches = rng.choice(len(filt_branches), size=n_points, p=branch_probs)
            positions = np.array([rng.uniform(0, filt_branches[b].total_length) for b in chosen_branches])

            mock_snap = SnapResult(
                branch_ids=chosen_branches.astype(int),
                branch_positions=positions,
                distances=np.zeros(n_points),
                valid=np.ones(n_points, dtype=bool),
            )

            curves = compute_curves(mock_snap, filt_branches, scales)
            if curves and "variance_values" in curves:
                mock_vmr_matrix.append(curves["variance_values"])

        if len(mock_vmr_matrix) < 10:
            results[regime] = None
            continue

        # Align to same length
        min_len = min(len(v) for v in mock_vmr_matrix)
        mock_matrix = np.array([v[:min_len] for v in mock_vmr_matrix])

        results[regime] = {
            "scales": scales[:min_len] if len(scales) > min_len else scales,
            "median": np.nanmedian(mock_matrix, axis=0),
            "ci_low": np.nanpercentile(mock_matrix, 2.5, axis=0),
            "ci_high": np.nanpercentile(mock_matrix, 97.5, axis=0),
            "n_mocks": len(mock_vmr_matrix),
        }

    return results


def compute_regime_vmr_per_neuron(snap_result, skeleton, regime_labels, n_scales=12,
                                    n_mocks=200, seed=42):
    """Full regime-grouped VMR for a single neuron.

    Returns dict with curves and null envelopes per regime.
    """
    branches = skeleton.branches
    curves = compute_regime_vmr_curves(snap_result, branches, regime_labels, n_scales)
    envelopes = compute_regime_null_envelopes(
        skeleton, branches, regime_labels, n_mocks=n_mocks, n_scales=n_scales, seed=seed
    )

    return {"curves": curves, "envelopes": envelopes}
