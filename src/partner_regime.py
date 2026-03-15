"""Partner compactness ratios stratified by structural regime.

Maps existing partner-level compactness data from input_clustering_results.json
to branch regimes. Tests whether partners on compartmentalized branches show
more compactness than those on summation-like branches.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from .branch_morphometry import load_neuron, DATA_DIR, PARTNER_DIR, NEURONS


RESULTS_DIR = Path.home() / "research" / "neurostat-input-clustering" / "results"
CLUSTERING_RESULTS_PATH = RESULTS_DIR / "input_clustering_results.json"


def load_clustering_results():
    """Load input_clustering_results.json."""
    with open(CLUSTERING_RESULTS_PATH) as f:
        return json.load(f)


def map_partner_synapses_to_regimes(label, root_id, regime_df):
    """Map each partner's synapses to branch regimes.

    Parameters
    ----------
    label : str
        Neuron label.
    root_id : str
        Neuron root ID.
    regime_df : DataFrame
        Branch features with 'regime' column for this neuron.

    Returns
    -------
    DataFrame with columns: partner_id, n_synapses, dominant_regime, regime_fractions,
    spans_regimes, regime_counts.
    """
    partner_path = PARTNER_DIR / f"{label}_presynaptic.csv"
    if not partner_path.exists():
        return None

    result = load_neuron(label, root_id)
    if result is None:
        return None
    skeleton, snap_valid = result

    # Load partner data and match to synapses
    partner_df = pd.read_csv(partner_path)
    partner_coords = partner_df[["x_nm", "y_nm", "z_nm"]].values.astype(float)

    syn_path = DATA_DIR / f"{label}_{root_id}_synapses.csv"
    syn_df = pd.read_csv(syn_path)
    syn_coords_nm = syn_df[["x_um", "y_um", "z_um"]].values * 1000.0

    # Match all synapses to partners
    tree = cKDTree(partner_coords)
    _, indices = tree.query(syn_coords_nm)
    all_partner_ids = partner_df["pre_root_id"].values[indices]

    # Filter to valid synapses
    snap_full = skeleton.snap_points(syn_coords_nm, d_max=50000.0)
    valid_mask = snap_full.valid
    valid_partner_ids = all_partner_ids[valid_mask]
    valid_branch_ids = snap_valid.branch_ids

    # Build branch_id -> regime mapping
    branch_to_regime = {}
    neuron_regime = regime_df[regime_df["neuron_label"] == label]
    for _, row in neuron_regime.iterrows():
        branch_to_regime[int(row["branch_idx"])] = int(row["regime"])

    # Map each synapse to its regime
    synapse_regimes = np.array([branch_to_regime.get(bid, -1) for bid in valid_branch_ids])

    # Group by partner
    partner_records = []
    unique_partners = np.unique(valid_partner_ids)
    for pid in unique_partners:
        mask = valid_partner_ids == pid
        n_syn = mask.sum()
        if n_syn < 3:
            continue

        regimes = synapse_regimes[mask]
        valid_regimes = regimes[regimes >= 0]
        if len(valid_regimes) == 0:
            continue

        # Regime counts and dominant
        regime_counts = {0: 0, 1: 0, 2: 0}
        for r in valid_regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1

        dominant = max(regime_counts, key=regime_counts.get)
        total_valid = len(valid_regimes)
        regime_fracs = {r: c / total_valid for r, c in regime_counts.items()}

        # Does this partner span multiple regimes?
        unique_regimes = set(r for r in valid_regimes if r >= 0)
        spans = len(unique_regimes) > 1

        partner_records.append({
            "partner_id": pid,
            "neuron_label": label,
            "n_synapses": n_syn,
            "n_valid_regime": total_valid,
            "dominant_regime": dominant,
            "regime_0_frac": regime_fracs.get(0, 0),
            "regime_1_frac": regime_fracs.get(1, 0),
            "regime_2_frac": regime_fracs.get(2, 0),
            "spans_regimes": spans,
        })

    if not partner_records:
        return None
    return pd.DataFrame(partner_records)


def merge_with_clustering_results(partner_regime_df, clustering_results):
    """Merge partner regime assignments with existing clustering results.

    Adds effect_ratio, mean_pairwise_distance, p_distance, max_branch_fraction
    from input_clustering_results.json.

    Parameters
    ----------
    partner_regime_df : DataFrame
        Output of map_partner_synapses_to_regimes.
    clustering_results : list
        Parsed input_clustering_results.json.

    Returns
    -------
    DataFrame with merged columns.
    """
    # Build lookup: (label, partner_id) -> partner result
    lookup = {}
    for neuron_result in clustering_results:
        label = neuron_result["label"]
        for pr in neuron_result.get("partner_test", {}).get("partner_results", []):
            key = (label, pr["partner_id"])
            lookup[key] = pr

    # Merge
    records = []
    for _, row in partner_regime_df.iterrows():
        key = (row["neuron_label"], row["partner_id"])
        pr = lookup.get(key, {})
        record = row.to_dict()
        record["effect_ratio"] = pr.get("effect_ratio", np.nan)
        record["mean_pairwise_distance"] = pr.get("observed", {}).get("mean_pairwise_distance", np.nan)
        record["max_branch_fraction"] = pr.get("observed", {}).get("max_branch_fraction", np.nan)
        record["p_distance"] = pr.get("p_distance", np.nan)
        record["z_distance"] = pr.get("z_distance", np.nan)
        records.append(record)

    return pd.DataFrame(records)


def test_compactness_by_regime(merged_df):
    """Test whether partner compactness differs by dominant regime.

    Returns dict with test results.
    """
    work = merged_df.dropna(subset=["effect_ratio"]).copy()
    work = work[work["dominant_regime"].isin([0, 1, 2])]

    groups = [g["effect_ratio"].values for _, g in work.groupby("dominant_regime")]
    groups = [g for g in groups if len(g) >= 3]

    result = {
        "n_partners": len(work),
        "regime_counts": work["dominant_regime"].value_counts().to_dict(),
    }

    if len(groups) < 2:
        result["kruskal_H"] = np.nan
        result["kruskal_p"] = np.nan
        return result

    H, p = stats.kruskal(*groups)
    result["kruskal_H"] = H
    result["kruskal_p"] = p

    # Per-regime summaries
    for regime_val in [0, 1, 2]:
        subset = work[work["dominant_regime"] == regime_val]["effect_ratio"]
        result[f"regime_{regime_val}_median"] = subset.median() if len(subset) > 0 else np.nan
        result[f"regime_{regime_val}_mean"] = subset.mean() if len(subset) > 0 else np.nan
        result[f"regime_{regime_val}_n"] = len(subset)

    # Pairwise Mann-Whitney: regime 0 vs 2
    g0 = work[work["dominant_regime"] == 0]["effect_ratio"].values
    g2 = work[work["dominant_regime"] == 2]["effect_ratio"].values
    if len(g0) >= 3 and len(g2) >= 3:
        U, p_mw = stats.mannwhitneyu(g0, g2, alternative="two-sided")
        result["mw_0v2_U"] = U
        result["mw_0v2_p"] = p_mw
    else:
        result["mw_0v2_U"] = np.nan
        result["mw_0v2_p"] = np.nan

    return result


def spanning_partner_analysis(merged_df):
    """Analyze partners whose synapses span multiple regimes.

    Returns summary DataFrame of spanning vs non-spanning partners.
    """
    work = merged_df.dropna(subset=["effect_ratio"]).copy()

    spanning = work[work["spans_regimes"]]
    non_spanning = work[~work["spans_regimes"]]

    summary = {
        "n_spanning": len(spanning),
        "n_non_spanning": len(non_spanning),
        "spanning_median_effect": spanning["effect_ratio"].median() if len(spanning) > 0 else np.nan,
        "non_spanning_median_effect": non_spanning["effect_ratio"].median() if len(non_spanning) > 0 else np.nan,
    }

    if len(spanning) >= 3 and len(non_spanning) >= 3:
        U, p = stats.mannwhitneyu(
            spanning["effect_ratio"].values,
            non_spanning["effect_ratio"].values,
            alternative="two-sided"
        )
        summary["mw_U"] = U
        summary["mw_p"] = p

    return summary


def run_partner_regime_analysis(regime_df, output_dir=None):
    """Run complete partner-regime analysis for all neurons.

    Parameters
    ----------
    regime_df : DataFrame
        All branches with regime column.
    output_dir : Path or None

    Returns
    -------
    dict with 'partner_regime_df', 'merged_df', 'test_results', 'spanning_analysis'.
    """
    if output_dir is None:
        output_dir = Path.home() / "research" / "dendritic-regime-coupling" / "results"
    output_dir = Path(output_dir)

    # Load clustering results
    clustering_results = load_clustering_results()

    # Map partners to regimes for all neurons
    all_partner_dfs = []
    for label, root_id in NEURONS:
        print(f"  Mapping partners for {label}...")
        pdf = map_partner_synapses_to_regimes(label, root_id, regime_df)
        if pdf is not None:
            all_partner_dfs.append(pdf)
            print(f"    {len(pdf)} partners with k>=3")

    if not all_partner_dfs:
        return None

    partner_regime_df = pd.concat(all_partner_dfs, ignore_index=True)
    merged_df = merge_with_clustering_results(partner_regime_df, clustering_results)

    # Save
    merged_df.to_csv(output_dir / "partner_regime_mapping.csv", index=False)

    # Test
    test_results = test_compactness_by_regime(merged_df)
    spanning = spanning_partner_analysis(merged_df)

    return {
        "partner_regime_df": partner_regime_df,
        "merged_df": merged_df,
        "test_results": test_results,
        "spanning_analysis": spanning,
    }
