"""Branch morphometry: compute features for every branch across all neurons.

Features per branch:
- Branch order (BFS from soma)
- Mean diameter (from SWC radii)
- Total length (from Branch.total_length)
- Electrotonic length (L/lambda, cable theory)
- Soma distance (Dijkstra geodesic)
- Synapse count (from SnapResult)
- Exc/inh synapse counts (from partner CSV)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from neurostat.io.swc import NeuronSkeleton, SnapResult
from soma_distance import precompute_branch_endpoint_distances, compute_soma_distances


# Cable theory constants
RM = 20_000  # membrane resistivity, Ohm * cm^2
RI = 150     # intracellular resistivity, Ohm * cm


NEURONS = [
    ("exc_23P", "864691135848859998"),
    ("exc_23P_2", "864691135866483845"),
    ("exc_4P", "864691135738528881"),
    ("exc_5PET", "864691135884866160"),
    ("exc_5PIT", "864691135256642223"),
    ("exc_6PCT", "864691135866795798"),
    ("inh_BC", "864691135293026230"),
    ("inh_BC_2", "864691135135829529"),
    ("inh_MC", "864691135273485073"),
    ("inh_MC_2", "864691136119505176"),
    ("inh_BPC", "864691136923311076"),
    ("inh_BPC_2", "864691135715512858"),
]

DATA_DIR = Path.home() / "research" / "neurostat-input-clustering" / "data" / "microns"
PARTNER_DIR = Path.home() / "research" / "neurostat-input-clustering" / "results"


def compute_branch_order(skeleton):
    """Compute branch order via BFS from the root branch.

    Returns array of ints, one per branch. The branch containing the root
    node gets order 0; its children get order 1, etc.
    """
    n_branches = len(skeleton.branches)
    order = np.full(n_branches, -1, dtype=int)

    # Build branch adjacency: node_id -> list of (branch_idx, endpoint)
    adj = skeleton._build_branch_adjacency()

    # Find the root branch (contains root_id)
    root_branch = None
    for i, br in enumerate(skeleton.branches):
        if skeleton.root_id in br.node_ids:
            root_branch = i
            break
    if root_branch is None:
        # Fallback: pick branch whose start_node is closest to root
        root_node = skeleton.root_id
        for i, br in enumerate(skeleton.branches):
            if br.start_node == root_node or br.end_node == root_node:
                root_branch = i
                break
    if root_branch is None:
        root_branch = 0

    # BFS over branch adjacency
    order[root_branch] = 0
    queue = [root_branch]
    while queue:
        current = queue.pop(0)
        curr_br = skeleton.branches[current]
        # Find neighbor branches via shared endpoints
        for endpoint_node in [curr_br.start_node, curr_br.end_node]:
            if endpoint_node not in adj:
                continue
            for (neighbor_idx, _) in adj[endpoint_node]:
                if neighbor_idx != current and order[neighbor_idx] == -1:
                    order[neighbor_idx] = order[current] + 1
                    queue.append(neighbor_idx)

    # Any unreached branches get max_order + 1
    unreached = order == -1
    if unreached.any():
        order[unreached] = order.max() + 1

    return order


def compute_mean_diameter(skeleton, branch_idx):
    """Mean diameter (2 * mean radius) for a branch, in nm."""
    branch = skeleton.branches[branch_idx]
    radii = []
    for nid in branch.node_ids:
        if nid in skeleton.nodes:
            radii.append(skeleton.nodes[nid].radius)
    if not radii:
        return np.nan
    return 2.0 * np.mean(radii)


def compute_electrotonic_length(length_nm, diameter_nm):
    """Compute electrotonic length L/lambda.

    lambda = sqrt(rm * d / (4 * ri))
    where rm in Ohm*cm^2, d in cm, ri in Ohm*cm.
    L and d must be converted from nm to cm.
    """
    if diameter_nm <= 0 or np.isnan(diameter_nm):
        return np.nan
    d_cm = diameter_nm * 1e-7   # nm -> cm
    L_cm = length_nm * 1e-7     # nm -> cm
    lam = np.sqrt(RM * d_cm / (4.0 * RI))
    if lam <= 0:
        return np.nan
    return L_cm / lam


def compute_soma_distance_per_branch(skeleton, endpoint_nodes, node_to_idx, endpoint_dists):
    """Compute geodesic distance from soma to each branch midpoint.

    Returns array of soma distances (nm), one per branch.
    """
    n_branches = len(skeleton.branches)
    soma_dists = np.full(n_branches, np.nan)

    # Find soma in endpoint nodes
    soma_idx = None
    if skeleton.root_id in node_to_idx:
        soma_idx = node_to_idx[skeleton.root_id]
    else:
        # Find closest endpoint to root
        root_node = skeleton.nodes[skeleton.root_id]
        min_dist = np.inf
        for nid in endpoint_nodes:
            if nid in skeleton.nodes:
                n = skeleton.nodes[nid]
                d = np.sqrt((n.x - root_node.x)**2 + (n.y - root_node.y)**2 + (n.z - root_node.z)**2)
                if d < min_dist:
                    min_dist = d
                    soma_idx = node_to_idx[nid]

    if soma_idx is None:
        return soma_dists

    for i, br in enumerate(skeleton.branches):
        # Distance to midpoint = distance to start + half length, or distance to end + half length
        start_idx = node_to_idx.get(br.start_node)
        end_idx = node_to_idx.get(br.end_node)
        if start_idx is not None:
            d_start = endpoint_dists[soma_idx, start_idx]
        else:
            d_start = np.inf
        if end_idx is not None:
            d_end = endpoint_dists[soma_idx, end_idx]
        else:
            d_end = np.inf

        # Soma distance to branch midpoint: min path via either endpoint + half length
        half_len = br.total_length / 2.0
        via_start = d_start + half_len
        via_end = d_end + half_len
        soma_dists[i] = min(via_start, via_end)

    return soma_dists


def count_synapses_per_branch(snap_result, n_branches):
    """Count total synapses per branch from SnapResult."""
    return np.bincount(snap_result.branch_ids, minlength=n_branches)


def count_exc_inh_per_branch(snap_result, pre_cell_types, n_branches):
    """Count excitatory and inhibitory synapses per branch.

    Parameters
    ----------
    snap_result : SnapResult
        Valid snapped synapses.
    pre_cell_types : np.ndarray of str
        Broad cell type for each synapse ('excitatory' or 'inhibitory').
    n_branches : int

    Returns
    -------
    exc_counts, inh_counts : np.ndarray, np.ndarray
    """
    exc_counts = np.zeros(n_branches, dtype=int)
    inh_counts = np.zeros(n_branches, dtype=int)
    for i, ct in enumerate(pre_cell_types):
        bid = snap_result.branch_ids[i]
        if ct == "excitatory":
            exc_counts[bid] += 1
        elif ct == "inhibitory":
            inh_counts[bid] += 1
    return exc_counts, inh_counts


def load_neuron(label, root_id):
    """Load skeleton and snap synapses. Returns (skeleton, snap_valid) or None."""
    swc_path = DATA_DIR / f"{label}_{root_id}.swc"
    syn_path = DATA_DIR / f"{label}_{root_id}_synapses.csv"

    if not swc_path.exists() or not syn_path.exists():
        print(f"  {label}: MISSING files, skipping")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skeleton_raw = NeuronSkeleton.from_swc_file(str(swc_path), scale_factor=1000.0)

    dendrite_skel = skeleton_raw.filter_by_type([1, 3, 4])
    if len(dendrite_skel.branches) < 3:
        dendrite_skel = skeleton_raw

    syn_df = pd.read_csv(syn_path)
    syn_coords_nm = syn_df[["x_um", "y_um", "z_um"]].values * 1000.0

    snap = dendrite_skel.snap_points(syn_coords_nm, d_max=50000.0)
    valid = snap.valid
    n_valid = valid.sum()
    if n_valid < 20:
        print(f"  {label}: only {n_valid} valid synapses, skipping")
        return None

    snap_valid = SnapResult(
        branch_ids=snap.branch_ids[valid],
        branch_positions=snap.branch_positions[valid],
        distances=snap.distances[valid],
        valid=np.ones(n_valid, dtype=bool),
    )
    return dendrite_skel, snap_valid


def load_partner_types(label, snap_result, syn_path):
    """Load partner CSV and match to snapped synapses.

    Returns array of broad cell types aligned with snap_result.
    """
    partner_path = PARTNER_DIR / f"{label}_presynaptic.csv"
    if not partner_path.exists():
        return None

    partner_df = pd.read_csv(partner_path)
    partner_coords = partner_df[["x_nm", "y_nm", "z_nm"]].values.astype(float)

    # Reconstruct valid synapse 3D coordinates from synapse CSV
    syn_df = pd.read_csv(syn_path)
    syn_coords_nm = syn_df[["x_um", "y_um", "z_um"]].values * 1000.0

    # Match using cKDTree
    tree = cKDTree(partner_coords)
    _, indices = tree.query(syn_coords_nm)

    # Partner types for all synapses
    all_types = partner_df["pre_cell_type_broad"].values[indices]

    return all_types


def compute_branch_features(label, root_id):
    """Compute all branch features for a single neuron.

    Returns DataFrame with one row per branch, or None on failure.
    """
    result = load_neuron(label, root_id)
    if result is None:
        return None

    skeleton, snap_valid = result
    n_branches = len(skeleton.branches)

    # Branch order
    orders = compute_branch_order(skeleton)

    # Diameter and electrotonic length
    diameters = np.array([compute_mean_diameter(skeleton, i) for i in range(n_branches)])
    lengths = np.array([br.total_length for br in skeleton.branches])
    e_lengths = np.array([
        compute_electrotonic_length(lengths[i], diameters[i])
        for i in range(n_branches)
    ])

    # Soma distance
    endpoint_nodes, node_to_idx, endpoint_dists = precompute_branch_endpoint_distances(skeleton)
    soma_dists = compute_soma_distance_per_branch(skeleton, endpoint_nodes, node_to_idx, endpoint_dists)

    # Synapse counts
    syn_counts = count_synapses_per_branch(snap_valid, n_branches)

    # Exc/inh counts
    syn_path = DATA_DIR / f"{label}_{root_id}_synapses.csv"
    all_types = load_partner_types(label, snap_valid, syn_path)

    if all_types is not None:
        # Filter to valid synapses (same mask used in load_neuron)
        syn_df = pd.read_csv(syn_path)
        syn_coords_nm = syn_df[["x_um", "y_um", "z_um"]].values * 1000.0
        snap_full = skeleton.snap_points(syn_coords_nm, d_max=50000.0)
        valid_mask = snap_full.valid
        valid_types = all_types[valid_mask]
        exc_counts, inh_counts = count_exc_inh_per_branch(snap_valid, valid_types, n_branches)
    else:
        exc_counts = np.full(n_branches, np.nan)
        inh_counts = np.full(n_branches, np.nan)

    # Determine cell type
    if label.startswith("exc"):
        cell_type = "excitatory"
    elif "BPC" in label:
        cell_type = "inhibitory_BPC"
    else:
        cell_type = "inhibitory"

    # Extract subtype from label
    parts = label.split("_")
    if label.startswith("exc"):
        subtype = parts[1]  # e.g., "23P", "4P", "5PET"
    else:
        subtype = parts[1]  # e.g., "BC", "MC", "BPC"

    df = pd.DataFrame({
        "neuron_label": label,
        "root_id": root_id,
        "cell_type": cell_type,
        "subtype": subtype,
        "branch_idx": np.arange(n_branches),
        "branch_order": orders,
        "mean_diameter_nm": diameters,
        "total_length_nm": lengths,
        "electrotonic_length": e_lengths,
        "soma_distance_nm": soma_dists,
        "synapse_count": syn_counts,
        "exc_count": exc_counts,
        "inh_count": inh_counts,
    })

    # Exc fraction (NaN if no synapses)
    total = df["exc_count"] + df["inh_count"]
    df["exc_fraction"] = np.where(total > 0, df["exc_count"] / total, np.nan)

    return df


def compute_all_branch_features(output_dir=None):
    """Compute branch features for all 12 neurons.

    Returns combined DataFrame and saves per-neuron + combined CSVs.
    """
    if output_dir is None:
        output_dir = Path.home() / "research" / "dendritic-regime-coupling" / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for label, root_id in NEURONS:
        print(f"Processing {label}...")
        df = compute_branch_features(label, root_id)
        if df is not None:
            df.to_csv(output_dir / f"{label}_branch_features.csv", index=False)
            all_dfs.append(df)
            print(f"  {len(df)} branches, {df['synapse_count'].sum()} synapses")
        else:
            print(f"  FAILED")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(output_dir / "all_branch_features.csv", index=False)
        print(f"\nTotal: {len(combined)} branches across {len(all_dfs)} neurons")
        return combined
    return None
