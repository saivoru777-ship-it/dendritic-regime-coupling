#!/usr/bin/env python3
"""Step 3: Compute spatial organization metrics + branch null z-scores.

Three metrics: Clark-Evans, interval CV, pairwise compactness.
Z-scored against per-branch uniform null (1000 draws each).

Outputs:
- results/all_branch_features_with_spatial.csv (adds 6 columns to regime CSV)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.branch_null import compute_branch_z_scores
from src.branch_morphometry import load_neuron, NEURONS

N_DRAWS = 1000


def main():
    results_dir = PROJECT_ROOT / "results"
    regime_path = results_dir / "all_branch_features_with_regimes.csv"

    if not regime_path.exists():
        print("ERROR: Run 02_classify_regimes.py first")
        sys.exit(1)

    print("=" * 60)
    print("Step 3: Computing spatial organization metrics + z-scores")
    print("=" * 60)

    df = pd.read_csv(regime_path)

    # Initialize new columns
    for col in ["clark_evans_raw", "interval_cv_raw", "pairwise_compactness_raw",
                "clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]:
        df[col] = np.nan

    for label, root_id in NEURONS:
        print(f"Processing {label}...")
        result = load_neuron(label, root_id)
        if result is None:
            print(f"  SKIPPED")
            continue

        skeleton, snap_valid = result
        neuron_mask = df["neuron_label"] == label

        # Compute z-scored metrics
        z_results = compute_branch_z_scores(
            snap_valid, skeleton.branches,
            n_draws=N_DRAWS, seed=42,
        )

        # Map back to DataFrame
        n_branches = len(skeleton.branches)
        branch_indices = df.loc[neuron_mask, "branch_idx"].values

        for metric in ["clark_evans", "interval_cv", "pairwise_compactness"]:
            raw_key = f"{metric}_raw"
            z_key = f"{metric}_z"
            raw_vals = z_results[raw_key]
            z_vals = z_results[z_key]

            for i, bidx in enumerate(branch_indices):
                if bidx < n_branches:
                    df.loc[neuron_mask & (df["branch_idx"] == bidx), raw_key] = raw_vals[bidx]
                    df.loc[neuron_mask & (df["branch_idx"] == bidx), z_key] = z_vals[bidx]

        n_valid = df.loc[neuron_mask, "clark_evans_z"].notna().sum()
        print(f"  {n_valid} branches with valid z-scores")

    # Summary
    for metric in ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]:
        valid = df[metric].dropna()
        print(f"\n{metric}:")
        print(f"  N valid: {len(valid)}")
        print(f"  Mean: {valid.mean():.3f}")
        print(f"  Std: {valid.std():.3f}")
        print(f"  Range: [{valid.min():.2f}, {valid.max():.2f}]")

    # Save
    df.to_csv(results_dir / "all_branch_features_with_spatial.csv", index=False)
    print(f"\nSaved to {results_dir / 'all_branch_features_with_spatial.csv'}")


if __name__ == "__main__":
    main()
