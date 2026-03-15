#!/usr/bin/env python3
"""Step 1: Compute branch morphometry features for all 12 MICrONS neurons.

Outputs:
- results/{label}_branch_features.csv per neuron
- results/all_branch_features.csv combined
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.branch_morphometry import compute_all_branch_features


def main():
    output_dir = PROJECT_ROOT / "results"
    print("=" * 60)
    print("Step 1: Computing branch morphometry features")
    print("=" * 60)

    df = compute_all_branch_features(output_dir=output_dir)

    if df is not None:
        print(f"\nSummary:")
        print(f"  Total branches: {len(df)}")
        print(f"  Branches with synapses: {(df['synapse_count'] > 0).sum()}")
        print(f"  Total synapses: {df['synapse_count'].sum()}")
        print(f"  Electrotonic length range: [{df['electrotonic_length'].min():.4f}, {df['electrotonic_length'].max():.2f}]")
        print(f"  Mean diameter range: [{df['mean_diameter_nm'].min():.0f}, {df['mean_diameter_nm'].max():.0f}] nm")

        # Per-neuron summary
        print(f"\nPer-neuron:")
        for label, grp in df.groupby("neuron_label"):
            n_syn = grp["synapse_count"].sum()
            n_br = len(grp)
            print(f"  {label}: {n_br} branches, {n_syn} synapses")

        print(f"\nSaved to {output_dir / 'all_branch_features.csv'}")
    else:
        print("ERROR: No data produced")
        sys.exit(1)


if __name__ == "__main__":
    main()
