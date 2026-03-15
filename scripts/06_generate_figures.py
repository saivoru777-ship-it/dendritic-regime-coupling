#!/usr/bin/env python3
"""Step 6: Generate all publication figures.

Main figures (7):
1. Branch morphometry distributions
2. Regime classification scatter
3. Spatial organization by regime (key figure)
4. Confound control panels
5. Partner compactness by regime
6. Sensitivity sweep
7. Regime-grouped VMR

Supplementary figures (selected):
S1. Cross cell type coupling
S4. Exc vs inh input type coupling
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import viz
from src.branch_morphometry import load_neuron, NEURONS
from src.regime_grouped_vmr import compute_regime_vmr_per_neuron


def main():
    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    spatial_path = results_dir / "all_branch_features_with_spatial.csv"
    if not spatial_path.exists():
        print("ERROR: Run pipeline steps 01-03 first")
        sys.exit(1)

    df = pd.read_csv(spatial_path)
    print(f"Loaded {len(df)} branches")

    # ===== Fig 1: Branch morphometry =====
    print("Generating Fig 1: Branch morphometry...")
    viz.fig_branch_morphometry(df, save_path=figures_dir / "fig_branch_morphometry.pdf")

    # ===== Fig 2: Regime classification =====
    print("Generating Fig 2: Regime classification...")
    viz.fig_regime_classification(df, save_path=figures_dir / "fig_regime_classification.pdf")

    # ===== Fig 3: Spatial org by regime (key figure) =====
    print("Generating Fig 3: Spatial organization by regime...")
    viz.fig_spatial_org_by_regime(df, save_path=figures_dir / "fig_spatial_org_by_regime.pdf")

    # ===== Fig 4: Confound control =====
    print("Generating Fig 4: Confound control...")
    viz.fig_confound_control(
        df,
        metric_raw="pairwise_compactness_raw",
        metric_z="pairwise_compactness_z",
        save_path=figures_dir / "fig_confound_control.pdf",
    )

    # ===== Fig 5: Partner compactness =====
    partner_path = results_dir / "partner_regime_mapping.csv"
    if partner_path.exists():
        print("Generating Fig 5: Partner compactness...")
        partner_df = pd.read_csv(partner_path)
        viz.fig_partner_compactness(partner_df, save_path=figures_dir / "fig_partner_compactness.pdf")
    else:
        print("Skipping Fig 5: partner_regime_mapping.csv not found")

    # ===== Fig 6: Sensitivity =====
    sensitivity_path = results_dir / "coupling_sensitivity_pairwise_compactness_z.csv"
    if sensitivity_path.exists():
        print("Generating Fig 6: Sensitivity...")
        sens_df = pd.read_csv(sensitivity_path)
        viz.fig_sensitivity(sens_df, save_path=figures_dir / "fig_sensitivity.pdf")
    else:
        print("Skipping Fig 6: sensitivity data not found")

    # ===== Fig 7: Regime-grouped VMR =====
    print("Generating Fig 7: Regime-grouped VMR...")
    # Use first excitatory neuron as representative
    label, root_id = NEURONS[0]  # exc_23P
    result = load_neuron(label, root_id)
    if result is not None:
        skeleton, snap_valid = result
        neuron_df = df[df["neuron_label"] == label]
        regime_labels = np.full(len(skeleton.branches), -1, dtype=int)
        for _, row in neuron_df.iterrows():
            bidx = int(row["branch_idx"])
            if bidx < len(regime_labels):
                regime_labels[bidx] = int(row["regime"]) if not pd.isna(row["regime"]) else -1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vmr_result = compute_regime_vmr_per_neuron(
                snap_valid, skeleton, regime_labels, n_scales=12, n_mocks=100, seed=42
            )
        viz.fig_regime_grouped_vmr(
            vmr_result["curves"],
            vmr_result["envelopes"],
            save_path=figures_dir / "fig_regime_grouped_vmr.pdf",
        )
    else:
        print("Skipping Fig 7: could not load representative neuron")

    # ===== Supplementary figures =====
    print("Generating supplementary figures...")

    # S1: Cross cell type
    viz.fig_cross_cell_type(df, save_path=figures_dir / "fig_cross_cell_type.pdf")

    # S4: Input type coupling
    viz.fig_input_type_coupling(df, save_path=figures_dir / "fig_input_type_coupling.pdf")

    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
