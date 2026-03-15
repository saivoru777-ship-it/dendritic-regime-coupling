#!/usr/bin/env python3
"""Step 2: Classify branches into structural regimes.

Primary: electrotonic length thresholds (0.1, 0.5)
Robustness: branch order alignment check (Cohen's kappa)
Supplementary: k-means

Outputs:
- results/all_branch_features_with_regimes.csv
- results/regime_summary.csv
- results/regime_agreement.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.structural_regimes import (
    classify_branches,
    regime_size_table,
    check_regime_agreement,
)


def main():
    results_dir = PROJECT_ROOT / "results"
    features_path = results_dir / "all_branch_features.csv"

    if not features_path.exists():
        print("ERROR: Run 01_compute_branch_features.py first")
        sys.exit(1)

    print("=" * 60)
    print("Step 2: Classifying structural regimes")
    print("=" * 60)

    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} branches")

    # Classify (tertile thresholds computed from data)
    df = classify_branches(df)

    t_low = df.attrs.get("threshold_low", "?")
    t_high = df.attrs.get("threshold_high", "?")
    print(f"\nThresholds: L/λ < {t_low:.4f} (summation-like), ≥ {t_high:.4f} (compartmentalized)")
    print(f"\nRegime distribution:")
    size_table = regime_size_table(df)
    print(size_table.to_string())

    # Agreement check
    agreement = check_regime_agreement(df["regime"].values, df["order_regime"].values)
    print(f"\nElectrotonic vs order agreement:")
    print(f"  Cohen's kappa: {agreement['kappa']:.3f}")
    print(f"  Agreement fraction: {agreement['agreement_fraction']:.3f}")
    print(f"  N valid: {agreement['n_valid']}")

    # BPC stats
    bpc = df[df["is_bpc"]]
    print(f"\nBPC branches: {len(bpc)}")
    if len(bpc) > 0:
        print(f"  BPC regime distribution:")
        print(f"  {bpc['regime_name'].value_counts().to_string()}")

    # Save
    df.to_csv(results_dir / "all_branch_features_with_regimes.csv", index=False)
    size_table.to_csv(results_dir / "regime_summary.csv")
    with open(results_dir / "regime_agreement.json", "w") as f:
        json.dump(agreement, f, indent=2, default=str)

    print(f"\nSaved to {results_dir / 'all_branch_features_with_regimes.csv'}")


if __name__ == "__main__":
    main()
