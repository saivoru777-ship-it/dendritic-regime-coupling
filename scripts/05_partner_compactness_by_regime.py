#!/usr/bin/env python3
"""Step 5: Partner compactness ratios stratified by structural regime.

Maps existing partner-level data from input_clustering_results.json to branch
regimes. Tests whether partners on compartmentalized branches show more
compactness than those on summation-like branches.

Outputs:
- results/partner_regime_mapping.csv
- results/partner_regime_tests.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.partner_regime import run_partner_regime_analysis


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def main():
    results_dir = PROJECT_ROOT / "results"
    spatial_path = results_dir / "all_branch_features_with_spatial.csv"

    if not spatial_path.exists():
        print("ERROR: Run 03_spatial_organization.py first")
        sys.exit(1)

    print("=" * 60)
    print("Step 5: Partner compactness by regime")
    print("=" * 60)

    df = pd.read_csv(spatial_path)

    result = run_partner_regime_analysis(df, output_dir=results_dir)
    if result is None:
        print("ERROR: No partner data available")
        sys.exit(1)

    merged = result["merged_df"]
    test_results = result["test_results"]
    spanning = result["spanning_analysis"]

    # Print results
    print(f"\nTotal partners mapped: {len(merged)}")
    print(f"Partners with effect_ratio: {merged['effect_ratio'].notna().sum()}")

    print(f"\nCompactness by dominant regime:")
    print(f"  {'Regime':<25} {'N':>5} {'Median ER':>12} {'Mean ER':>12}")
    print(f"  {'-'*55}")
    for regime in [0, 1, 2]:
        n = test_results.get(f"regime_{regime}_n", 0)
        med = test_results.get(f"regime_{regime}_median", np.nan)
        mean = test_results.get(f"regime_{regime}_mean", np.nan)
        from src.structural_regimes import REGIME_NAMES
        name = REGIME_NAMES.get(regime, "unknown")
        print(f"  {name:<25} {n:>5} {med:>12.4f} {mean:>12.4f}")

    print(f"\nKruskal-Wallis H={test_results['kruskal_H']:.3f}, p={test_results['kruskal_p']:.4g}")
    if not np.isnan(test_results.get("mw_0v2_p", np.nan)):
        print(f"Mann-Whitney (summation vs compartmentalized): U={test_results['mw_0v2_U']:.0f}, p={test_results['mw_0v2_p']:.4g}")

    print(f"\nSpanning partner analysis:")
    print(f"  Spanning: {spanning['n_spanning']} (median ER={spanning['spanning_median_effect']:.4f})")
    print(f"  Non-spanning: {spanning['n_non_spanning']} (median ER={spanning['non_spanning_median_effect']:.4f})")

    # Save test results
    with open(results_dir / "partner_regime_tests.json", "w") as f:
        json.dump({"test_results": test_results, "spanning_analysis": spanning},
                  f, indent=2, cls=NumpyEncoder)

    print(f"\nSaved to {results_dir}")


if __name__ == "__main__":
    main()
