#!/usr/bin/env python3
"""Step 4: Coupling analysis — does structural regime predict spatial organization?

Tests:
- Mixed-effects model with covariates
- Kruskal-Wallis per neuron (BH-FDR)
- Permutation test (regime shuffle within neuron)
- Within-order stratified analysis
- Residual analysis (regress out branch order)
- Regime shuffle within order bins
- Sensitivity sweep (threshold robustness)

Outputs:
- results/coupling_results.json
- results/coupling_sensitivity.csv
- results/coupling_kruskal_wallis.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.coupling_tests import run_all_coupling_tests


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
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
    print("Step 4: Coupling analysis")
    print("=" * 60)

    df = pd.read_csv(spatial_path)
    print(f"Loaded {len(df)} branches")

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    covariates = ["synapse_count", "total_length_nm", "exc_fraction"]

    all_results = {}

    for metric in metrics:
        print(f"\n{'='*40}")
        print(f"Metric: {metric}")
        print(f"{'='*40}")

        results = run_all_coupling_tests(df, metric, covariates=covariates, n_perms=10000, seed=42)

        # Print key results
        mm = results["mixed_model"]
        if mm.get("converged"):
            print(f"  Mixed model: converged, n={mm['n_obs']}")
            print(f"  Regime coefficients:")
            for k, v in mm["regime_coefficients"].items():
                p = mm["regime_pvalues"].get(k, np.nan)
                print(f"    {k}: coef={v:.4f}, p={p:.4g}")
        else:
            print(f"  Mixed model: FAILED ({mm.get('error', 'unknown')})")

        perm = results["permutation"]
        print(f"  Permutation test: stat={perm['observed_stat']:.4f}, p={perm['p_value']:.4g}")

        resid = results["residual"]
        if resid.get("converged"):
            print(f"  Residual analysis: order R²={resid['order_r2']:.3f}, residual p={resid['residual_p']:.4g}")

        order_shuffle = results["order_shuffle"]
        print(f"  Order-shuffle test: stat={order_shuffle['observed_stat']:.4f}, p={order_shuffle['p_value']:.4g}")

        # Store (remove non-serializable model object)
        serializable = {}
        for key, val in results.items():
            if key == "mixed_model":
                serializable[key] = {k: v for k, v in val.items() if k != "model"}
            elif isinstance(val, pd.DataFrame):
                serializable[key] = val.to_dict(orient="records")
            elif isinstance(val, dict) and "null_distribution" in val:
                serializable[key] = {k: v for k, v in val.items() if k != "null_distribution"}
            else:
                serializable[key] = val
        all_results[metric] = serializable

        # Save per-metric files
        kw = results["kruskal_wallis"]
        kw.to_csv(results_dir / f"coupling_kw_{metric}.csv", index=False)

        sens = results["sensitivity"]
        sens.to_csv(results_dir / f"coupling_sensitivity_{metric}.csv", index=False)

    # Save combined results
    with open(results_dir / "coupling_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {results_dir}")

    # === Summary table ===
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'MM coef':>10} {'MM p':>10} {'Perm p':>10} {'Shuffle p':>10}")
    print("-" * 70)
    for metric in metrics:
        r = all_results[metric]
        mm = r["mixed_model"]
        coefs = mm.get("regime_coefficients", {})
        pvals = mm.get("regime_pvalues", {})
        max_coef = max(abs(v) for v in coefs.values()) if coefs else np.nan
        min_p = min(pvals.values()) if pvals else np.nan
        perm_p = r["permutation"].get("p_value", np.nan)
        shuf_p = r["order_shuffle"].get("p_value", np.nan)
        print(f"  {metric:<28} {max_coef:>10.4f} {min_p:>10.4g} {perm_p:>10.4g} {shuf_p:>10.4g}")


if __name__ == "__main__":
    main()
