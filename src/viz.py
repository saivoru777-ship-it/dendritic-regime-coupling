"""Publication figures for dendritic regime coupling analysis.

Uses Wong 2011 colorblind-safe palette, 8pt Arial, 300 DPI.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Wong 2011 colorblind-safe palette
COLORS = {
    "summation-like": "#0072B2",      # blue
    "nonlinear-prone": "#E69F00",     # orange
    "compartmentalized": "#D55E00",   # vermillion
    "exc": "#009E73",                 # green
    "inh": "#CC79A7",                 # pink
    "bpc": "#F0E442",                 # yellow
    "null": "#999999",                # grey
}

REGIME_ORDER = ["summation-like", "nonlinear-prone", "compartmentalized"]

FIGSIZE_SINGLE = (3.5, 3.0)
FIGSIZE_DOUBLE = (7.0, 3.0)
FIGSIZE_TRIPLE = (7.0, 6.0)


def _setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def fig_branch_morphometry(df, save_path=None):
    """Fig 1: Distributions of electrotonic length, diameter, soma distance."""
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_DOUBLE)

    features = [
        ("electrotonic_length", "Electrotonic length (L/λ)", True),
        ("mean_diameter_nm", "Mean diameter (nm)", False),
        ("soma_distance_nm", "Soma distance (nm)", False),
    ]

    for ax, (col, xlabel, log_scale) in zip(axes, features):
        for ct, color in [("excitatory", COLORS["exc"]), ("inhibitory", COLORS["inh"]),
                          ("inhibitory_BPC", COLORS["bpc"])]:
            subset = df[df["cell_type"] == ct][col].dropna()
            subset = subset[np.isfinite(subset)]
            if len(subset) == 0:
                continue
            label = ct.replace("inhibitory_", "")
            ax.hist(subset, bins=30, alpha=0.5, color=color, label=label, density=True)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        if log_scale:
            ax.set_xscale("log")
        ax.legend(frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def fig_regime_classification(df, save_path=None):
    """Fig 2: Scatter (electrotonic_length x diameter) colored by regime."""
    _setup_style()
    fig, (ax_main, ax_inset) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE,
                                              gridspec_kw={"width_ratios": [2, 1]})

    valid = df[df["regime"] >= 0].copy()
    for regime_name in REGIME_ORDER:
        subset = valid[valid["regime_name"] == regime_name]
        ax_main.scatter(
            subset["electrotonic_length"],
            subset["mean_diameter_nm"],
            c=COLORS[regime_name],
            s=8, alpha=0.5, label=regime_name, edgecolors="none",
        )

    ax_main.set_xlabel("Electrotonic length (L/λ)")
    ax_main.set_ylabel("Mean diameter (nm)")
    ax_main.set_xscale("log")
    ax_main.axvline(0.1, color="k", ls="--", lw=0.5, alpha=0.5)
    ax_main.axvline(0.5, color="k", ls="--", lw=0.5, alpha=0.5)
    ax_main.legend(frameon=False, markerscale=2)

    # Inset: order-regime agreement
    from .structural_regimes import check_regime_agreement
    agreement = check_regime_agreement(valid["regime"].values, valid["order_regime"].values)
    kappa = agreement["kappa"]
    ax_inset.bar(["Cohen's κ"], [kappa], color="#666666", width=0.4)
    ax_inset.set_ylim(0, 1)
    ax_inset.set_ylabel("Agreement")
    ax_inset.set_title(f"Order alignment: κ={kappa:.2f}")
    ax_inset.axhline(0.5, color="k", ls=":", lw=0.5, alpha=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def fig_spatial_org_by_regime(df, metrics=None, save_path=None):
    """Fig 3: Violin/box of z-scored spatial org by regime (key figure)."""
    _setup_style()
    if metrics is None:
        metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]

    metric_labels = {
        "clark_evans_z": "Clark-Evans (z)",
        "interval_cv_z": "Interval CV (z)",
        "pairwise_compactness_z": "Pairwise compactness (z)",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.5))
    if len(metrics) == 1:
        axes = [axes]

    valid = df[df["regime"] >= 0].copy()

    for ax, metric in zip(axes, metrics):
        data_by_regime = []
        positions = []
        colors = []
        for j, regime_name in enumerate(REGIME_ORDER):
            subset = valid[valid["regime_name"] == regime_name][metric].dropna()
            if len(subset) > 0:
                data_by_regime.append(subset.values)
                positions.append(j)
                colors.append(COLORS[regime_name])

        if data_by_regime:
            parts = ax.violinplot(data_by_regime, positions=positions, showextrema=False)
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.3)

            bp = ax.boxplot(data_by_regime, positions=positions, widths=0.3,
                           patch_artist=True, showfliers=False)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # Overlay individual points (jittered)
            rng = np.random.default_rng(42)
            for j, (data, pos) in enumerate(zip(data_by_regime, positions)):
                jitter = rng.normal(0, 0.06, size=len(data))
                ax.scatter(pos + jitter, data, c=colors[j], s=4, alpha=0.3, edgecolors="none")

        ax.set_xticks(range(len(REGIME_ORDER)))
        ax.set_xticklabels([r.replace("-", "\n") for r in REGIME_ORDER], rotation=0)
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.axhline(0, color="k", ls=":", lw=0.5, alpha=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def fig_confound_control(df, metric_raw, metric_z, save_path=None):
    """Fig 4: Confound control panels."""
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))

    valid = df[df["regime"] >= 0].copy()

    # Panel (a): Raw metric shows order confound
    ax = axes[0]
    ax.scatter(valid["branch_order"], valid[metric_raw], s=4, alpha=0.3,
              c=[COLORS[r] for r in valid["regime_name"]], edgecolors="none")
    ax.set_xlabel("Branch order")
    ax.set_ylabel(f"Raw {metric_raw}")
    ax.set_title("(a) Raw metric vs order")

    # Panel (b): Z-scored removes confound
    ax = axes[1]
    ax.scatter(valid["branch_order"], valid[metric_z], s=4, alpha=0.3,
              c=[COLORS[r] for r in valid["regime_name"]], edgecolors="none")
    ax.set_xlabel("Branch order")
    ax.set_ylabel(f"Z-scored {metric_z}")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.set_title("(b) Z-score removes confound")

    # Panel (c): Within-order stratified
    from .coupling_tests import within_order_stratified_test
    strat = within_order_stratified_test(valid, metric_z)
    significant = strat["p_value"].dropna() < 0.05
    ax = axes[2]
    ax.bar(range(len(strat)), -np.log10(strat["p_value"].fillna(1.0)),
          color=["#D55E00" if s else "#999999" for s in significant])
    ax.axhline(-np.log10(0.05), color="k", ls="--", lw=0.5)
    ax.set_xlabel("Branch order stratum")
    ax.set_ylabel("-log10(p)")
    ax.set_title("(c) Within-order coupling")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def fig_partner_compactness(merged_df, save_path=None):
    """Fig 5: Partner effect ratios stratified by branch regime."""
    _setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    valid = merged_df.dropna(subset=["effect_ratio"]).copy()
    valid = valid[valid["dominant_regime"].isin([0, 1, 2])]

    data_by_regime = []
    colors = []
    for regime, name in enumerate(REGIME_ORDER):
        subset = valid[valid["dominant_regime"] == regime]["effect_ratio"]
        data_by_regime.append(subset.values)
        colors.append(COLORS[name])

    if data_by_regime:
        bp = ax.boxplot(data_by_regime, patch_artist=True, showfliers=False, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        rng = np.random.default_rng(42)
        for i, (data, color) in enumerate(zip(data_by_regime, colors)):
            jitter = rng.normal(0, 0.08, size=len(data))
            ax.scatter(i + 1 + jitter, data, c=color, s=6, alpha=0.4, edgecolors="none")

    ax.set_xticklabels([r.replace("-", "\n") for r in REGIME_ORDER])
    ax.set_ylabel("Effect ratio (obs / null)")
    ax.axhline(1.0, color="k", ls=":", lw=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def fig_sensitivity(sensitivity_df, save_path=None):
    """Fig 6: Coupling effect size vs threshold sweep."""
    _setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    bio = sensitivity_df[sensitivity_df["source"] == "biophysical"]
    quant = sensitivity_df[sensitivity_df["source"] == "quantile"]

    if len(quant) > 0:
        ax.plot(range(len(quant)), quant["max_abs_coefficient"].values,
               "o-", color="#666666", ms=4, label="Quantile thresholds")

    if len(bio) > 0:
        ax.axhline(bio["max_abs_coefficient"].values[0], color="#D55E00",
                   ls="--", lw=1.5, label="Biophysical (0.1, 0.5)")

    ax.set_xlabel("Threshold variant")
    ax.set_ylabel("|Regime coefficient|")
    ax.legend(frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def fig_regime_grouped_vmr(regime_curves, regime_envelopes=None, save_path=None):
    """Fig 7: VMR fingerprint curves per regime with null envelopes."""
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)

    for regime, (ax, name) in enumerate(zip(axes, REGIME_ORDER)):
        rc = regime_curves.get(regime)
        if rc is None or rc.get("curves") is None:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                   transform=ax.transAxes, fontsize=8)
            ax.set_title(name)
            continue

        curves = rc["curves"]
        color = COLORS[name]

        # Null envelope
        if regime_envelopes and regime in regime_envelopes and regime_envelopes[regime] is not None:
            env = regime_envelopes[regime]
            n_pts = min(len(env["scales"]), len(env["ci_low"]))
            ax.fill_between(env["scales"][:n_pts] / 1000,
                          env["ci_low"][:n_pts], env["ci_high"][:n_pts],
                          color=COLORS["null"], alpha=0.3, label="95% null CI")

        # Observed
        scales_um = np.array(curves["scales"]) / 1000
        ax.plot(scales_um, curves["variance_values"], "o-", color=color, ms=3, lw=1.5)
        ax.axhline(1, color="k", ls=":", lw=0.5)
        ax.set_xlabel("Scale (μm)")
        if regime == 0:
            ax.set_ylabel("VMR")
        ax.set_xscale("log")
        ax.set_title(f"{name}\n(n={rc['n_branches']} br, {rc['n_synapses']} syn)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# --- Supplementary figures ---

def fig_cross_cell_type(df, metric_z="clark_evans_z", save_path=None):
    """Fig S1: Coupling patterns by cell type, BPC highlighted."""
    _setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)

    valid = df[(df["regime"] >= 0) & df[metric_z].notna()].copy()

    subtypes = valid["subtype"].unique()
    x_positions = {}
    for i, st in enumerate(sorted(subtypes)):
        x_positions[st] = i

    rng = np.random.default_rng(42)
    for _, row in valid.iterrows():
        x = x_positions[row["subtype"]] + rng.normal(0, 0.1)
        color = COLORS.get(row["regime_name"], "#999999")
        marker = "D" if row.get("is_bpc", False) else "o"
        ax.scatter(x, row[metric_z], c=color, s=8, alpha=0.4, marker=marker, edgecolors="none")

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(sorted(subtypes)), rotation=45)
    ax.set_ylabel(metric_z)
    ax.axhline(0, color="k", ls=":", lw=0.5)

    legend_elements = [Patch(facecolor=COLORS[r], label=r) for r in REGIME_ORDER]
    ax.legend(handles=legend_elements, frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def fig_input_type_coupling(df, save_path=None):
    """Fig S4: Exc vs inh spatial organization within each regime."""
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))

    valid = df[(df["regime"] >= 0)].copy()
    # Classify branches as exc-dominated or inh-dominated
    valid["input_dominant"] = np.where(valid["exc_fraction"] > 0.5, "exc-dominant", "inh-dominant")

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    for ax, metric in zip(axes, metrics):
        for j, regime_name in enumerate(REGIME_ORDER):
            subset = valid[valid["regime_name"] == regime_name].dropna(subset=[metric, "input_dominant"])
            exc_dom = subset[subset["input_dominant"] == "exc-dominant"][metric]
            inh_dom = subset[subset["input_dominant"] == "inh-dominant"][metric]

            offset = j * 3
            if len(exc_dom) > 0:
                ax.bar(offset, exc_dom.mean(), width=0.8, color=COLORS["exc"], alpha=0.6)
                ax.errorbar(offset, exc_dom.mean(), yerr=exc_dom.sem(), color="k", capsize=2, lw=0.5)
            if len(inh_dom) > 0:
                ax.bar(offset + 1, inh_dom.mean(), width=0.8, color=COLORS["inh"], alpha=0.6)
                ax.errorbar(offset + 1, inh_dom.mean(), yerr=inh_dom.sem(), color="k", capsize=2, lw=0.5)

        ticks = [j * 3 + 0.5 for j in range(3)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([r[:6] for r in REGIME_ORDER], rotation=45)
        ax.set_ylabel(metric.replace("_z", " (z)"))
        ax.axhline(0, color="k", ls=":", lw=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
