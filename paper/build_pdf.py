#!/usr/bin/env python3
"""Build a clean PDF of the paper from the LaTeX content using fpdf2."""

import os
from fpdf import FPDF


def sanitize(text):
    """Replace Unicode characters with latin-1 compatible equivalents."""
    replacements = {
        "\u2014": "--",    # em dash
        "\u2013": "-",     # en dash
        "\u2192": "->",    # right arrow
        "\u2265": ">=",    # >=
        "\u2264": "<=",    # <=
        "\u2212": "-",     # minus sign
        "\u03bb": "lambda",  # lambda
        "\u03b2": "beta",    # beta
        "\u03ba": "kappa",   # kappa
        "\u03c1": "rho",     # rho
        "\u03c3": "sigma",   # sigma
        "\u03bc": "mu",      # mu
        "\u00b7": "*",       # middle dot
        "\u00b2": "^2",      # superscript 2
        "\u00b3": "^3",      # superscript 3
        "\u03a9": "Ohm",     # Omega
        "\u00d7": "x",       # multiplication sign
        "\u2248": "~",       # approximately
        "\u2260": "!=",      # not equal
        chr(8226): "-",      # bullet
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# ── Paths ──────────────────────────────────────────────────────────────
PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(PAPER_DIR), "figures")
OUT_PATH = os.path.join(PAPER_DIR, "main.pdf")

# ── Custom PDF class ──────────────────────────────────────────────────
class PaperPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 5, "Voruganti -- Dendritic Regime Coupling", align="L")
            self.cell(0, 5, f"{self.page_no()}", align="R")
            self.ln(8)

    def footer(self):
        pass

    def section_title(self, title, level=1):
        self.ln(4)
        if level == 1:
            self.set_font("Helvetica", "B", 13)
        elif level == 2:
            self.set_font("Helvetica", "B", 11)
        else:
            self.set_font("Helvetica", "BI", 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, sanitize(title))
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, sanitize(text))
        self.ln(1)

    def bold_inline(self, bold_part, rest):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 30)
        self.write(5, sanitize(bold_part))
        self.set_font("Helvetica", "", 10)
        self.write(5, sanitize(rest))
        self.ln(6)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.cell(5, 5, "-")
        self.multi_cell(0, 5, sanitize(text))
        self.ln(0.5)

    def add_figure(self, fig_name, caption="", width=170):
        fig_path = os.path.join(FIG_DIR, fig_name)
        if os.path.exists(fig_path):
            self.ln(4)
            x = (210 - width) / 2
            self.image(fig_path, x=x, w=width)
            if caption:
                self.ln(2)
                self.set_font("Helvetica", "I", 9)
                self.set_text_color(80, 80, 80)
                self.multi_cell(0, 4.5, sanitize(caption))
            self.ln(4)

    def table_header(self, cols, widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(0, 0, 0)
        for col, w in zip(cols, widths):
            self.cell(w, 6, sanitize(col), border="B", fill=True, align="C")
        self.ln()

    def table_row(self, cells, widths, bold_first=False):
        self.set_text_color(30, 30, 30)
        for i, (cell, w) in enumerate(zip(cells, widths)):
            if i == 0 and bold_first:
                self.set_font("Helvetica", "B", 9)
            else:
                self.set_font("Helvetica", "", 9)
            self.cell(w, 5.5, cell, align="C" if i > 0 else "L")
        self.ln()


# ── Build the PDF ─────────────────────────────────────────────────────
pdf = PaperPDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# ── Title ──────────────────────────────────────────────────────────────
pdf.set_font("Helvetica", "B", 16)
pdf.set_text_color(0, 0, 0)
pdf.multi_cell(0, 8, "Electrotonic Structure Shapes Partner-Level\nSynaptic Organization on Cortical Dendrites", align="C")
pdf.ln(3)

pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 6, "Pretham Voruganti", align="C")
pdf.ln(10)

# ── Abstract ───────────────────────────────────────────────────────────
pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(0, 0, 0)
pdf.cell(0, 6, "Abstract")
pdf.ln(4)

pdf.set_font("Helvetica", "I", 9.5)
pdf.set_text_color(30, 30, 30)
abstract = (
    "How synaptic inputs are spatially organized on dendritic trees is thought to influence dendritic "
    "computation, yet the relationship between branch-level structural properties and input spatial "
    "organization has not been systematically characterized. Here we analyze 1,234 dendritic branches "
    "across 12 electron-microscopy-reconstructed cortical neurons from the MICrONS dataset, classifying "
    "each branch into structural regimes by electrotonic length and quantifying spatial organization "
    "with three independent metrics. We find that spatial organization is systematically coupled to "
    "structural regime: electrotonically compact branches exhibit more spatially clustered synaptic "
    "inputs than expected under a geometry-matched null model (pairwise compactness z-score, "
    "mixed-effects p = 9.3e-6). Formal mediation analysis reveals that this coupling is largely "
    "driven by partner-level innervation structure \u2014 compact branches receive concentrated input "
    "from fewer presynaptic partners making multiple contacts, while electrotonically isolated branches "
    "receive distributed input from many single-contact partners. The mediation is cell-type-specific: "
    "partner structure fully accounts for the coupling in excitatory neurons (100% mediated, bootstrap "
    "CI excludes zero) but only partially in inhibitory neurons (39% mediated), indicating a second, "
    "partner-independent organizational principle in inhibitory circuits. These results establish that "
    "electrotonic structure organizes synaptic input through cell-type-specific channels and provide "
    "an empirical foundation linking dendritic morphology to the spatial statistics of connectivity."
)
pdf.multi_cell(0, 4.5, sanitize(abstract))
pdf.ln(4)

# ── Introduction ───────────────────────────────────────────────────────
pdf.section_title("1  Introduction")

pdf.body_text(
    "Dendrites are not passive conduits. The spatial arrangement of synaptic inputs along dendritic "
    "branches can qualitatively change integration outcomes, enabling operations from linear summation "
    "to supralinear amplification depending on input proximity and dendritic properties (London & "
    "Hausser, 2005; Mel, 1994). Thin, electrotonically isolated branches can function as independent "
    "computational subunits (Polsky et al., 2004; Poirazi et al., 2003), while thick proximal dendrites "
    "sum inputs more linearly (Rall, 1967). This structural heterogeneity \u2014 quantified by electrotonic "
    "length (L/\u03bb) \u2014 creates a landscape of potential computational modes along a single neuron's "
    "arbor (Koch, 1999)."
)

pdf.body_text(
    "If spatial organization of inputs were systematically coupled to structural regime, it would "
    "suggest that synaptic placement is not merely constrained by geometry but co-organized with the "
    "biophysical properties that determine how those inputs are integrated. Several lines of evidence "
    "hint at such coupling: functionally related inputs cluster on specific branches in sensory cortex "
    "(Jia et al., 2010; Druckmann et al., 2014), dendritic spikes require spatially clustered "
    "excitation (Larkum et al., 2009; Branco et al., 2010), and presynaptic partners concentrate "
    "their synapses on select branches (Kasthuri et al., 2015). However, whether these observations "
    "reflect a systematic relationship between branch structure and input spatial organization across "
    "diverse cell types has not been tested."
)

pdf.body_text(
    "Here we address this question directly. Using serial electron microscopy reconstructions from the "
    "MICrONS dataset (MICrONS Consortium, 2021), we characterize the spatial organization of synaptic "
    "inputs on every branch of 12 cortical neurons (6 excitatory, 6 inhibitory including 2 bipolar "
    "cells). We classify branches into structural regimes by electrotonic length, quantify spatial "
    "organization with three independent metrics (Clark\u2013Evans ratio, interval coefficient of "
    "variation, pairwise compactness), and test for coupling using mixed-effects models with rigorous "
    "confound controls."
)

pdf.body_text(
    "We find that compact branches harbor more spatially clustered inputs than expected, while isolated "
    "branches approach uniform placement. Mediation analysis reveals that this coupling is largely "
    "explained by partner-level innervation architecture: compact branches attract repeat-contact "
    "partners whose boutons cluster, while isolated branches diversify across many single-contact "
    "partners. This mediation is complete in excitatory neurons but partial in inhibitory neurons, "
    "pointing to a second organizational axis in inhibitory circuits that remains to be characterized."
)

# ── Results ────────────────────────────────────────────────────────────
pdf.section_title("2  Results")

# 2.1
pdf.section_title("2.1  Branch morphometry and structural regime classification", level=2)

pdf.body_text(
    "We computed branch-level features for 1,234 dendritic branches across 12 MICrONS cortical "
    "neurons (Table 1). Electrotonic length (L/\u03bb) was calculated from cable theory using standard "
    "passive parameters (Rm = 20 k\u03a9\u00b7cm\u00b2, Ri = 150 \u03a9\u00b7cm) applied to EM-measured branch "
    "lengths and diameters (Methods)."
)

pdf.body_text(
    "The observed electrotonic length distribution was compressed relative to textbook values: 62.4% "
    "of branches fell below L/\u03bb = 0.1, the classical compact threshold (Rall, 1967), with a median "
    "of 0.059 and maximum of 0.50. This compression reflects the thicker diameters measured by EM "
    "compared to light microscopy, yielding larger space constants and smaller L/\u03bb values. We "
    "therefore classified branches into structural regimes using tertile thresholds on the observed "
    "distribution (L/\u03bb < 0.027: summation-like; 0.027 \u2264 L/\u03bb < 0.117: nonlinear-prone; "
    "L/\u03bb \u2265 0.117: compartmentalized), producing balanced groups of 411, 412, and 411 branches "
    "respectively (Figure 1)."
)

pdf.body_text(
    "Electrotonic regime classification was essentially independent of branch order (Cohen's \u03ba = "
    "-0.06), confirming that electrotonic length captures structural variation beyond simple "
    "topological depth. Sensitivity analysis across six alternative quantile thresholds showed "
    "stable coupling effect sizes, and the biophysical thresholds (0.1, 0.5) were included as a "
    "reference in all sensitivity comparisons."
)

pdf.add_figure("fig_regime_classification.png",
               "Figure 1. Structural regime classification by electrotonic length and diameter. "
               "Tertile thresholds on observed L/\u03bb distribution produce balanced groups.",
               width=150)

# 2.2
pdf.section_title("2.2  Spatial organization is coupled to structural regime", level=2)

pdf.body_text(
    "We quantified spatial organization of synaptic inputs on each branch using three independent "
    "metrics: Clark\u2013Evans nearest-neighbor ratio, interval coefficient of variation (CV), and "
    "pairwise compactness (mean pairwise distance normalized by branch length). To control for the "
    "trivial dependence of raw metrics on branch geometry, we z-scored each metric against a "
    "per-branch uniform null (1,000 draws of k synapses on a branch of length L; Methods)."
)

pdf.body_text(
    "Pairwise compactness showed the strongest coupling to regime (Figure 2). Summation-like branches "
    "had a mean z-score of \u22120.96 (substantially more compact than expected), while nonlinear-prone "
    "(\u22120.31) and compartmentalized (\u22120.39) branches were closer to the null expectation. A "
    "mixed-effects model with neuron as random intercept and synapse count, branch length, and "
    "excitatory fraction as covariates yielded significant regime coefficients (regime 1 vs. 0: "
    "\u03b2 = 0.60, p = 9.3e-6; regime 2 vs. 0: \u03b2 = 0.82, p = 3.5e-5). Permutation testing "
    "(10,000 regime-label shuffles within neurons) confirmed significance (p < 0.001)."
)

pdf.body_text(
    "Clark\u2013Evans trended in the same direction (regime 2 coefficient: \u03b2 = 0.35, p = 0.047; "
    "permutation p = 0.002), while interval CV showed a weaker pattern (\u03b2 = \u22120.41, p = 0.037; "
    "permutation p = 0.50). The convergence of two metrics and the weakness of the third is "
    "informative: Clark\u2013Evans and pairwise compactness both capture overall clustering tendency, "
    "while interval CV measures spacing regularity between consecutive synapses. Spatial organization "
    "by regime manifests as differences in overall clustering, not in the regularity of inter-synapse "
    "intervals."
)

pdf.add_figure("fig_spatial_org_by_regime.png",
               "Figure 2. Spatial organization z-scores by structural regime for three independent "
               "metrics. Pairwise compactness shows the strongest coupling.",
               width=160)

# 2.3
pdf.section_title("2.3  Confound controls", level=2)

pdf.body_text("Three analyses addressed whether the coupling is an artifact of branch order or other confounds.")

pdf.bold_inline("Residual analysis. ",
    "Regressing pairwise compactness z-scores on branch order explained only 1.1% of variance "
    "(R\u00b2 = 0.011). Kruskal\u2013Wallis on the residuals by regime remained highly significant "
    "(p = 1.2e-4), confirming that regime captures structure beyond topological position.")

pdf.bold_inline("Within-order stratification. ",
    "Within individual branch-order strata, regime coupling persisted in the largest strata "
    "(order 1: p = 0.0008, n = 97; order 2: p = 0.043, n = 148). Of 18 testable strata, "
    "3 reached significance at p < 0.05.")

pdf.bold_inline("Sensitivity to thresholds. ",
    "Coupling effect size was stable across all six quantile-based threshold variants (maximum "
    "absolute regime coefficient range: 0.81\u20131.27), confirming that the result is not an artifact "
    "of a particular threshold choice.")

pdf.add_figure("fig_confound_control.png",
               "Figure 3. Confound controls. (a) Raw metric shows order confound, (b) z-scored "
               "removes it, (c) within-order stratified analysis.",
               width=160)

# 2.4
pdf.section_title("2.4  Coverage bias assessment", level=2)

pdf.body_text(
    "The minimum synapse requirements for spatial metrics created regime-asymmetric exclusion: 61.6% "
    "of summation-like branches were excluded from pairwise compactness analysis (vs. 1.7% of "
    "compartmentalized branches), because compact branches are shorter and more likely to have few "
    "synapses. Excluded branches had a mean synapse count of 0.8 (vs. 16.5 for included branches). "
    "To test whether this selection drives the result, we verified that regime predicts partner "
    "structure even after controlling for synapse count (p = 2.3e-43), confirming that the "
    "coverage-surviving subset is not artifactually enriched."
)

# 2.5
pdf.section_title("2.5  Partner-level innervation structure mediates the coupling", level=2)

pdf.body_text(
    "The coupling between regime and spatial compactness could reflect either (a) a direct structural "
    "constraint on synapse placement or (b) an indirect effect mediated by the partner-level "
    "innervation landscape. We found that compact branches receive qualitatively different innervation "
    "from isolated branches (Table 2): summation-like branches averaged 1.51 synapses per partner per "
    "branch with 30.5% of partners making multiple contacts, while compartmentalized branches averaged "
    "1.19 synapses per partner with only 13.4% multi-contact partners. Conversely, compartmentalized "
    "branches received input from 19.7 unique partners on average versus 5.3 for summation-like branches."
)

pdf.body_text("Formal mediation analysis (Baron & Kenny, 1986) tested whether partner structure "
    "(mean synapses per partner per branch) mediates the regime\u2013compactness link:")

pdf.bullet("Path a (regime \u2192 mediator): a = \u22120.138, p = 1.1e-21")
pdf.bullet("Path b (mediator \u2192 compactness, controlling regime): b = \u22121.12, p = 5.0e-16")
pdf.bullet("Direct effect after mediation: c' = 0.069, p = 0.24 (non-significant)")
pdf.bullet("Indirect effect: ab = 0.156; proportion mediated: 69.4%")
pdf.bullet("Sobel test: z = 6.33, p = 2.4e-10")
pdf.bullet("Bootstrap 95% CI for indirect effect: [0.093, 0.231] (5,000 resamples, excludes zero)")

pdf.ln(2)
pdf.body_text(
    "Synapse density provided a second mediation pathway (proportion mediated: 107%). In a joint model "
    "controlling both mediators simultaneously, the direct regime effect vanished entirely "
    "(\u03b2 = \u22120.06, p = 0.37), while both density (p = 5.4e-5) and partner structure "
    "(p = 1.4e-10) remained significant. Mediation proportions exceeding 100% reflect suppression "
    "between correlated mediators sharing explained variance (density\u2013partner \u03c1 = 0.34; all "
    "variance inflation factors < 2), not overcounting of the indirect effect."
)

pdf.body_text(
    "The coverage bias diagnostic confirmed that regime predicts partner structure even after "
    "controlling for synapse count (regime coefficients: p = 1.3e-25 and p = 2.3e-43), "
    "establishing that the partner gradient is genuinely tied to electrotonic structure rather than "
    "an artifact of differential branch inclusion."
)

# 2.6
pdf.section_title("2.6  Cell-type-specific mediation reveals two organizational axes", level=2)

pdf.bold_inline("Excitatory neurons ",
    "(493 branches, 6 neurons). The partner gradient was large (Cohen's d = 0.89, Mann\u2013Whitney "
    "p = 3.1e-11). Pooled mediation was complete: total regime effect c = 0.192; direct effect after "
    "mediator c' = \u22120.0004; proportion mediated 100.2%. The spatial clustering on compact "
    "excitatory branches is entirely attributable to repeat-contact partners making clustered boutons.")

pdf.bold_inline("Inhibitory neurons ",
    "(277 branches, 4 non-BPC neurons). The partner gradient existed but with smaller effect size "
    "(Cohen's d = 0.56, significant at p = 3.1e-9 after controlling synapse count). Pooled mediation "
    "was partial: total effect c = 0.260; direct effect c' = 0.159; proportion mediated 38.9%. Over "
    "60% of the regime\u2013compactness coupling in inhibitory neurons operates through a "
    "partner-independent mechanism.")

pdf.bold_inline("Bipolar cells ",
    "(83 branches, 2 neurons). No significant partner gradient (p = 0.07 after controlling synapse "
    "count, Cohen's d = 0.18). Mediation was negligible (1.4%). The BPC coupling, though underpowered, "
    "appears independent of partner structure entirely.")

pdf.body_text(
    "The per-neuron gradient was consistent: 9 of 11 testable neurons showed higher "
    "synapses-per-partner on summation-like than compartmentalized branches (binomial p = 0.033). "
    "The two exceptions were both inhibitory (inh_MC, inh_BPC), consistent with the cell-type "
    "dissociation."
)

pdf.add_figure("fig_cross_cell_type.png",
               "Figure 4. Cell-type-specific mediation. Excitatory neurons show complete mediation; "
               "inhibitory neurons show partial mediation with a residual partner-independent axis.",
               width=160)

# 2.7
pdf.section_title("2.7  Partner compactness by regime", level=2)

pdf.body_text(
    "To connect these findings to partner-level spatial clustering, we mapped 1,254 presynaptic "
    "partners (each with \u22653 synapses) to the structural regime of their target branches. Partner "
    "compactness (observed mean pairwise distance / null median, \"effect ratio\") differed "
    "significantly by dominant target regime (Kruskal\u2013Wallis H = 9.46, p = 0.009). Non-spanning "
    "partners (all synapses within one regime) showed greater compactness (median effect ratio 0.70) "
    "than regime-spanning partners (0.88), indicating that single-regime targeting is associated with "
    "tighter spatial organization."
)

pdf.add_figure("fig_partner_compactness.png",
               "Figure 5. Partner compactness by structural regime of target branches.",
               width=140)

# ── Discussion ─────────────────────────────────────────────────────────
pdf.section_title("3  Discussion")

pdf.body_text(
    "We report three principal findings. First, spatial organization of synaptic inputs is "
    "systematically coupled to dendritic structural regime across 1,234 branches from 12 cortical "
    "neurons. Second, this coupling is largely mediated by partner-level innervation architecture: "
    "compact branches consolidate input from repeat-contact partners whose boutons cluster, while "
    "isolated branches diversify across many single-contact partners. Third, the mediation is "
    "cell-type-specific, revealing two distinct organizational axes \u2014 one fully explained by "
    "partner structure in excitatory neurons, and a second, partner-independent axis in inhibitory "
    "neurons."
)

pdf.section_title("3.1  The direction of coupling", level=2)

pdf.body_text(
    "The observed coupling \u2014 compact branches more clustered, isolated branches closer to random "
    "\u2014 is the opposite of what a simple nonlinear-subunit model would predict (where isolated "
    "branches would need clustered inputs for local spike generation). This does not contradict the "
    "subunit hypothesis but rather suggests that spatial organization is not optimized for computation "
    "at the single-branch level. Instead, it reflects the innervation constraints: many axons "
    "converging on a short, accessible proximal branch create clustering as a byproduct of repeated "
    "contact, while the longer path to distal branches favors single-contact diversification."
)

pdf.section_title("3.2  Two organizational axes", level=2)

pdf.body_text(
    "The cell-type dissociation is the most generative finding. In excitatory neurons, partner "
    "structure fully explains the coupling \u2014 compact branches are clustered because they attract "
    "repeat visitors. In inhibitory neurons, a substantial residual (61%) persists after accounting "
    "for partner structure, pointing to a second organizational principle. This could reflect "
    "zone-specific inhibitory targeting logic, where different classes of inhibitory afferents "
    "(e.g., basket cells vs. chandelier cells vs. Martinotti cells) preferentially target specific "
    "dendritic compartments regardless of how many synapses each makes. The precise characterization "
    "of this partner-independent axis remains an open question."
)

pdf.section_title("3.3  Electrotonic length distribution", level=2)

pdf.body_text(
    "The finding that 62% of MICrONS branches fall below L/\u03bb = 0.1 warrants comment. "
    "EM-reconstructed dendrites yield thicker diameter measurements than light microscopy, inflating "
    "the space constant \u03bb and compressing L/\u03bb. Our tertile thresholds (0.027, 0.117) capture "
    "relative structural variation within this compressed range. The coupling we report is a graded "
    "relationship that is stable across threshold choices (sensitivity analysis), not a sharp "
    "transition between discrete computational modes. This is consistent with dendritic biophysics: "
    "electrotonic properties vary continuously, and integration mode depends on the interaction of "
    "structure, channel expression, and input timing (Stuart et al., 1997)."
)

pdf.add_figure("fig_sensitivity.png",
               "Figure 6. Sensitivity analysis: coupling effect size across threshold choices.",
               width=140)

pdf.section_title("3.4  Limitations", level=2)

pdf.body_text(
    "The coverage bias from minimum synapse requirements creates asymmetric exclusion (61.6% of "
    "summation-like branches excluded vs. 1.7% of compartmentalized branches). Although the coverage "
    "diagnostic confirms the partner gradient survives controlling for synapse count, the "
    "summation-like branches that contribute to our analysis are a filtered subset with atypically "
    "high synapse density. Per-neuron mediation estimates are variable due to limited within-neuron "
    "sample sizes, though the pooled mediation is robust (bootstrap CI excludes zero). Passive cable "
    "parameters were assumed throughout; active conductances would modify the effective electrotonic "
    "structure."
)

# ── Methods ────────────────────────────────────────────────────────────
pdf.section_title("4  Methods")

pdf.section_title("4.1  Data", level=2)
pdf.body_text(
    "Twelve neurons (6 excitatory: 23P, 23P, 4P, 5P-ET, 5P-IT, 6P-CT; 6 inhibitory: 2 basket cells, "
    "2 Martinotti cells, 2 bipolar cells) were selected from the MICrONS cortical mm\u00b3 dataset "
    "(MICrONS Consortium, 2021). SWC skeleton files and synapse coordinates were obtained from prior "
    "reconstruction. Presynaptic partner identity and broad cell type (excitatory/inhibitory) were "
    "determined from the connectome."
)

pdf.section_title("4.2  Branch morphometry", level=2)
pdf.body_text(
    "Skeletons were loaded with a scale factor of 1,000 (converting \u03bcm to nm) and filtered to "
    "dendritic compartments (SWC types 1, 3, 4). Synapses were snapped to the skeleton using a "
    "KD-tree nearest-neighbor search with a maximum distance threshold of 50,000 nm. Branches were "
    "defined as maximal paths between branching or terminal points."
)
pdf.body_text(
    "For each branch, we computed: branch order (BFS from soma), mean diameter (2 \u00d7 mean SWC "
    "radius), total path length, electrotonic length (L/\u03bb, where \u03bb = sqrt(Rm\u00b7d / 4Ri), "
    "Rm = 20,000 \u03a9\u00b7cm\u00b2, Ri = 150 \u03a9\u00b7cm, d = mean diameter), soma distance "
    "(geodesic via Dijkstra on branch endpoints), synapse count, and excitatory/inhibitory synapse "
    "counts from partner cell type classification."
)

pdf.section_title("4.3  Regime classification", level=2)
pdf.body_text(
    "Branches were classified into three structural regimes by electrotonic length using tertile "
    "thresholds computed from the observed distribution. The 33rd percentile (L/\u03bb = 0.027) "
    "separated summation-like from nonlinear-prone; the 67th percentile (L/\u03bb = 0.117) separated "
    "nonlinear-prone from compartmentalized. Sensitivity analysis swept six alternative quantile-based "
    "threshold pairs and included the biophysical reference thresholds (0.1, 0.5)."
)

pdf.section_title("4.4  Spatial organization metrics", level=2)

pdf.bold_inline("Clark\u2013Evans ratio ",
    "(\u22655 synapses). Mean observed nearest-neighbor distance divided by expected under uniform "
    "placement with Donnelly edge correction (Clark & Evans, 1954; Donnelly, 1978): E[NN] = "
    "L/(2n) + 0.0514\u00b7L/n\u00b2. Values below 1 indicate clustering.")

pdf.bold_inline("Interval CV ",
    "(\u22653 synapses). Standard deviation of inter-synapse gaps divided by their mean. Values "
    "above 1 indicate clustered (heterogeneous) spacing.")

pdf.bold_inline("Pairwise compactness ",
    "(\u22653 synapses). Mean pairwise distance between synapses divided by branch length. Lower "
    "values indicate more compact distributions.")

pdf.section_title("4.5  Branch null model and z-scoring", level=2)
pdf.body_text(
    "For each branch with k synapses on length L, 1,000 draws of k positions from Uniform(0, L) "
    "generated null distributions for each metric. The z-score for each branch was "
    "(x_obs \u2212 \u03bc_null) / \u03c3_null. This controls for branch geometry (length and synapse "
    "count) so that z-scores are comparable across branches."
)

pdf.section_title("4.6  Statistical testing", level=2)

pdf.bold_inline("Mixed-effects model. ",
    "z-score ~ C(regime) + synapse_count + branch_length + exc_fraction + (1|neuron) via REML. "
    "Neuron as random intercept pools information across the 12 neurons while accounting for "
    "between-neuron variability.")

pdf.bold_inline("Permutation test. ",
    "Regime labels were shuffled within each neuron (10,000 permutations). The test statistic was "
    "the difference in mean z-score between regime 0 and regime 2.")

pdf.bold_inline("Per-neuron Kruskal\u2013Wallis. ",
    "Nonparametric test of z-score by regime within each neuron, corrected for multiple comparisons "
    "via Benjamini\u2013Hochberg FDR (Benjamini & Hochberg, 1995).")

pdf.bold_inline("Within-order stratification. ",
    "Kruskal\u2013Wallis on z-score by regime within each branch-order stratum. If coupling persists "
    "within order strata, it is not an artifact of branch order.")

pdf.bold_inline("Residual analysis. ",
    "Regressed z-score on branch order via OLS, then tested regime effect on residuals.")

pdf.section_title("4.7  Mediation analysis", level=2)
pdf.body_text(
    "Mediation followed the Baron and Kenny framework (1986) with bootstrap inference. For each "
    "candidate mediator M (mean synapses per partner per branch; synapse density): Path a: "
    "regime \u2192 M (OLS); Path b: M \u2192 compactness z, controlling regime (OLS); Indirect effect: "
    "ab; proportion mediated: (c \u2212 c')/c. Statistical significance was assessed via the Sobel "
    "test (Sobel, 1982) and bootstrap resampling (5,000 iterations, percentile confidence intervals). "
    "Mediation proportions exceeding 100% indicate suppression between correlated mediators sharing "
    "explained variance (confirmed by variance inflation factors < 2 for all predictors)."
)

pdf.section_title("4.8  Software", level=2)
pdf.body_text(
    "All analyses were implemented in Python using NumPy, SciPy, pandas, statsmodels, scikit-learn, "
    "and the Neurostat package for dendritic spatial statistics. Code is available at "
    "https://github.com/saivoru777-ship-it/dendritic-regime-coupling."
)

# ── Tables ─────────────────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("Tables")

# Table 1
pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 6, "Table 1. Summary of neurons analyzed.")
pdf.ln(6)

widths_t1 = [35, 40, 25, 25, 30]
pdf.table_header(["Label", "Cell type", "Branches", "Synapses", "Median L/\u03bb"], widths_t1)

neurons = [
    ("exc_23P", "Excitatory (L2/3)", "132", "1,063", "0.057"),
    ("exc_23P_2", "Excitatory (L2/3)", "122", "937", "0.047"),
    ("exc_4P", "Excitatory (L4)", "84", "990", "0.064"),
    ("exc_5PET", "Excitatory (L5 ET)", "220", "2,490", "0.043"),
    ("exc_5PIT", "Excitatory (L5 IT)", "141", "1,199", "0.062"),
    ("exc_6PCT", "Excitatory (L6 CT)", "102", "488", "0.063"),
    ("inh_BC", "Basket cell", "91", "1,617", "0.106"),
    ("inh_BC_2", "Basket cell", "100", "1,491", "0.065"),
    ("inh_MC", "Martinotti cell", "79", "1,584", "0.047"),
    ("inh_MC_2", "Martinotti cell", "59", "1,230", "0.072"),
    ("inh_BPC", "Bipolar cell", "56", "669", "0.072"),
    ("inh_BPC_2", "Bipolar cell", "48", "586", "0.060"),
]
for n in neurons:
    pdf.table_row(list(n), widths_t1)

# Total row
pdf.set_font("Helvetica", "B", 9)
pdf.cell(widths_t1[0], 5.5, "Total")
pdf.cell(widths_t1[1], 5.5, "", align="C")
pdf.cell(widths_t1[2], 5.5, "1,234", align="C")
pdf.cell(widths_t1[3], 5.5, "14,344", align="C")
pdf.cell(widths_t1[4], 5.5, "0.059", align="C")
pdf.ln(10)

# Table 2
pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 6, "Table 2. Partner innervation structure by structural regime.")
pdf.ln(6)

widths_t2 = [55, 35, 35, 45]
pdf.table_header(["", "Summation", "Nonlinear", "Compartment."], widths_t2)

t2_rows = [
    ("Mean synapses/partner", "1.51", "1.22", "1.19"),
    ("Frac. multi-contact", "30.5%", "16.0%", "13.4%"),
    ("Frac. syn from multi", "44.5%", "26.4%", "24.6%"),
    ("Mean partners/branch", "5.3", "9.4", "19.7"),
]
for r in t2_rows:
    pdf.table_row(list(r), widths_t2)

# ── References ─────────────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("References")

refs = [
    "Baron, R.M. & Kenny, D.A. (1986). The moderator-mediator variable distinction in social psychological research. J. Personality and Social Psychology, 51(6), 1173\u20131182.",
    "Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. J. Royal Statistical Society B, 57(1), 289\u2013300.",
    "Branco, T., Clark, B.A. & Hausser, M. (2010). Dendritic discrimination of temporal input sequences in cortical neurons. Science, 329(5999), 1671\u20131675.",
    "Clark, P.J. & Evans, F.C. (1954). Distance to nearest neighbor as a measure of spatial relationships in populations. Ecology, 35(4), 445\u2013453.",
    "Donnelly, K.P. (1978). Simulations to determine the variance and edge effect of total nearest-neighbour distance. Simulation Methods in Archaeology, 91\u201395.",
    "Druckmann, S. et al. (2014). Structured synaptic connectivity between hippocampal regions. Neuron, 81(3), 629\u2013640.",
    "Jia, H. et al. (2010). Dendritic organization of sensory input to cortical neurons in vivo. Nature, 464(7293), 1307\u20131312.",
    "Kasthuri, N. et al. (2015). Saturated reconstruction of a volume of neocortex. Cell, 162(3), 648\u2013661.",
    "Koch, C. (1999). Biophysics of Computation: Information Processing in Single Neurons. Oxford University Press.",
    "Larkum, M.E. et al. (2009). Synaptic integration in tuft dendrites of layer 5 pyramidal neurons. Science, 325(5941), 756\u2013760.",
    "London, M. & Hausser, M. (2005). Dendritic computation. Annual Review of Neuroscience, 28, 503\u2013532.",
    "Mel, B.W. (1994). Information processing in dendritic trees. Neural Computation, 6(6), 1031\u20131085.",
    "MICrONS Consortium (2021). Functional connectomics spanning multiple areas of mouse visual cortex. bioRxiv, doi:10.1101/2021.07.28.454025.",
    "Poirazi, P., Brannon, T. & Mel, B.W. (2003). Pyramidal neuron as two-layer neural network. Neuron, 37(6), 989\u2013999.",
    "Polsky, A., Mel, B.W. & Schiller, J. (2004). Computational subunits in thin dendrites of pyramidal cells. Nature Neuroscience, 7(6), 621\u2013627.",
    "Rall, W. (1967). Distinguishing theoretical synaptic potentials computed for different soma-dendritic distributions of synaptic input. J. Neurophysiology, 30(5), 1138\u20131168.",
    "Sobel, M.E. (1982). Asymptotic confidence intervals for indirect effects in structural equation models. Sociological Methodology, 13, 290\u2013312.",
    "Stuart, G. et al. (1997). Action potential initiation and backpropagation in neurons of the mammalian CNS. Trends in Neurosciences, 20(3), 125\u2013131.",
]

pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(30, 30, 30)
for i, ref in enumerate(refs, 1):
    pdf.multi_cell(0, 4.5, sanitize(f"[{i}] {ref}"))
    pdf.ln(1.5)

# ── Supplementary Figures ──────────────────────────────────────────────
pdf.add_page()
pdf.section_title("Supplementary Figures")

pdf.add_figure("fig_branch_morphometry.png",
               "Figure S1. Distributions of electrotonic length, diameter, and soma distance "
               "across all 1,234 branches, colored by neuron type.",
               width=160)

pdf.add_figure("fig_regime_grouped_vmr.png",
               "Figure S2. Variance-mean ratio fingerprint curves per structural regime with "
               "null envelopes.",
               width=140)

pdf.add_figure("fig_input_type_coupling.png",
               "Figure S3. Excitatory vs. inhibitory spatial organization within each regime.",
               width=140)

# ── Save ───────────────────────────────────────────────────────────────
pdf.output(OUT_PATH)
print(f"PDF written to {OUT_PATH}")
print(f"Pages: {pdf.page_no()}")
