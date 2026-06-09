"""
Generate all documentation plots for mlquantify user guide.

Run from the project root:
    python docs/source/images/plot_scripts/generate_doc_plots.py

Outputs are saved to docs/source/images/.
"""

import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import roc_curve

# ── output directory ──────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "..")   # docs/source/images/

STYLE = {
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
}
plt.rcParams.update(STYLE)

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"
PURPLE = "#9467bd"
GREY   = "#7f7f7f"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CC Bias  (foundations)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cc_bias():
    """Show how CC over/underestimates prevalence versus the ideal diagonal."""

    rng = np.random.default_rng(42)
    n = 800
    X, y = make_classification(n_samples=n, n_features=20, n_informative=5,
                               weights=[0.5, 0.5], random_state=0)

    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X, y)

    true_prevs  = np.linspace(0.01, 0.99, 40)
    cc_prevs    = []
    pcc_prevs   = []

    for p in true_prevs:
        n_pos = max(1, int(p * 500))
        n_neg = 500 - n_pos
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if len(pos_idx) < n_pos or len(neg_idx) < n_neg:
            cc_prevs.append(np.nan)
            pcc_prevs.append(np.nan)
            continue
        idx = np.concatenate([
            rng.choice(pos_idx, n_pos, replace=False),
            rng.choice(neg_idx, n_neg, replace=False),
        ])
        X_s, y_s = X[idx], y[idx]
        hard = clf.predict(X_s)
        soft = clf.predict_proba(X_s)[:, 1]
        cc_prevs.append(hard.mean())
        pcc_prevs.append(soft.mean())

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot([0, 1], [0, 1], color=GREY, lw=1.5, ls="--", label="Ideal (no bias)")
    ax.plot(true_prevs, cc_prevs,  color=RED,    lw=2,   label="CC  (biased)")
    ax.plot(true_prevs, pcc_prevs, color=ORANGE, lw=2,   label="PCC (less biased)")

    ax.set_xlabel("True prevalence", fontsize=12)
    ax.set_ylabel("Estimated prevalence", fontsize=12)
    ax.set_title("CC Bias Under Prior Probability Shift\n"
                 "(trained on 50/50, tested at varying prevalences)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    path = os.path.join(OUT, "cc_bias.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Dataset Shift  (foundations)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset_shift():
    """Illustrate prior probability shift: same feature distributions, different priors."""

    rng = np.random.default_rng(7)
    n = 500

    # Class-conditional feature distributions (same in train and test)
    mu_neg, mu_pos = 0.0, 2.5
    sigma = 1.0

    # Training: 70 % negative, 30 % positive
    n_neg_tr, n_pos_tr = int(0.70 * n), int(0.30 * n)
    X_neg_tr = rng.normal(mu_neg, sigma, n_neg_tr)
    X_pos_tr = rng.normal(mu_pos, sigma, n_pos_tr)

    # Test: 20 % negative, 80 % positive  (shift!)
    n_neg_te, n_pos_te = int(0.20 * n), int(0.80 * n)
    X_neg_te = rng.normal(mu_neg, sigma, n_neg_te)
    X_pos_te = rng.normal(mu_pos, sigma, n_pos_te)

    bins = np.linspace(-3, 6, 40)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, neg, pos, title, p_neg, p_pos in [
        (axes[0], X_neg_tr, X_pos_tr, "Training set\n(70% negative, 30% positive)", 0.70, 0.30),
        (axes[1], X_neg_te, X_pos_te, "Test set\n(20% negative, 80% positive)",     0.20, 0.80),
    ]:
        ax.hist(neg, bins=bins, alpha=0.6, color=BLUE,   density=True, label=f"Negative ({p_neg:.0%})")
        ax.hist(pos, bins=bins, alpha=0.6, color=ORANGE, density=True, label=f"Positive ({p_pos:.0%})")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Feature value", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("Prior Probability Shift  —  class-conditional distributions unchanged,\n"
                 "only the class proportions differ", fontsize=11, y=1.03)
    fig.tight_layout()

    path = os.path.join(OUT, "dataset_shift.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ROC Threshold Selection  (adjust_counting)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_thresholds():
    """Highlight where each threshold-adjustment method sits on the ROC curve."""

    X, y = make_classification(n_samples=800, n_features=20, n_informative=6,
                               weights=[0.6, 0.4], random_state=5)
    clf = LogisticRegression(max_iter=1000, random_state=5)
    scores = cross_val_predict(clf, X, y, cv=10, method="predict_proba")[:, 1]
    fpr_arr, tpr_arr, thresh = roc_curve(y, scores)

    # Remove inf threshold at start
    fpr_arr, tpr_arr, thresh = fpr_arr[1:], tpr_arr[1:], thresh[1:]

    def find_threshold(rule):
        if rule == "TAC":
            idx = np.argmin(np.abs(thresh - 0.5))
        elif rule == "TX":
            idx = np.argmin(np.abs((1 - tpr_arr) - fpr_arr))
        elif rule == "TMAX":
            idx = np.argmax(tpr_arr - fpr_arr)
        elif rule == "T50":
            idx = np.argmin(np.abs(tpr_arr - 0.5))
        return fpr_arr[idx], tpr_arr[idx], thresh[idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, color=BLUE, lw=2, label="ROC curve")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)

    markers = {"TAC": ("s", RED,    "TAC  (τ=0.5)"),
               "TX":  ("^", ORANGE, "TX   (FPR = 1−TPR)"),
               "TMAX":("D", GREEN,  "TMAX (max |TPR−FPR|)"),
               "T50": ("o", PURPLE, "T50  (TPR ≈ 0.5)")}

    for rule, (marker, color, label) in markers.items():
        fx, tx, _ = find_threshold(rule)
        ax.scatter(fx, tx, s=120, zorder=5, color=color, marker=marker, label=label)

    # MS sweep region
    ax.fill_between(fpr_arr, tpr_arr, alpha=0.08, color=BLUE, label="MS  (median over all thresholds)")

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title("Threshold Selection Policies on the ROC Curve", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    path = os.path.join(OUT, "roc_threshold_policies.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DyS / Histogram Matching  (distribution_matching)
# ─────────────────────────────────────────────────────────────────────────────

def plot_histogram_matching():
    """Show how DyS finds the mixture of class histograms that matches the test histogram."""

    rng = np.random.default_rng(3)

    # Simulate classifier positive-class scores
    pos_scores = rng.beta(6, 2, 300)   # concentrated near 1
    neg_scores = rng.beta(2, 6, 300)   # concentrated near 0

    true_prev = 0.35
    n_test = 400
    n_pos = int(true_prev * n_test)
    test_scores = np.concatenate([
        rng.beta(6, 2, n_pos),
        rng.beta(2, 6, n_test - n_pos),
    ])

    bins = np.linspace(0, 1, 21)
    centers = 0.5 * (bins[:-1] + bins[1:])

    h_pos  = np.histogram(pos_scores,  bins=bins, density=True)[0]
    h_neg  = np.histogram(neg_scores,  bins=bins, density=True)[0]
    h_test = np.histogram(test_scores, bins=bins, density=True)[0]

    # Best-fit mixture at true_prev
    h_mix  = true_prev * h_pos + (1 - true_prev) * h_neg

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    axes[0].bar(centers, h_pos,  width=0.045, color=ORANGE, alpha=0.8, label="Positive class")
    axes[0].bar(centers, h_neg,  width=0.045, color=BLUE,   alpha=0.8, label="Negative class")
    axes[0].set_title("Training score histograms\n(class-conditional)", fontsize=11)
    axes[0].set_xlabel("Classifier score"); axes[0].legend(fontsize=9)

    axes[1].bar(centers, h_test, width=0.045, color=GREEN, alpha=0.8, label="Test histogram")
    axes[1].set_title("Test score histogram\n(unlabelled, unknown prevalence)", fontsize=11)
    axes[1].set_xlabel("Classifier score"); axes[1].legend(fontsize=9)

    axes[2].bar(centers, h_test, width=0.045, color=GREEN, alpha=0.5, label="Test histogram")
    axes[2].step(centers, h_mix, where="mid", color=RED, lw=2.5,
                 label=f"Best-fit mixture  (α={true_prev:.0%})")
    axes[2].set_title(f"Mixture matching\n(estimated prevalence ≈ {true_prev:.0%})", fontsize=11)
    axes[2].set_xlabel("Classifier score"); axes[2].legend(fontsize=9)

    fig.suptitle("DyS / HDy Concept: Find the mixture α·H$^+$ + (1−α)·H$^-$ closest to H$^U$",
                 fontsize=12)
    fig.tight_layout()

    path = os.path.join(OUT, "histogram_matching.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  APP Protocol Grid  (protocols)
# ─────────────────────────────────────────────────────────────────────────────

def plot_app_protocol():
    """Visualise how APP sweeps the prevalence space vs a single test split."""

    n_prev    = 11   # 0.0, 0.1, ..., 1.0
    repeats   = 5
    prevs     = np.linspace(0, 1, n_prev)

    rng = np.random.default_rng(0)

    # APP: regular grid with noise
    app_x = np.repeat(prevs, repeats)
    app_y = rng.uniform(0.00, 0.03, len(app_x))  # jitter for visibility

    # Natural split: cluster near 0.5
    natural_x = rng.normal(0.5, 0.08, 30).clip(0, 1)
    natural_y = np.ones(30) * 0.5

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    # — APP panel —
    ax = axes[0]
    ax.scatter(app_x, app_y + 0.5, s=60, color=BLUE, alpha=0.8, zorder=3)
    for p in prevs:
        ax.axvline(p, color=BLUE, lw=0.7, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Positive class prevalence", fontsize=12)
    ax.set_title(f"APP  — {n_prev} prevalence levels × {repeats} repeats\n"
                 f"= {n_prev * repeats} test samples", fontsize=11)
    ax.set_yticks([])
    ax.set_ylabel("Samples (stacked)", fontsize=11)

    # — NPP panel —
    ax = axes[1]
    ax.scatter(natural_x, np.full(30, 0.5), s=60, color=ORANGE, alpha=0.8, zorder=3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Positive class prevalence", fontsize=12)
    ax.set_title("NPP  — natural prevalence variation\n(clusters near training prevalence)",
                 fontsize=11)
    ax.set_yticks([])

    fig.suptitle("Evaluation Protocols: APP (systematic sweep) vs NPP (natural)",
                 fontsize=12)
    fig.tight_layout()

    path = os.path.join(OUT, "app_protocol.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAE vs RAE weighting  (evaluation_metrics)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_comparison():
    """Show how MAE and RAE weight absolute errors differently across prevalences."""

    true_prevs = np.linspace(0.01, 0.99, 200)
    abs_err    = 0.05   # fixed absolute error at every point

    mae_weight = np.ones_like(true_prevs) * abs_err       # constant
    rae_weight = abs_err / (true_prevs + 1e-6)            # amplified at low prevalences
    rae_weight = np.clip(rae_weight, 0, 5)                # cap for display

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: weight vs prevalence
    ax = axes[0]
    ax.plot(true_prevs, mae_weight * 100, color=BLUE,   lw=2.5, label="MAE  (constant weight)")
    ax.plot(true_prevs, rae_weight * 100, color=ORANGE, lw=2.5, label="RAE  (amplified at low prev)")
    ax.set_xlabel("True prevalence", fontsize=12)
    ax.set_ylabel("Contribution to total error (%)", fontsize=11)
    ax.set_title("How MAE and RAE weight a fixed 5 pp error", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)

    # Right: example scenario
    scenario_prev = [0.05, 0.10, 0.30, 0.50, 0.70, 0.90]
    scenario_est  = [p + 0.05 for p in scenario_prev]
    mae_s = [abs(e - t) for e, t in zip(scenario_est, scenario_prev)]
    rae_s = [abs(e - t) / (t + 1e-6) for e, t in zip(scenario_est, scenario_prev)]

    x = np.arange(len(scenario_prev))
    w = 0.35
    ax2 = axes[1]
    ax2.bar(x - w/2, mae_s, width=w, color=BLUE,   alpha=0.8, label="MAE")
    ax2.bar(x + w/2, rae_s, width=w, color=ORANGE, alpha=0.8, label="RAE")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{p:.0%}" for p in scenario_prev], fontsize=9)
    ax2.set_xlabel("True prevalence (fixed +5 pp error)", fontsize=11)
    ax2.set_ylabel("Error value", fontsize=11)
    ax2.set_title("Same absolute error, different RAE penalty\n(RAE penalises rare classes much more)",
                  fontsize=11)
    ax2.legend(fontsize=10)

    fig.tight_layout()

    path = os.path.join(OUT, "metrics_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Method family accuracy overview  (foundations)
# ─────────────────────────────────────────────────────────────────────────────

def plot_method_comparison():
    """Synthetic performance sketch of main method families across prevalence levels."""

    rng  = np.random.default_rng(42)
    prevs = np.linspace(0.02, 0.98, 50)

    def simulate(method):
        """Return (mean error, std error) across prevalences — illustrative only."""
        if method == "CC":
            # biased — high error at extremes
            err = 0.20 * np.abs(prevs - 0.5) + rng.normal(0, 0.01, len(prevs))
        elif method == "PCC":
            err = 0.12 * np.abs(prevs - 0.5) + rng.normal(0, 0.01, len(prevs))
        elif method == "ACC/MS":
            err = 0.06 + 0.04 * np.abs(prevs - 0.5) + rng.normal(0, 0.008, len(prevs))
        elif method == "EMQ":
            err = 0.04 + 0.02 * np.abs(prevs - 0.5) + rng.normal(0, 0.005, len(prevs))
        elif method == "DyS/KDEy":
            err = 0.05 + 0.015 * np.abs(prevs - 0.5) + rng.normal(0, 0.006, len(prevs))
        return np.clip(err, 0, None)

    methods   = ["CC", "PCC", "ACC/MS", "EMQ", "DyS/KDEy"]
    colors    = [RED, ORANGE, PURPLE, GREEN, BLUE]
    styles    = ["-", "-", "--", "-", "--"]
    widths    = [1.5, 1.5, 2, 2.5, 2.5]

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, color, ls, lw in zip(methods, colors, styles, widths):
        err = simulate(method)
        ax.plot(prevs, err, color=color, ls=ls, lw=lw, label=method)

    ax.set_xlabel("True positive-class prevalence", fontsize=12)
    ax.set_ylabel("Absolute Error (AE)", fontsize=12)
    ax.set_title("Illustrative Error Profile of Main Method Families\n"
                 "(synthetic — for concept illustration only)", fontsize=11)
    ax.legend(fontsize=10, loc="upper center", ncol=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.35)

    path = os.path.join(OUT, "method_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating documentation plots …")
    plot_cc_bias()
    plot_dataset_shift()
    plot_roc_thresholds()
    plot_histogram_matching()
    plot_app_protocol()
    plot_metrics_comparison()
    plot_method_comparison()
    print("Done.")
