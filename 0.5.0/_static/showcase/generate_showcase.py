"""Generate the landing-page module showcase thumbnails.

Produces a set of transparent-background PNGs styled to match the dark, violet
→ magenta → pink brand palette of the mlquantify landing page. Each image
illustrates what one module/family of the library produces.

Run from anywhere::

    python docs/source/_static/showcase/generate_showcase.py

Images are written next to this script (``docs/source/_static/showcase/``) and
referenced from ``_templates/index.html``.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# -- Brand palette (mirrors _static/variables.css) --------------------------- #
VIOLET = "#A855F7"
MAGENTA = "#D946EF"
PINK = "#F43F5E"
TEAL = "#2DD4BF"
AMBER = "#FBBF24"
SKY = "#38BDF8"
SEQ = [VIOLET, TEAL, AMBER, SKY, PINK]

INK = "#e2e8f0"      # light text on dark cards
MUTED = "#94a3b8"    # axis chrome
GRID = "#64748b"


def _style(ax):
    """Apply the dark-card-friendly chrome to an axes."""
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_color(INK)
        lbl.set_fontsize(9)
    if ax.get_title():
        ax.title.set_color(INK)
        ax.title.set_fontsize(10)


def _new(figsize=(5.0, 3.1)):
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_alpha(0.0)
    return fig, ax


def _save(fig, name):
    fig.tight_layout(pad=0.6)
    path = os.path.join(HERE, name)
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", os.path.relpath(path))


# --------------------------------------------------------------------------- #
# 1. Counting — true vs predicted diagonal
# --------------------------------------------------------------------------- #
def counting():
    rng = np.random.default_rng(1)
    true = np.linspace(0.03, 0.97, 60)
    pred = np.clip(true + rng.normal(0, 0.045, true.size), 0, 1)
    fig, ax = _new()
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1.2)
    ax.scatter(true, pred, c=true, cmap="cool", s=42, alpha=0.9,
               edgecolors="white", linewidths=0.4)
    ax.set_xlabel("True prevalence")
    ax.set_ylabel("Predicted")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(ax)
    _save(fig, "counting.png")


# --------------------------------------------------------------------------- #
# 2. Distribution matching — histogram mixture
# --------------------------------------------------------------------------- #
def matching():
    x = np.linspace(0, 1, 22)
    neg = np.exp(-((x - 0.28) ** 2) / 0.02)
    pos = np.exp(-((x - 0.72) ** 2) / 0.025)
    neg /= neg.sum()
    pos /= pos.sum()
    test = 0.62 * pos + 0.38 * neg + np.random.default_rng(0).normal(0, 0.002, x.size)
    test = np.clip(test, 0, None)
    fig, ax = _new()
    ax.fill_between(x, test, step="mid", color=MUTED, alpha=0.30, label="test")
    ax.step(x, neg, where="mid", color=TEAL, lw=2.0, label="class −")
    ax.step(x, pos, where="mid", color=VIOLET, lw=2.0, label="class +")
    ax.step(x, 0.62 * pos + 0.38 * neg, where="mid", color=MAGENTA, lw=2.4,
            label="best-fit mix")
    ax.set_xlabel("Classifier score")
    ax.set_ylabel("Frequency")
    ax.set_yticks([])
    leg = ax.legend(frameon=False, fontsize=7, labelcolor=INK, loc="upper center",
                    ncol=2, handlelength=1.2, columnspacing=1.0)
    _style(ax)
    _save(fig, "matching.png")


# --------------------------------------------------------------------------- #
# 3. Likelihood (EMQ) — EM convergence
# --------------------------------------------------------------------------- #
def likelihood():
    it = np.arange(0, 16)
    target = 0.8
    curve = target - (target - 0.5) * np.exp(-it / 2.6)
    fig, ax = _new()
    ax.axhline(target, ls="--", color=MUTED, lw=1.2)
    ax.plot(it, curve, "-o", color=MAGENTA, lw=2.4, ms=5,
            markerfacecolor=VIOLET, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Estimated prevalence")
    ax.set_ylim(0.45, 0.9)
    _style(ax)
    _save(fig, "likelihood.png")


# --------------------------------------------------------------------------- #
# 4. Representations — overlapping KDE densities
# --------------------------------------------------------------------------- #
def representations():
    x = np.linspace(-3.2, 3.6, 400)

    def g(mu, sd):
        return np.exp(-((x - mu) ** 2) / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))

    fig, ax = _new()
    for mu, sd, c in [(-1.1, 0.7, VIOLET), (0.4, 0.9, TEAL), (1.8, 0.6, AMBER)]:
        y = g(mu, sd)
        ax.fill_between(x, y, color=c, alpha=0.30)
        ax.plot(x, y, color=c, lw=2.0)
    ax.set_xlabel("Feature space")
    ax.set_ylabel("Density")
    ax.set_yticks([])
    _style(ax)
    _save(fig, "representations.png")


# --------------------------------------------------------------------------- #
# 5. Confidence regions — ternary simplex cloud + ellipse
# --------------------------------------------------------------------------- #
def confidence():
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    rng = np.random.default_rng(3)
    prev = rng.dirichlet([12, 7, 9], size=260)
    xy = prev @ corners
    fig, ax = _new(figsize=(4.6, 3.2))
    tri = np.vstack([corners, corners[0]])
    ax.plot(tri[:, 0], tri[:, 1], color=MUTED, lw=1.4)
    ax.scatter(xy[:, 0], xy[:, 1], s=22, color=VIOLET, alpha=0.40,
               edgecolors="none")
    # 2-sigma ellipse from the cloud covariance.
    from matplotlib.patches import Ellipse
    mean = xy.mean(0)
    cov = np.cov(xy.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * 2.45 * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, w, h, angle=angle, fill=False,
                         edgecolor=MAGENTA, lw=2.4))
    ax.scatter(*mean, color=PINK, s=70, marker="*", edgecolors="white",
               linewidths=0.5, zorder=5)
    for pt, lab in zip(corners, ["A", "B", "C"]):
        ax.annotate(lab, pt, color=INK, fontsize=10, fontweight="bold",
                    ha="center", va="center")
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "confidence.png")


# --------------------------------------------------------------------------- #
# 6. Protocols — sampled prevalences on the simplex
# --------------------------------------------------------------------------- #
def protocols():
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    rng = np.random.default_rng(7)
    prev = rng.dirichlet([1, 1, 1], size=500)
    xy = prev @ corners
    fig, ax = _new(figsize=(4.6, 3.2))
    tri = np.vstack([corners, corners[0]])
    ax.plot(tri[:, 0], tri[:, 1], color=MUTED, lw=1.4)
    ax.scatter(xy[:, 0], xy[:, 1], c=prev[:, 0] - prev[:, 2], cmap="plasma",
               s=20, alpha=0.75, edgecolors="none")
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "protocols.png")


# --------------------------------------------------------------------------- #
# 7. Metrics — error vs prior-probability shift
# --------------------------------------------------------------------------- #
def metrics():
    shift = np.linspace(0, 0.5, 40)
    rng = np.random.default_rng(5)
    fig, ax = _new()
    for base, c, name in [(0.02, TEAL, "ACC"), (0.015, VIOLET, "EMQ"),
                          (0.05, PINK, "CC")]:
        slope = {TEAL: 0.05, VIOLET: 0.04, PINK: 0.45}[c]
        mean = base + slope * shift
        band = 0.01 + 0.03 * shift
        ax.plot(shift, mean, color=c, lw=2.2, label=name)
        ax.fill_between(shift, mean - band, mean + band, color=c, alpha=0.16)
    ax.set_xlabel("Prior-probability shift")
    ax.set_ylabel("Absolute error")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    _style(ax)
    _save(fig, "metrics.png")


# --------------------------------------------------------------------------- #
# 8. Visualization — bias boxplots by prevalence bin
# --------------------------------------------------------------------------- #
def visualization():
    rng = np.random.default_rng(2)
    data = [rng.normal(m, s, 120) for m, s in
            [(-0.04, 0.05), (0.0, 0.04), (0.03, 0.05), (0.01, 0.06), (-0.02, 0.05)]]
    fig, ax = _new()
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color="white", lw=1.6),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                    flierprops=dict(marker="o", markersize=2,
                                    markerfacecolor=MUTED, markeredgecolor="none"))
    for patch, c in zip(bp["boxes"], [VIOLET, MAGENTA, TEAL, SKY, AMBER]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("none")
    ax.axhline(0, color=MUTED, ls="--", lw=1.0)
    ax.set_xlabel("True-prevalence bin")
    ax.set_ylabel("Bias")
    ax.set_xticklabels([])
    _style(ax)
    _save(fig, "visualization.png")


# --------------------------------------------------------------------------- #
# 9. Calibration — reliability curve
# --------------------------------------------------------------------------- #
def calibration():
    x = np.linspace(0, 1, 11)
    uncal = np.clip(x ** 1.7, 0, 1)
    cal = np.clip(x + 0.03 * np.sin(6 * x), 0, 1)
    fig, ax = _new()
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1.2, label="perfect")
    ax.plot(x, uncal, "-o", color=PINK, lw=2.2, ms=4,
            markeredgecolor="white", markeredgewidth=0.4, label="uncalibrated")
    ax.plot(x, cal, "-o", color=TEAL, lw=2.2, ms=4,
            markeredgecolor="white", markeredgewidth=0.4, label="calibrated")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")
    _style(ax)
    _save(fig, "calibration.png")


# --------------------------------------------------------------------------- #
# 10. Meta — ensemble member spread
# --------------------------------------------------------------------------- #
def meta():
    rng = np.random.default_rng(9)
    members = [rng.normal(m, 0.04, 200) for m in (0.55, 0.6, 0.58, 0.62, 0.57)]
    fig, ax = _new()
    parts = ax.violinplot(members, showmeans=False, showextrema=False,
                          widths=0.85)
    for body, c in zip(parts["bodies"], SEQ):
        body.set_facecolor(c)
        body.set_alpha(0.55)
        body.set_edgecolor("white")
        body.set_linewidth(0.5)
    ax.axhline(0.59, color=MAGENTA, ls="--", lw=1.6)
    ax.set_xlabel("Ensemble member")
    ax.set_ylabel("Prevalence estimate")
    ax.set_xticklabels([])
    _style(ax)
    _save(fig, "meta.png")


# --------------------------------------------------------------------------- #
# 11. Datasets — synthetic labelled clusters (make_quantification)
# --------------------------------------------------------------------------- #
def datasets():
    rng = np.random.default_rng(4)
    clusters = [((-1.35, 0.65), TEAL), ((1.15, 1.0), VIOLET), ((0.25, -1.15), AMBER)]
    fig, ax = _new()
    for (cx, cy), c in clusters:
        pts = rng.normal((cx, cy), (0.55, 0.5), size=(90, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=26, color=c, alpha=0.78,
                   edgecolors="white", linewidths=0.3)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xticks([])
    ax.set_yticks([])
    _style(ax)
    _save(fig, "datasets.png")


def main():
    counting()
    matching()
    likelihood()
    representations()
    confidence()
    protocols()
    metrics()
    visualization()
    calibration()
    meta()
    datasets()


if __name__ == "__main__":
    main()
