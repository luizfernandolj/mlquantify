import numpy as np
import matplotlib.pyplot as plt
from mlquantify.representations import HistogramRepresentation

rng = np.random.default_rng(3)
scores = 0.2 + 0.3 * rng.beta(2, 3, size=(3000, 1))   # concentrated in [0.2, 0.5]
y = (scores[:, 0] > scores[:, 0].mean()).astype(int)
n = 12

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

rep_fixed = HistogramRepresentation(bins=(n,), range=(0.0, 1.0),
                                    bin_edges="fixed").fit(scores, y)
hf = rep_fixed.transform(scores)
ef = np.linspace(0, 1, n + 1)
axes[0].bar((ef[:-1] + ef[1:]) / 2, hf, width=np.diff(ef) * 0.9,
            color="#b9542a", edgecolor="white", linewidth=0.5)
axes[0].set_title("bin_edges='fixed'\n(equal width over range=[0, 1])", fontsize=10)

rep_auto = HistogramRepresentation(bins=(n,), bin_edges="auto").fit(scores, y)
ha = rep_auto.transform(scores)
ea = rep_auto.edges_[0][0]                              # learned edges
axes[1].bar((ea[:-1] + ea[1:]) / 2, ha, width=np.diff(ea) * 0.9,
            color="#2a9b5c", edgecolor="white", linewidth=0.5)
axes[1].set_title("bin_edges='auto'\n(edges fit to the data range)", fontsize=10)

for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_xlabel("classifier score")
    ax.axvspan(scores.min(), scores.max(), color="0.85", zorder=0)
axes[0].set_ylabel("normalised mass")
fig.tight_layout()