import numpy as np
import matplotlib.pyplot as plt
from mlquantify.representations import HistogramRepresentation

rng = np.random.default_rng(0)
scores = rng.beta(2, 5, size=(2000, 1))          # classifier scores in [0, 1]
y = (scores[:, 0] > 0.3).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
for ax, n in zip(axes, (5, 10, 25)):
    rep = HistogramRepresentation(bins=(n,), range=(0.0, 1.0)).fit(scores, y)
    h = rep.transform(scores)                     # normalised mass per bin
    edges = np.linspace(0, 1, n + 1)
    ax.bar((edges[:-1] + edges[1:]) / 2, h, width=(1.0 / n) * 0.9,
           color="#2a7ab9", edgecolor="white", linewidth=0.5)
    ax.set_title(f"bins = {n}", fontsize=10)
    ax.set_xlabel("classifier score")
    ax.set_xlim(0, 1)
axes[0].set_ylabel("normalised mass")
fig.tight_layout()